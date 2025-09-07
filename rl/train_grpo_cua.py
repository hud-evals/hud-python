# train_grpo_cua.py
# Qwen2.5-VL + LoRA + GRPO (terminal reward), vLLM sampling via HUD GenericOpenAIChatAgent
import os, io, base64, json, time, math, random, asyncio, requests
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from openai import AsyncOpenAI

# HUD imports
import hud
from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
from hud.datasets import Task

# ---------------------------
# Config
# ---------------------------

@dataclass
class Config:
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "token-abc123"

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    )

    # Processor image budget
    min_pixels: int = 256*28*28   # ~0.2MP
    max_pixels: int = 1024*28*28  # ~1.0-1.2MP effective input size

    # GRPO
    group_size: int = 6
    clip_eps: float = 0.2
    kl_beta: float = 1e-3
    lr: float = 1e-4
    grad_accum_steps: int = 1
    epochs: int = 1                       # >1 requires storing old logprobs
    token_agg: str = "mean"               # "mean" is recommended

    # Turn weighting
    turn_weighting: str = "last"          # "last" | "last_k" | "all_discounted"
    last_k: int = 3
    gamma: float = 0.9

    # Shaping (env is terminal-only; these help separate failures)
    step_penalty: float = 0.0             # e.g., 0.01 to discourage long episodes
    format_penalty: float = 0.0           # -1.0 if action is unparsable

    # Training cadence
    episodes_per_batch: int = 48
    save_every_batches: int = 1
    out_dir: str = "./checkpoints"
    adapter_prefix: str = "cua-grpo-step"

    # Seeds
    seed: int = 1234
    
    # Dataset configuration
    dataset_name: str = "hud-evals/2048-taskset"  # HuggingFace dataset name
    dataset_split: str = "train"
    system_prompt: str = "You are an expert agent. Complete the task efficiently."
    
    # Episode collection
    max_steps_per_episode: int = 100
    parallel_episodes: int = 4  # Run episodes in parallel
    temperature: float = 0.7
    
    # Training limits
    max_training_steps: int = 1000  # Stop after this many gradient steps
    max_total_episodes: int = 10000  # Stop after collecting this many episodes

cfg = Config()

# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def b64_to_pil(b: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")

def blocks_to_images(blocks: List[Dict[str, Any]]) -> List[Image.Image]:
    """HUD ContentBlocks -> list of PIL images."""
    imgs = []
    for b in blocks:
        if b.get("type") == "image":
            data = b.get("data") or b.get("bytes")
            if isinstance(data, str):  # base64
                imgs.append(b64_to_pil(data))
            elif isinstance(data, (bytes, bytearray)):
                imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return imgs

# ---------------------------
# Model, processor, optimizer
# ---------------------------

def load_models_and_processor(cfg: Config):
    try:
        print(f"  Loading processor from {cfg.base_model}...")
        processor = AutoProcessor.from_pretrained(
            cfg.base_model, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels
        )
        print(f"  ✓ Processor loaded")
        
        print(f"  Loading policy model...")
        policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print(f"  ✓ Policy model loaded")
        
        # Wrap policy with LoRA (text tower)
        print(f"  Adding LoRA adapters (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            task_type="CAUSAL_LM", bias="none",
            target_modules=list(cfg.target_modules)
        )
        policy = get_peft_model(policy, lora_cfg)
        print(f"  ✓ LoRA adapters added")

        # Frozen SFT reference (no LoRA)
        print(f"  Loading reference model...")
        ref = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        print(f"  ✓ Reference model loaded")

        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        print(f"  ✓ Optimizer initialized (lr={cfg.lr})")
        
        return processor, policy, ref, opt
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print(f"[ERROR] Make sure you have:")
        print(f"  1. Installed transformers: pip install transformers")
        print(f"  2. Installed peft: pip install peft")
        print(f"  3. Model access: {cfg.base_model}")
        raise

# ---------------------------
# Packing / tokenization
# ---------------------------

def pack_user_turn(processor, history_msgs: List[Dict[str, Any]],
                   images: List[Image.Image], user_text: str = "Continue."):
    """Serialize messages+images for the current user turn; returns HF tensors."""
    text = processor.apply_chat_template(
        history_msgs + [{"role":"user","content":[*([{"type":"image"}]*len(images)),
                                                 {"type":"text","text":user_text}]}],
        add_generation_prompt=True, tokenize=False, add_vision_id=True
    )
    return processor(text=text, images=images, return_tensors="pt")

def build_full_ids_for_teacher_forcing(processor, history_msgs, images, assistant_text):
    """Prompt + assistant completion → tokenized full input_ids (for delta slicing)."""
    text = processor.apply_chat_template(
        history_msgs + [{"role":"assistant","content":[{"type":"text","text":assistant_text}]}],
        add_generation_prompt=False, tokenize=False, add_vision_id=True
    )
    return processor(text=text, images=images, return_tensors="pt")

def tok_logprobs_on_delta(model, inputs, completion_ids):
    """Return per-token logprobs on the assistant delta span."""
    full_ids = torch.cat([inputs.input_ids, completion_ids], dim=1)
    out = model(input_ids=full_ids.to(model.device),
                pixel_values=inputs.get("pixel_values", None).to(model.device) if inputs.get("pixel_values", None) is not None else None)
    # next-token LM: align logits with labels (shifted)
    T = completion_ids.shape[1]
    logits = out.logits[:, -T-1:-1, :]
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, completion_ids.unsqueeze(-1).to(model.device)).squeeze(-1)  # [1, T]

# ---------------------------
# GRPO loss (assistant-only, turn-sum)
# ---------------------------

def grpo_loss_on_episode(policy, _ref, sample_turns, clip_eps, kl_beta, token_agg="mean"):
    """
    sample_turns: list of dicts, each with {
        'inputs': HF batch (prompt up to user_k),
        'completion_ids': assistant delta tokens at turn k,
        'A': scalar advantage * turn_weight (already combined),
        'old_lp_tok': per-token logprob under π_old (optional if epochs==1),
        'ref_lp_tok': per-token logprob under π_ref
    }
    """
    policy_terms, kl_terms = [], []
    for s in sample_turns:
        # current log-probs on delta
        pol_tok_lp = tok_logprobs_on_delta(policy, s["inputs"], s["completion_ids"])
        # old/ref (stored or recomputed pre-update)
        old_tok_lp = s["old_lp_tok"]
        _ref_tok_lp = s["ref_lp_tok"]
        # ratios per token
        ratio_tok = torch.exp(pol_tok_lp - old_tok_lp)
        # reduce over tokens
        if token_agg == "mean":
            ratio = ratio_tok.mean()
        else:
            raise ValueError("Unsupported token_agg")
        uncl = ratio * s["A"]
        clip = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * s["A"]
        policy_terms.append(-torch.minimum(uncl, clip))

        # KL vs ref (reverse KL approx) – token-mean
        rho_tok = torch.exp(pol_tok_lp - _ref_tok_lp)
        kl_terms.append((rho_tok - torch.log(rho_tok) - 1).mean())

    return torch.stack(policy_terms).sum() + kl_beta * torch.stack(kl_terms).sum()

# ---------------------------
# Turn weighting
# ---------------------------

def turn_weights(num_turns: int, scheme: str, last_k: int, gamma: float) -> List[float]:
    if num_turns == 0: return []
    if scheme == "last":
        w = [0.0]*(num_turns-1) + [1.0]
    elif scheme == "last_k":
        w = [0.0]*num_turns
        K = min(last_k, num_turns)
        for i in range(num_turns-K, num_turns):
            w[i] = gamma**(num_turns-1 - i)
    elif scheme == "all_discounted":
        w = [gamma**(num_turns-1 - i) for i in range(num_turns)]
    else:
        raise ValueError("Unknown scheme")
    # Normalize weights so total weight is 1 (optional but helps scale)
    s = sum(w) or 1.0
    return [wi/s for wi in w]

# ---------------------------
# vLLM LoRA hot-load
# ---------------------------

def hotload_lora(adapter_name: str, adapter_path: str, base_url: str, api_key: str):
    url = f"{base_url}/load_lora_adapter"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"lora_name": adapter_name, "lora_path": adapter_path}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    r.raise_for_status()

# ---------------------------
# Training Orchestration
# ---------------------------

class ReplayBuffer:
    def __init__(self, maxlen=64):
        self.maxlen = maxlen
        self.buf = []

    def push(self, episode):
        # store only successes
        if episode["terminal_reward"] > 0:
            self.buf.append(episode)
            if len(self.buf) > self.maxlen:
                self.buf.pop(0)

    def sample_success(self):
        return random.choice(self.buf) if self.buf else None

async def train_loop(cfg: Config):
    print(f"[TRAIN] Starting training loop...")
    print(f"[TRAIN] Config: episodes_per_batch={cfg.episodes_per_batch}, max_steps={cfg.max_training_steps}")
    
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    print(f"[TRAIN] Loading models and processor...")
    processor, policy, ref, opt = load_models_and_processor(cfg)
    print(f"[TRAIN] Models loaded successfully")
    
    step = 0
    total_episodes = 0
    replay = ReplayBuffer()

    print(f"[TRAIN] Loading tasks from dataset: {cfg.dataset_name}")
    tasks = get_training_tasks()
    current_adapter = f"{cfg.adapter_prefix}-{step:05d}"

    print(f"[TRAIN] Starting main training loop...")
    while step < cfg.max_training_steps and total_episodes < cfg.max_total_episodes:
        # 1) Collect episodes with current adapter via HUD actor / vLLM
        episodes = await collect_episodes_with_hud(
            tasks=tasks,
            adapter_name=current_adapter,
            episodes_per_batch=cfg.episodes_per_batch
        )

        # Update counters and save successes to replay
        total_episodes += len(episodes)
        for ep in episodes:
            if ep.get("terminal_reward", 0.0) > 0:
                replay.push(ep)

        # 2) Build training samples (assistant turns to reward)
        #    Each episode must yield: a list of (history_msgs, obs_blocks, assistant_text) triples
        #    We'll compute delta tokens and store old/ref logprobs.
        print(f"[TRAIN] Building training samples from {len(episodes)} episodes...")
        groups: Dict[str, List[Dict[str, Any]]] = {}  # key -> list of per-episode samples
        for ep in episodes:
            turns = extract_assistant_turns(ep)  # list[{"history_msgs", "obs_blocks", "assistant_text"}]
            num_turns = len(turns)
            if num_turns == 0: continue

            # terminal reward + optional shaping (step penalty, format penalty)
            R = float(ep.get("terminal_reward", 0.0))
            if cfg.step_penalty:
                R -= cfg.step_penalty * num_turns
            if cfg.format_penalty:
                R += compute_format_penalty(ep)  # implement if you want; else return 0.0

            # choose per-turn weights
            w = turn_weights(num_turns, cfg.turn_weighting, cfg.last_k, cfg.gamma)

            # pack each rewarded turn
            per_turn = []
            for k, t in enumerate(turns):
                if w[k] == 0.0: continue
                images = blocks_to_images(t["obs_blocks"])
                # Use the user's last message text if available, otherwise default
                user_text = "Continue."
                if t["history_msgs"]:
                    last_user = [m for m in t["history_msgs"] if m.get("role") == "user"]
                    if last_user:
                        content = last_user[-1].get("content", "")
                        if isinstance(content, str):
                            user_text = content
                        elif isinstance(content, list) and content:
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    user_text = item.get("text", "Continue.")
                                    break
                inputs = pack_user_turn(processor, t["history_msgs"], images, user_text)
                # teacher-force to get delta IDs
                full = build_full_ids_for_teacher_forcing(processor, t["history_msgs"], images, t["assistant_text"])
                completion_ids = full.input_ids[:, inputs.input_ids.shape[-1]:].to(policy.device)

                with torch.no_grad():
                    old_lp_tok = tok_logprobs_on_delta(policy, inputs, completion_ids)  # π_old (pre-update)
                    ref_lp_tok = tok_logprobs_on_delta(ref,    inputs, completion_ids)  # π_ref (fixed)

                per_turn.append({
                    "inputs": {k: v.to(policy.device) for k, v in inputs.items()},
                    "completion_ids": completion_ids,
                    "weight": w[k],
                    "old_lp_tok": old_lp_tok,
                    "ref_lp_tok": ref_lp_tok
                })

            # add to a group keyed by (task_bucket or "final_turn") — simplest: one bucket
            key = episode_group_key(ep)  # e.g., ep["task_id"] bucket or "final"
            groups.setdefault(key, []).append({
                "turns": per_turn,
                "R": R,
                "episode": ep
            })

        # 3) For each group: compute GRPO advantages, handle all-zero with replay, do an update
        print(f"[TRAIN] Computing GRPO updates for {len(groups)} groups...")
        for key, samples in groups.items():
            print(f"  Processing group '{key}' with {len(samples)} samples...")
            # Inject 1 success if all-zero and we have replay
            if all(s["R"] <= 0.0 for s in samples):
                suc = replay.sample_success()
                if suc:
                    # Extract turns from the success episode
                    suc_turns = extract_assistant_turns(suc)
                    if suc_turns:
                        # Build per-turn samples from replay
                        num_turns = len(suc_turns)
                        w = turn_weights(num_turns, cfg.turn_weighting, cfg.last_k, cfg.gamma)
                        
                        per_turn = []
                        for k, t in enumerate(suc_turns):
                            if w[k] == 0.0: continue
                            
                            images = blocks_to_images(t["obs_blocks"])
                            inputs = pack_user_turn(processor, t["history_msgs"], images, user_text="Continue.")
                            full = build_full_ids_for_teacher_forcing(processor, t["history_msgs"], images, t["assistant_text"])
                            completion_ids = full.input_ids[:, inputs.input_ids.shape[-1]:].to(policy.device)
                            
                            if completion_ids.numel() > 0:  # Skip empty completions
                                with torch.no_grad():
                                    old_lp_tok = tok_logprobs_on_delta(policy, inputs, completion_ids)
                                    ref_lp_tok = tok_logprobs_on_delta(ref, inputs, completion_ids)
                                
                                per_turn.append({
                                    "inputs": {k: v.to(policy.device) for k, v in inputs.items()},
                                    "completion_ids": completion_ids,
                                    "weight": w[k],
                                    "old_lp_tok": old_lp_tok,
                                    "ref_lp_tok": ref_lp_tok
                                })
                        
                        if per_turn:
                            samples.append({
                                "turns": per_turn,
                                "R": 1.0,
                                "episode": suc
                            })
                            print(f"[INFO] Injected success replay for group '{key}'")

            Rs = torch.tensor([s["R"] for s in samples], device=policy.device, dtype=torch.float32)
            mu, sigma = Rs.mean(), Rs.std()
            if sigma < 1e-6:
                # still no variance → skip this group
                continue

            for _ in range(cfg.epochs):
                opt.zero_grad()
                loss_terms = []
                for R_i, s in zip(Rs, samples):
                    # advantage for this episode:
                    A = (R_i - mu) / (sigma + 1e-6)
                    # build per-turn items with combined weight
                    per_turn_items = []
                    for t in s["turns"]:
                        if t["completion_ids"].numel() == 0:  # empty delta (rare)
                            continue
                        per_turn_items.append({
                            "inputs": t["inputs"],
                            "completion_ids": t["completion_ids"],
                            "A": A * t["weight"],
                            "old_lp_tok": t["old_lp_tok"],
                            "ref_lp_tok": t["ref_lp_tok"],
                        })
                    if not per_turn_items:
                        continue
                    loss_i = grpo_loss_on_episode(
                        policy, ref, per_turn_items,
                        clip_eps=cfg.clip_eps, kl_beta=cfg.kl_beta, token_agg=cfg.token_agg
                    )
                    loss_terms.append(loss_i)

                if not loss_terms:
                    continue

                loss = torch.stack(loss_terms).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

        # 4) Save adapter and hot-load to vLLM
        step += 1
        if step % cfg.save_every_batches == 1 or cfg.save_every_batches == 1:
            save_dir = os.path.join(cfg.out_dir, f"{cfg.adapter_prefix}-{step:05d}")
            ensure_dir(save_dir)
            policy.save_pretrained(save_dir)
            # hot-load
            try:
                hotload_lora(
                    adapter_name=f"{cfg.adapter_prefix}-{step:05d}",
                    adapter_path=save_dir,
                    base_url=cfg.vllm_base_url,
                    api_key=cfg.vllm_api_key
                )
                current_adapter = f"{cfg.adapter_prefix}-{step:05d}"
                print(f"[vLLM] Loaded adapter {current_adapter}")
            except Exception as e:
                print(f"[WARN] Failed to hot-load LoRA: {e}")
        
        # Log progress
        print(f"[INFO] Step {step}, Total episodes: {total_episodes}")
    
    print(f"[INFO] Training completed. Final step: {step}, Total episodes: {total_episodes}")

# ---------------------------
# HUD glue
# ---------------------------

def get_training_tasks() -> List[Task]:
    """
    Load training tasks from HuggingFace dataset.
    Returns list of Task objects ready for execution.
    """
    print(f"[INFO] Loading dataset: {cfg.dataset_name}")
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    
    tasks = []
    for item in dataset:
        # Each item in the dataset is a dict-like object with Task fields
        task = Task(
            id=item.get("id") if hasattr(item, "get") else None,
            prompt=item["prompt"],
            mcp_config=item["mcp_config"],
            setup_tool=item.get("setup_tool") if hasattr(item, "get") else None,
            evaluate_tool=item.get("evaluate_tool") if hasattr(item, "get") else None,
            system_prompt=item.get("system_prompt", cfg.system_prompt) if hasattr(item, "get") else cfg.system_prompt,
            metadata=item.get("metadata", {}) if hasattr(item, "get") else {}
        )
        tasks.append(task)
    
    print(f"[INFO] Loaded {len(tasks)} tasks from dataset")
    return tasks

async def collect_episodes_with_hud(tasks: List[Task], adapter_name: str, episodes_per_batch: int) -> List[Dict[str, Any]]:
    """
    Collect episodes using GenericOpenAIChatAgent with vLLM backend.
    Uses the agent's conversation_history directly for turn extraction.
    """
    # Setup OpenAI client pointing to vLLM
    openai_client = AsyncOpenAI(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
    )
    
    episodes = []
    
    # Run episodes in parallel batches
    for batch_start in range(0, episodes_per_batch, cfg.parallel_episodes):
        batch_size = min(cfg.parallel_episodes, episodes_per_batch - batch_start)
        batch_tasks = []
        
        for _ in range(batch_size):
            # Sample a random task
            task = random.choice(tasks)
            
            # Create agent with current adapter
            agent = GenericOpenAIChatAgent(
                openai_client=openai_client,
                model_name=adapter_name,  # This tells vLLM which LoRA adapter to use
                parallel_tool_calls=False,
                append_setup_output=False,
                system_prompt=task.system_prompt or cfg.system_prompt,
                completion_kwargs={
                    "temperature": cfg.temperature,
                    "max_tokens": 512,
                }
            )
            
            # Create async task to run episode
            async def run_single_episode(task: Task, agent: GenericOpenAIChatAgent):
                try:
                    # Run the episode
                    with hud.trace(f"Episode {task.id or 'unknown'}"):
                        result = await agent.run(
                            task,
                            max_steps=cfg.max_steps_per_episode
                        )
                    
                    # Extract conversation history directly from agent
                    conversation_history = agent.conversation_history if hasattr(agent, 'conversation_history') else []
                    
                    # Pre-extract turns from conversation history
                    turns = extract_turns_from_conversation(conversation_history)
                    
                    return {
                        "task_id": task.id or "unknown",
                        "task": task,
                        "terminal_reward": float(result.reward),
                        "conversation_history": conversation_history,
                        "turns": turns,
                        "info": result.info if hasattr(result, 'info') else {},
                        "metadata": {
                            "adapter": adapter_name,
                            "steps": len(turns),
                        }
                    }
                except Exception as e:
                    print(f"[ERROR] Episode failed: {e}")
                    return {
                        "task_id": task.id or "unknown",
                        "task": task,
                        "terminal_reward": 0.0,
                        "conversation_history": [],
                        "turns": [],
                        "info": {"error": str(e)},
                        "metadata": {"adapter": adapter_name}
                    }
            
            batch_tasks.append(run_single_episode(task, agent))
        
        # Run batch in parallel
        batch_results = await asyncio.gather(*batch_tasks)
        episodes.extend(batch_results)
    
    # Log statistics
    successes = sum(1 for ep in episodes if ep["terminal_reward"] > 0)
    print(f"[INFO] Collected {len(episodes)} episodes: {successes} successes, {len(episodes)-successes} failures")
    
    return episodes


def extract_turns_from_conversation(conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract assistant turns from OpenAI-format conversation history.
    Each turn includes the history up to that point, any images, and the assistant's response.
    """
    turns = []
    
    for i, msg in enumerate(conversation_history):
        if msg.get("role") == "assistant":
            # Get history up to (but not including) this assistant message
            history_msgs = conversation_history[:i]
            
            # Extract any images from the preceding user message
            obs_blocks = []
            if i > 0 and conversation_history[i-1].get("role") == "user":
                user_content = conversation_history[i-1].get("content", [])
                if isinstance(user_content, list):
                    for item in user_content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            # Extract base64 image data
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "")
                            if url.startswith("data:image"):
                                # Extract base64 data after the comma
                                data = url.split(",", 1)[1] if "," in url else url
                                obs_blocks.append({
                                    "type": "image",
                                    "data": data
                                })
            
            # Extract assistant's response text
            assistant_text = ""
            if msg.get("content"):
                assistant_text = msg["content"]
            elif msg.get("tool_calls"):
                # Format tool calls as text
                tool_texts = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        tool_texts.append(f"Tool: {func.get('name')}({func.get('arguments', '{}')})") 
                    else:
                        # Handle OpenAI tool call objects
                        tool_texts.append(f"Tool: {tc.function.name}({tc.function.arguments})")
                assistant_text = "\n".join(tool_texts)
            
            if assistant_text:
                turns.append({
                    "history_msgs": history_msgs,
                    "obs_blocks": obs_blocks,
                    "assistant_text": assistant_text
                })
    
    return turns

def extract_assistant_turns(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract assistant turns from episode.
    Returns list of dicts with history_msgs, obs_blocks, and assistant_text.
    """
    # If already pre-extracted, return them
    if "turns" in episode and episode["turns"]:
        return episode["turns"]
    
    # Otherwise extract from conversation_history
    if "conversation_history" in episode:
        return extract_turns_from_conversation(episode["conversation_history"])
    
    return []

def episode_group_key(episode: Dict[str, Any]) -> str:
    """
    Group episodes by task characteristics for variance reduction.
    Can be customized based on task metadata.
    """
    # Use task metadata if available
    if "task" in episode and hasattr(episode["task"], "metadata"):
        metadata = episode["task"].metadata
        # Group by task type if specified
        if "task_type" in metadata:
            return metadata["task_type"]
        # Group by difficulty if specified
        if "difficulty" in metadata:
            return f"difficulty_{metadata['difficulty']}"
    
    # Default: single group for all episodes
    return "all"

def compute_format_penalty(episode: Dict[str, Any]) -> float:
    """
    Penalize episodes with errors to encourage proper formatting.
    """
    penalty = 0.0
    
    # Check for errors in episode info
    if "info" in episode:
        info = episode["info"]
        if "error" in info:
            penalty -= cfg.format_penalty
    
    # Check conversation history for tool errors
    if "conversation_history" in episode:
        for msg in episode["conversation_history"]:
            # Check for tool result messages that indicate errors
            if msg.get("role") == "tool" and "error" in msg.get("content", "").lower():
                penalty -= cfg.format_penalty * 0.5
    
    return penalty

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GRPO training for Qwen2.5-VL with HUD")
    parser.add_argument("--test", action="store_true", help="Run in test mode with minimal iterations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.test:
        print("[TEST MODE] Running with minimal configuration for testing")
        cfg.episodes_per_batch = 2
        cfg.parallel_episodes = 1
        cfg.max_training_steps = 1
        cfg.max_total_episodes = 2
        cfg.save_every_batches = 1
        cfg.group_size = 2
        cfg.epochs = 1
        # Use a smaller test dataset if available
        if "test" in cfg.dataset_name:
            pass  # Already using test dataset
        else:
            print(f"[TEST MODE] Using dataset: {cfg.dataset_name}")
    
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("[DEBUG] Debug logging enabled")
    
    asyncio.run(train_loop(cfg))
