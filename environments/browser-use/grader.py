"""LiveWeb's existing weighted criterion and regex grading."""

import base64
import glob
import json
import logging
import os
import re
from typing import Any

from anthropic import AsyncAnthropic
from hud.settings import settings
from rubric import Criterion, Rubric
from rubric.autograders import PerCriterionGrader
from rubric.autograders.per_criterion_grader import PerCriterionOutput

logger = logging.getLogger(__name__)
SCREENSHOT_DIR = os.environ.get("LIVEWEB_EVIDENCE_DIR", "/tmp/hud-liveweb-evidence")
MAX_RESPONSE_CHARS = 200_000
VISUAL_GRADING_ENABLED = True


def load_screenshots_for_grading(max_screenshots: int = 10) -> list[dict]:
    if not VISUAL_GRADING_ENABLED or max_screenshots <= 0:
        return []
    screenshot_files = sorted(
        glob.glob(f"{SCREENSHOT_DIR}/*.png"),
        key=lambda path: (os.path.basename(path) == "final.png", os.path.getmtime(path), path),
    )
    if not screenshot_files:
        return []
    images = []
    for filepath in screenshot_files[-max_screenshots:]:
        try:
            with open(filepath, "rb") as f:
                image_data = f.read()
            images.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image_data).decode(),
                    },
                }
            )
        except OSError as e:
            logger.warning("Failed to load screenshot %s: %s", filepath, e)
    logger.info("Loaded %d screenshots for visual grading", len(images))
    return images


def truncate_response_for_grading(response: str, max_chars: int = MAX_RESPONSE_CHARS) -> str:
    if len(response) <= max_chars:
        return response
    start_chars = max_chars // 5
    end_chars = max_chars - start_chars - 100
    return (
        response[:start_chars]
        + f"\n\n... [TRUNCATED: {len(response) - max_chars} characters removed for grading] ...\n\n"
        + response[-end_chars:]
    )


def extract_verdict_from_response(text: str) -> tuple[str, str]:
    text_stripped = text.strip()
    json_match = re.search(r'\{[^{}]*"criterion_status"[^{}]*\}', text_stripped)
    if json_match:
        try:
            data = json.loads(json_match.group())
            status = str(data.get("criterion_status", "")).upper()
            if status in {"MET", "UNMET"}:
                return status, str(data.get("explanation") or data.get("reason", ""))
        except (json.JSONDecodeError, TypeError):
            pass
    try:
        data = json.loads(text_stripped)
        if isinstance(data, dict):
            status_upper = str(data.get("criterion_status") or data.get("status") or data.get("verdict", "")).upper()
            verdict = "MET" if status_upper == "MET" else "UNMET"
            explanation = data.get("explanation") or data.get("reason") or data.get("reasoning", "")
            return verdict, str(explanation)
    except (json.JSONDecodeError, TypeError):
        pass
    verdict = "UNMET"
    explanation_lines = []
    for line in text_stripped.split("\n"):
        line_upper = line.strip().upper()
        if line_upper.startswith("VERDICT:"):
            verdict_part = line_upper.replace("VERDICT:", "").strip()
            verdict = "MET" if ("MET" in verdict_part and "UNMET" not in verdict_part) else "UNMET"
        else:
            explanation_lines.append(line)
    return verdict, "\n".join(explanation_lines).strip()


async def generate_with_anthropic(
    system_prompt: str,
    user_prompt: str,
    images: list[dict] | None = None,
) -> Any:
    hud_api_key = (settings.api_key or "").strip()
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    auth_token = (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()

    if hud_api_key:
        hud_inference_url = os.environ.get("HUD_INFERENCE_URL", "https://inference.hud.ai")
        client = AsyncAnthropic(api_key=hud_api_key, base_url=hud_inference_url)
    elif api_key:
        client = AsyncAnthropic(api_key=api_key)
    elif auth_token:
        client = AsyncAnthropic(auth_token=auth_token)
    else:
        raise RuntimeError("Missing credentials: set HUD_API_KEY or ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")

    domain_hint = (
        "You may also be provided with a <domains_visited> section listing the domains the agent visited "
        "during its browsing session. Use this to verify whether the agent visited the correct source "
        "websites when the criterion requires it.\n"
    )
    if images:
        grading_prompt = (
            f"{system_prompt}\n\nYou are evaluating whether a criterion is met based on:\n"
            "1. The agent's response (the text below)\n"
            "2. Screenshots from the agent's browser session (attached)\n"
            "3. The domains the agent visited (if provided in <domains_visited>)\n\n"
            "Use both text AND visual evidence to determine if the criterion is satisfied.\n"
            f"{domain_hint}\nAfter your reasoning, you MUST end with exactly one of these two lines:\n"
            "VERDICT: MET\nVERDICT: UNMET"
        )
        content: str | list[dict[str, Any]] = [{"type": "text", "text": user_prompt}, *images]
    else:
        grading_prompt = (
            f"{system_prompt}\n\nEvaluate whether this criterion is met by the response.\n"
            f"{domain_hint}\nAfter your reasoning, you MUST end with exactly one of these two lines:\n"
            "VERDICT: MET\nVERDICT: UNMET"
        )
        content = user_prompt

    response = await client.messages.create(
        model=os.environ.get("GRADER_MODEL", "claude-opus-4-5"),
        max_tokens=1024,
        system=grading_prompt,
        messages=[{"role": "user", "content": content}],
    )
    text = response.content[0].text.strip()
    verdict, explanation = extract_verdict_from_response(text)
    if not explanation:
        explanation = text
    return PerCriterionOutput(criterion_status=verdict, explanation=explanation)


async def grade_with_regex(answer: str, regex: str) -> float:
    return 1.0 if re.search(regex, answer, re.IGNORECASE) else 0.0


async def grade_with_rubric(
    answer: str,
    rubric_items: list[dict],
    query: str,
    use_visual_grading: bool = False,
    visited_domains: list[str] | None = None,
) -> dict:
    answer = truncate_response_for_grading(answer)
    screenshots: list[dict] = []
    if use_visual_grading:
        screenshots = load_screenshots_for_grading(max_screenshots=10)

    domain_context = ""
    if visited_domains:
        domain_context = f"\n\n<domains_visited>\n{', '.join(visited_domains)}\n</domains_visited>"

    async def generate_with_images(system_prompt: str, user_prompt: str) -> Any:
        prompt_with_context = user_prompt + domain_context if domain_context else user_prompt
        return await generate_with_anthropic(system_prompt, prompt_with_context, images=screenshots)

    criteria = [Criterion(weight=item["weight"], requirement=item["requirement"]) for item in rubric_items]
    rubric_obj = Rubric(criteria)
    grader = PerCriterionGrader(generate_fn=generate_with_images)
    result = await rubric_obj.grade(query=query, to_grade=answer, autograder=grader)

    positive_weight_sum = sum(c.weight for c in result.report if c.weight > 0) or 1.0
    negative_weight_sum = abs(sum(c.weight for c in result.report if c.weight < 0))
    negative_scale = 1.0
    if negative_weight_sum >= positive_weight_sum:
        negative_scale = (positive_weight_sum * 0.9) / negative_weight_sum

    subscores: dict[str, float] = {}
    weights: dict[str, float] = {}
    report: list[dict[str, Any]] = []
    computed_score = 0.0

    for i, criterion in enumerate(result.report):
        key = f"criterion_{i + 1}"
        criterion_score = 1.0 if criterion.verdict == "MET" else 0.0
        if criterion.weight >= 0:
            normalized_weight = criterion.weight / positive_weight_sum
            subscores[key] = criterion_score
            weights[key] = round(normalized_weight, 3)
        else:
            normalized_weight = (criterion.weight * negative_scale) / positive_weight_sum
        computed_score += criterion_score * normalized_weight
        report.append(
            {
                "criterion": key,
                "requirement": criterion.requirement,
                "verdict": criterion.verdict,
                "reason": criterion.reason,
                "raw_weight": criterion.weight,
                "weight": round(normalized_weight, 3),
                "score": criterion_score,
            }
        )

    return {
        "score": round(max(0.0, min(1.0, computed_score)), 3),
        "subscores": subscores,
        "weights": weights,
        "report": report,
    }


def build_regex_pattern(patterns: list[str]) -> str:
    flags: set[str] = set()
    clean_patterns: list[str] = []
    for pattern in patterns:
        match = re.match(r"^\(\?([imsx]+)\)", pattern)
        if match:
            flags.update(match.group(1))
            clean_patterns.append(pattern[match.end() :])
        else:
            clean_patterns.append(pattern)
    combined = "|".join(clean_patterns)
    if flags:
        return f"(?{''.join(sorted(flags))})({combined})"
    return combined
