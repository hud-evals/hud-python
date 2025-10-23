"""online-mind2web evaluators webjudge"""

""" reference: https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/main/src/methods/webjudge_online_mind2web.py """

import os, json, logging, base64, re
import openai
from typing import Optional
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)

MAX_IMAGE = 50  # Maximum screenshot of history to judge


@evaluate.tool("webjudge")
async def webjudge(ctx: Context, task_description: dict | str):
    return await webjudge_eval(ctx, task_description)


async def identify_key_point(task_description: dict | str) -> dict:
    """Identify key points in a task description using GPT-4.

    Args:
        task_description: The task to analyze (dict or JSON string)

    Returns:
        Dict containing the identified key points
    """

    if type(task_description) == str:
        task_description = json.loads(task_description)

    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Debug what we get from environment
    logging.info(f"DEBUG: Raw environment variable type: {type(openai_api_key)}")
    if openai_api_key:
        logging.info(
            f"DEBUG: Raw key repr: {repr(openai_api_key[:10])}"
        )  # Show first 10 chars with repr to see any weird characters
    if openai_api_key is None:
        logging.error("OPENAI_API_KEY environment variable not set")
        return {"success": False, "error": "OPENAI_API_KEY environment variable not set"}

    try:
        logging.info("Webjudge evaluation: identify_key_point")

        # Extract task text
        task_text = (
            task_description.get("confirmed_task", str(task_description))
            if isinstance(task_description, dict)
            else str(task_description)
        )

        system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""

        user_prompt = f"Task: {task_text}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]

        # Debug the actual API key being used
        logging.info(
            f"DEBUG: Creating OpenAI client with key: {openai_api_key[:10]}...{openai_api_key[-10:]}"
        )
        logging.info(f"DEBUG: Full API key length: {len(openai_api_key)}")

        # Check if the key looks valid (should start with sk-)
        if not openai_api_key.startswith("sk-"):
            logging.error(f"DEBUG: API key doesn't start with 'sk-': {openai_api_key[:10]}")

        client = openai.OpenAI(api_key=openai_api_key)
        logging.info("DEBUG: OpenAI client created successfully")

        # Log the request we're about to make
        logging.info(f"DEBUG: Making request to model: gpt-4o")
        logging.info(f"DEBUG: Message count: {len(messages)}")

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=500,  # Increased for key points list
            )
            logging.info("DEBUG: API call completed successfully")
        except Exception as api_error:
            logging.error(f"DEBUG: API call failed with error: {api_error}")
            logging.error(f"DEBUG: Error type: {type(api_error)}")
            if hasattr(api_error, "response"):
                logging.error(f"DEBUG: Error response: {api_error.response}")
            # Re-check the environment variable at the moment of failure
            current_key = os.getenv("OPENAI_API_KEY")
            logging.error(
                f"DEBUG: Env var at failure - length: {len(current_key) if current_key else 'None'}"
            )
            if current_key:
                logging.error(
                    f"DEBUG: Env var at failure: {current_key[:20]}...{current_key[-10:]}"
                )
            raise

        # Parse the response
        key_points_text = response.choices[0].message.content.strip()

        logging.info(f"Identified key points: {key_points_text}")

        return {
            "success": True,
            "key_points": key_points_text,
            "task_description": task_text,
            "model": "gpt-4o",
        }

    except Exception as e:
        logging.error(f"Key point identification failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_description": task_text if "task_text" in locals() else str(task_description),
        }


async def judge_image(
    base64_images: list | str, task_description: dict | str, key_points: str
) -> dict:
    """Judge image(s) for task completion using GPT-4V

    Args:
        base64_images: List of base64 encoded images or single base64 string
        task_description: The task to evaluate (dict or JSON string)
        key_points: Key points for task completion

    Returns:
        Dict containing evaluation results with reasoning and scores
    """

    if type(task_description) == str:
        task_description = json.loads(task_description)

    if type(base64_images) == str:
        base64_images = [base64_images]

    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        logging.error("OPENAI_API_KEY environment variable not set")
        return {"success": False, "error": "OPENAI_API_KEY environment variable not set"}

    try:
        logging.info(f"Judging {len(base64_images)} images for task completion")

        # Extract task text
        task_text = (
            task_description.get("confirmed_task", str(task_description))
            if isinstance(task_description, dict)
            else str(task_description)
        )

        system_msg = """You are an expert evaluator tasked with determining whether the provided images contain information about the necessary steps to complete a task.

**Objective**: Analyze the provided images and decide if they show essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of each image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the images and evaluate whether they contain necessary steps or evidence crucial to task completion:
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.
- Do the images show actions, progress indicators, or critical information directly related to completing the task?
- Is this information indispensable for understanding or ensuring task success?
- If the images contain partial but relevant information, consider their usefulness rather than dismissing them outright.
- Consider the progression across multiple images if provided.

3. Provide your response in the following format:
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the images that indicate necessary steps, evidence, or lack thereof.
- **Score**: Assign a score based on the reasoning, using the following scale:
    - **1**: The images do not contain any necessary steps or relevant information.
    - **2**: The images contain minimal or ambiguous information, unlikely to be essential.
    - **3**: The images include some relevant steps or hints but lack clarity or completeness.
    - **4**: The images contain important steps or evidence that are highly relevant but not fully comprehensive.
    - **5**: The images clearly display necessary steps or evidence crucial for completing the task.

Respond with:
1. **Reasoning**: [Your detailed explanation]
2. **Score**: [1-5]"""

        prompt = f"""**Task**: {task_text}

**Key Points for Task Completion**: {key_points}

The snapshots of the web page progression are shown in the images below."""

        # Create message content with text and images
        message_content = [{"type": "text", "text": prompt}]

        for base64_img in base64_images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}", "detail": "high"},
                }  # type: ignore
            )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": message_content},
        ]

        client = openai.OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4V
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
        )

        # Parse the response
        result_text = response.choices[0].message.content.strip()

        logging.info(f"Image judgment result: {result_text[:200]}...")

        return {
            "success": True,
            "judgment": result_text,
            "task_description": task_text,
            "key_points": key_points,
            "images_processed": len(base64_images),
            "model": "gpt-4o",
        }

    except Exception as e:
        logging.error(f"Image judgment failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_description": task_text if "task_text" in locals() else str(task_description),
            "images_processed": len(base64_images) if "base64_images" in locals() else 0,
        }


async def webjudge_eval(ctx: Context, task_description: dict | str, score_threshold: int = 3):
    """WebJudge Online Mind2Web evaluation using screenshot history and action history

    Args:
        task_description: Task description (dict or JSON string)
        score_threshold: Minimum score threshold for image filtering (1-5)

    Returns:
        Dict containing evaluation results with success/failure status
    """
    if type(task_description) == str:
        task_description = json.loads(task_description)

    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        logging.error("OPENAI_API_KEY environment variable not set")
        return EvaluationResult(
            isError=True, info={"Exception": f"OPENAI_API_KEY environment variable not set"}
        )

    try:
        logging.info("Starting WebJudge Online Mind2Web evaluation")

        # Extract task text
        task_text = task_description.get("confirmed_task")

        # Get screenshots from /screenshot directory
        screenshot_dir = "/screenshot"
        screenshot_history = []

        try:
            if os.path.exists(screenshot_dir):
                # Get all PNG files sorted by modification time (newest last)
                screenshot_files = []
                for file in os.listdir(screenshot_dir):
                    if file.endswith(".png") and file.startswith("screenshot_"):
                        filepath = os.path.join(screenshot_dir, file)
                        mtime = os.path.getmtime(filepath)
                        screenshot_files.append((mtime, filepath))

                # Sort by modification time
                screenshot_files.sort(key=lambda x: x[0])

                for _, filepath in screenshot_files[-MAX_IMAGE:]:
                    try:
                        with open(filepath, "rb") as f:
                            image_data = f.read()
                            screenshot_b64 = base64.b64encode(image_data).decode("utf-8")
                            screenshot_history.append(screenshot_b64)
                    except Exception as e:
                        logging.warning(f"Failed to read screenshot {filepath}: {e}")

                logging.info(f"Loaded {len(screenshot_history)} screenshots from {screenshot_dir}")
            else:
                logging.warning(f"Screenshot directory {screenshot_dir} does not exist")

        except Exception as e:
            logging.error(f"Failed to load screenshots from {screenshot_dir}: {e}")

        if not screenshot_history:
            logging.warning("No screenshot history available")
            return EvaluationResult(
                reward=0.0,
                done=True,
                content="No screenshot avaliable",
                info={"task_description": task_text, "status": "No screenshot avaliable"},
            )

        # Get action history from file
        action_history = []
        try:
            action_history_file = "/action_history/action_history.txt"
            if os.path.exists(action_history_file):
                with open(action_history_file, "r", encoding="utf-8") as f:
                    action_history = [line.strip() for line in f if line.strip()]
                logging.info(f"Loaded {len(action_history)} actions from history file")
            else:
                logging.warning("No action history file found")
        except Exception as e:
            logging.warning(f"Failed to load action history: {e}")
            action_history = []

        # Get all actions for evaluation
        last_actions = action_history if action_history else []

        logging.info(
            f"Found {len(screenshot_history)} screenshots and {len(action_history)} actions"
        )

        # Step 1: Identify key points
        logging.info(f"Webjudge step 1: Identify key points")
        key_points_result = await identify_key_point(task_description)
        if not key_points_result.get("success"):
            logger.error(f"Key point identification failed: {key_points_result.get('error')}")
            return EvaluationResult(
                isError=True,
                info={
                    "Exception": f"Key point identification failed: {key_points_result.get('error')}"
                },
            )

        key_points = key_points_result["key_points"]

        # Clean up key points formatting
        key_points = key_points.replace("\n\n", "\n")
        try:
            if "**Key Points**:" in key_points:
                key_points = key_points.split("**Key Points**:")[1]
            elif "Key Points:" in key_points:
                key_points = key_points.split("Key Points:")[-1]
            key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
        except:
            pass

        # Step 2: Judge images using screenshot history
        logging.info(f"Webjudge step 2: Judge images using screenshot history")
        judge_result = await judge_image(
            base64_images=screenshot_history,
            task_description=task_description,
            key_points=key_points,
        )

        if not judge_result.get("success"):
            return EvaluationResult(
                isError=True,
                info={"Exception": f"Image judgment failed: {judge_result.get('error')}"},
            )

        # Parse judgment result for score
        judgment_text = judge_result["judgment"]
        pattern = r"[1-5]"
        try:
            scores = re.findall(pattern, judgment_text.split("Score")[1])
            main_score = int(scores[0]) if scores else 3
        except:
            main_score = 3  # Default score if parsing fails
        logger.info("Score: ", main_score)

        # Extract reasoning
        try:
            reasoning = (
                judgment_text.split("**Reasoning**:")[-1].strip().split("**Score**:")[0].strip()
            )
        except:
            reasoning = "Unable to extract reasoning"

        # Step 3: Final evaluation using GPT-4
        logging.info(f"Webjudge step 3: Final evaluation using GPT-4")

        system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, and analysis of important web pages, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these screenshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements, the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""

        # Prepare final evaluation content
        if main_score >= score_threshold:
            # Include high-scoring screenshots in final evaluation
            final_images = []
            for screenshot_b64 in screenshot_history:  # All screenshots
                final_images.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                            "detail": "high",
                        },
                    }
                )

            prompt_with_images = f"""User Task: {task_text}

Key Points: {key_points}

Action History:
{chr(10).join(f"{i + 1}. {action}" for i, action in enumerate(last_actions))}

Image Analysis Results:
Score: {main_score}/5
Reasoning: {reasoning}"""

            content = [{"type": "text", "text": prompt_with_images}] + final_images
        else:
            # Text-only evaluation if images don't meet threshold
            prompt_text_only = f"""User Task: {task_text}

Key Points: {key_points}

Action History:
{chr(10).join(f"{i + 1}. {action}" for i, action in enumerate(last_actions))}

Note: Screenshot analysis scored {main_score}/5, below threshold of {score_threshold}."""

            content = [{"type": "text", "text": prompt_text_only}]

        # Final evaluation
        client = openai.OpenAI(api_key=openai_api_key)

        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": content}]

        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.0, max_tokens=500
        )

        final_result = response.choices[0].message.content.strip()

        # Parse final result
        try:
            thoughts = final_result.split("Thoughts:")[1].split("Status:")[0].strip()
            status = final_result.split("Status:")[1].strip().strip('"').lower()
            success = status == "success"
        except:
            logging.info(f"Warning: Final result parsing failed: {final_result}")
            thoughts = final_result
            success = "success" in final_result.lower()
            status = "success" if success else "failure"

        logging.info(f"WebJudge evaluation result: {status}")

        return EvaluationResult(
            reward=1.0 if success else 0.0,
            done=True,
            content=final_result,
            info={"task_description": task_text, "status": status, "thoughts": thoughts},
            isError=False,
        )

    except Exception as e:
        logging.error(f"WebJudge evaluation failed: {e}")
        return EvaluationResult(isError=True, info={"Exception": str(e)})
