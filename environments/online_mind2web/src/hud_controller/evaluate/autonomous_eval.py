"""online=mind2web evaluators."""

import os, json, logging
from hud.tools.types import EvaluationResult
from . import evaluate
from fastmcp import Context

logger = logging.getLogger(__name__)


# @evaluate.tool("autonomous_eval")
async def autonomous_eval(
    ctx: Context,
    task_description: dict | str,
) -> dict | EvaluationResult:
    logging.info((task_description))
    if type(task_description) == str:
        task_description = json.loads(task_description)
    try:
        # check openai api key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            logging.error("OPENAI_API_KEY environment variable not set")
            return EvaluationResult(
                reward=0.0,
                done=False,
                info={"error": "OPENAI_API_KEY environment variable not set"},
                isError=True,
            )

        persistent_ctx = evaluate.env
        playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
        if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
            logger.error("No browser page available")
            return EvaluationResult(
                reward=0.0,
                done=False,
                content="No browser page available",
                info={"error": "No browser page available"},
                isError=True,
            )

        # Load action history from file
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

        # Get last 10 actions for evaluation
        last_actions = action_history[-10:] if action_history else []

        logging.info("Taking screenshot for VLM evaluation...")

        # take screen shot
        screenshot_bytes = await playwright_tool.page.screenshot()
        import base64

        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        # Create evaluation prompt using Autonomous_eval structure
        system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the bot encounters an exception and respond with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches, and then return the top items. If the ordering action is missing, the task is likely to fail.
2. Site navigation: The user wants to navigate to a specific page. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
"""

        prompt = f"""User Intent: {task_description["confirmed_task"]}

Action History:
{chr(10).join(f"{i + 1}. {action}" for i, action in enumerate(last_actions))}

The last snapshot of the web page is shown in the image."""
        logging.info("Calling GPT-4.1 for evaluation...")
        import openai

        # Check for any environment variables that might affect OpenAI client
        proxy_related_vars = {
            k: v for k, v in os.environ.items() if "proxy" in k.lower() or "http" in k.lower()
        }
        if proxy_related_vars:
            logging.info(f"Found proxy-related env vars: {proxy_related_vars}")

        try:
            # Try creating client with minimal parameters to avoid 'proxies' error
            client = openai.OpenAI(api_key=openai_api_key)
            logging.info("OpenAI client created successfully")
        except Exception as e:
            logging.error(f"Failed to create OpenAI client: {e}")
            logging.error(f"OpenAI version: {openai.__version__}")
            import traceback

            logging.error(traceback.format_exc())
            raise
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=500,  # Increased for detailed thoughts and status
        )

        # Parse result according to new format
        result_text = response.choices[0].message.content.strip()

        # Extract thoughts and status
        try:
            thoughts = result_text.split("Thoughts:")[1].split("Status:")[0].strip()
            status = result_text.split("Status:")[1].strip().strip('"').lower()
            success = status == "success"
        except:
            thoughts = result_text
            success = "success" in result_text.lower()
            status = "success" if success else "failure"

        logging.info(f"Autonomous evaluation result: {status} (thoughts: {thoughts[:100]}...)")

        return EvaluationResult(
            reward=1.0 if success else 0.0,
            done=True,
            content=f"Status: {status}\nThoughts: {thoughts}",
            info={
                "status": status,
                "thoughts": thoughts,
                "task_description": task_description,
                "actions_count": len(action_history),
            },
        )

    except Exception as e:
        logging.error(f"VLM evaluation failed: {e}")
        return EvaluationResult(isError=True, info={"Exception": str(e)})
