""" online=mind2web evaluators."""

import logging
import math
import httpx
from hud.tools.types import EvaluationResult
from . import evaluate
from fastmcp import Context  # for type annotations
import os, json
from hud.tools.types import EvaluationResult
from controller import mcp, http_client
from hud.tools import playwright

playwright_tool = None  # Store playwright tool globally for launch_app

logger = logging.getLogger(__name__)

@evaluate.tool("vlm_judge")
async def vlm_judge(
    ctx: Context,
    task_description: str,
    website: str = None,
    original_task_data: dict = None
) -> dict:
    """Evaluate task completion using GPT-4.1 VLM

    Args:
        task_description: The task to evaluate
        website: Target website URL
        original_task_data: Original task metadata

    Returns:
        Evaluation result with passes=True/False
    """
    await ctx.info(f"VLM evaluation for task: {task_description[:50]}...")

    try:
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            await ctx.error("OPENAI_API_KEY environment variable not set")
            return {
                "passes": False,
                "error": "OPENAI_API_KEY not set",
                "hint": "Ensure OPENAI_API_KEY is provided in task configuration"
            }

        # Check if playwright tool is available
        if not playwright_tool or not playwright_tool.page:
            await ctx.error("Browser/page not available for screenshot")
            return {"passes": False, "error": "Browser not available"}

        # Take screenshot
        await ctx.info("Taking screenshot for VLM evaluation...")
        screenshot_bytes = await playwright_tool.page.screenshot()

        import base64
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        # Get page content
        try:
            page_content = await playwright_tool.page.evaluate('document.body.innerText')
        except Exception as e:
            page_content = f"Could not extract page content: {e}"

        # Create evaluation prompt
        prompt = f"""Evaluate whether this web task was completed successfully.

        Task: {task_description}
        Website: {website or "Not specified"}

        Page content summary: {page_content[:500] + "..." if len(page_content) > 500 else page_content}

        Based on the screenshot and page content, determine if the task was completed successfully.
        Look for evidence that the specific task requirements were met.

        Respond with ONLY "true" or "false" - no other text."""

        # Call GPT-4.1 API
        await ctx.info("Calling GPT-4.1 for evaluation...")
        import openai

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4.1
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
                ]
            }],
            temperature=0.0,
            max_tokens=10
        )

        # Parse result
        result_text = response.choices[0].message.content.strip().lower()
        passes = result_text == "true"

        await ctx.info(f"VLM evaluation result: {passes} (response: '{result_text}')")

        return {
            "passes": passes,
            "vlm_response": result_text,
            "task_description": task_description,
            "website": website,
            "model": "gpt-4o",
            "original_task_data": original_task_data
        }

    except Exception as e:
        await ctx.error(f"VLM evaluation failed: {e}")
        return {
            "passes": False,
            "error": str(e),
            "task_description": task_description
        }



# @evaluate.tool("vlm_judge_by_screenshot")
@mcp.tool
async def vlm_judge_by_screenshot(
    ctx: Context,
    task_description: dict | str,
) -> dict:
    logging.info((task_description))
    if type(task_description) == str: task_description = json.loads(task_description)
    try:
        # check vlm(open ai) api key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None:
            logging.error("OPENAI_API_KEY environment variable not set")
            return EvaluationResult(
                reward=0.,
                done=False,
                content="OPENAI_API_KEY environment variable not set"
            )


        # check the browser
        # from ..tools import playwright as playwright_tool
        if not playwright_tool or not playwright_tool.page:
            logging.error("Browser/page not available")
            return EvaluationResult(
                reward=0.,
                done=False, 
                content="Browser/page not available"
            )
        
        # get page content
        try:
            page_content = await playwright_tool.page.evaluate('document.body.innerText')
        except Exception as e:
            page_content = f"Could not extract page content: {e}"
            
        logging.info("Taking screenshot for VLM evaluation...")
        
        # take screen shot
        screenshot_bytes = await playwright_tool.page.screenshot()
        import base64
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        # Create evaluation prompt
        prompt = f"""Evaluate whether this web task was completed successfully.

        Task: {task_description["confirmed_task"]}
        Website: {task_description["website"]}

        Page content summary: {page_content[:500] + "..." if len(page_content) > 500 else page_content}

        Based on the screenshot and page content, determine if the task was completed successfully.
        Look for evidence that the specific task requirements were met.

        Respond with ONLY "true" or "false" - no other text."""
        logging.info("Calling GPT-4.1 for evaluation...")
        import openai

        logging.info(f"Creating OpenAI client with api_key: {openai_api_key[:10]}...")

        # Check for any environment variables that might affect OpenAI client
        import os
        proxy_related_vars = {k: v for k, v in os.environ.items() if 'proxy' in k.lower() or 'http' in k.lower()}
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
        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4.1
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
                ]
            }],
            temperature=0.0,
            max_tokens=10
        )
        # Parse result
        result_text = response.choices[0].message.content.strip().lower()
        passes = result_text == "true"

        logging.info(f"VLM evaluation result: {passes} (response: '{result_text}')")

        return EvaluationResult(
                reward=0.,
                done=False, 
                content=response,
                info=task_description
            )
        
    except Exception as e:
        logging.error(f"VLM evaluation failed: {e}")
        return {
            "passes": False,
            "error": str(e),
            "task_description": task_description["confirmed_task"]
        }
