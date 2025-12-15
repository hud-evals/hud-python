"""Task loading and setup tools for tau2-bench."""

from typing import Dict, Any, Optional, Callable, List
from tau2.environment.environment import Environment
from tau2.data_model.tasks import Task
from tau2.registry import registry
from task import Tau2Task, _format_system_prompt
from server.setup import setup

# Global task state
tau2_task = Tau2Task()


@setup.tool("load")
async def load(
    domain: str,
    task_id: str,
    task_split: Optional[str] = None,
    solo_mode: bool = False,
    start_conversation: bool = False,
    initial_greeting: str = "Hi! How can I help you today?"
) -> Dict[str, Any]:
    """
    Complete setup: load domain, set task, and initialize environment.

    Args:
        domain: The domain to use (airline, retail, telecom, or mock)
        task_id: The task ID to load
        task_split: Optional task split name (e.g., "base", "train", "test")
        solo_mode: Whether to run in solo mode (default: False)
        start_conversation: If True and solo_mode=False, send initial greeting (default: False)
        initial_greeting: The greeting to send if start_conversation=True

    Returns:
        Setup status with task and environment information, plus user's first response if start_conversation=True
    """
    try:
        # 1. Load tasks for the domain
        task_loader = registry.get_tasks_loader(domain)
        tasks: List[Task] = task_loader(task_split_name=task_split)
        tau2_task.domain = domain
        tau2_task.tasks = tasks

        # 2. Set the specific task
        success = tau2_task.set_task(task_id)
        if not success:
            return {
                "error": f"Task {task_id} not found",
                "available_tasks": [t.id for t in tasks],
            }

        # 3. Initialize environment
        env_constructor: Callable[[], Environment] = registry.get_env_constructor(domain)
        tau2_task.environment = env_constructor(solo_mode=solo_mode)
        tau2_task.solo_mode = solo_mode

        # 4. Apply initial state (matches upstream Environment/Orchestrator semantics)
        if tau2_task.task.initial_state is not None:
            initialization_data = tau2_task.task.initial_state.initialization_data
            initialization_actions = tau2_task.task.initial_state.initialization_actions

            if initialization_data is not None:
                if initialization_data.agent_data is not None:
                    tau2_task.environment.tools.update_db(initialization_data.agent_data)
                if (
                    initialization_data.user_data is not None
                    and getattr(tau2_task.environment, "user_tools", None) is not None
                ):
                    tau2_task.environment.user_tools.update_db(initialization_data.user_data)

            if initialization_actions is not None:
                for action in initialization_actions:
                    tau2_task.environment.run_env_function_call(action)

        # 5. Get policy + format HUD system prompt
        policy = tau2_task.get_policy()
        system_prompt = _format_system_prompt(policy, solo_mode)

        result = {
            "status": "ready",
            "domain": domain,
            "initial_greeting": initial_greeting,
            "initial_response": None
        }

        # 6. Initialize conversation tool for multi-turn mode
        if not solo_mode:
            from server.tools.conversation import ConversationTool

            ConversationTool.initialize_global(tau2_task)
            # result["conversation_initialized"] = True

            if start_conversation:
                from server.main import get_conversation_tool

                conversation_tool = get_conversation_tool()
                if conversation_tool:
                    response = await conversation_tool(initial_greeting)
                    # result["conversation_started"] = True
                    result["initial_response"] = response[0].text if response else "No response"

        return result

    except Exception as e:
        import traceback

        return {"error": f"Setup failed: {str(e)}", "traceback": traceback.format_exc()}


def get_tau2_task() -> Tau2Task:
    """Internal helper to get the global Tau2Task instance."""
    return tau2_task