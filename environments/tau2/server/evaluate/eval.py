"""Task evaluation tools for tau2-bench."""

from typing import Dict, Any
import uuid
from hud.tools.types import EvaluationResult
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
from tau2.utils.utils import get_now
from server.setup.load import get_tau2_task

from . import evaluate


@evaluate.tool("evaluate_task")
async def evaluate_task(evaluation_type: str = "all") -> EvaluationResult:
    """
    Evaluate the current task based on the conversation history.

    Args:
        evaluation_type: Type of evaluation to run
            - "env": Environment state evaluation
            - "action": Action completion evaluation
            - "communicate": Communication evaluation
            - "nl_assertions": Natural language assertions
            - "all": All evaluation types combined (default)

    Returns:
        EvaluationResult with reward, done status, and feedback
    """
    tau2_task = get_tau2_task()

    if not tau2_task.is_initialized():
        return EvaluationResult(
            reward=0.0,
            done=True,
            content="Error: Environment not initialized. Call setup_task first.",
            isError=True,
        )

    try:
        assert tau2_task.task is not None
        assert tau2_task.domain is not None

        # Map string to EvaluationType enum
        eval_type_map = {
            "env": EvaluationType.ENV,
            "action": EvaluationType.ACTION,
            "communicate": EvaluationType.COMMUNICATE,
            "nl_assertions": EvaluationType.NL_ASSERTIONS,
            "all": EvaluationType.ALL,
            "all_with_nl": EvaluationType.ALL_WITH_NL_ASSERTIONS,
        }

        eval_type = eval_type_map.get(evaluation_type.lower(), EvaluationType.ALL)

        # Create SimulationRun from current state with all required fields
        current_time = get_now()
        simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=tau2_task.task.id,
            start_time=current_time,
            end_time=current_time,
            duration=0.0,
            messages=tau2_task.messages,
            termination_reason=TerminationReason.AGENT_STOP,  # Assume agent stopped
        )

        # Run evaluation
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=tau2_task.task,
            evaluation_type=eval_type,
            solo_mode=tau2_task.solo_mode,
            domain=tau2_task.domain,
        )

        # Update simulation with reward info
        simulation.reward_info = reward_info

        # Format detailed feedback
        feedback_lines = [f"Evaluation complete. Reward: {reward_info.reward:.2f}"]

        # Add reward breakdown if available
        if reward_info.reward_breakdown:
            feedback_lines.append("\nReward Breakdown:")
            for reward_type, value in reward_info.reward_breakdown.items():
                feedback_lines.append(f"  - {reward_type.value}: {value:.2f}")

        # Add database check results (only when DB is part of scoring).
        # In multi-turn tasks with user tools, the user DB can legitimately change during troubleshooting,
        # and upstream env evaluator's db_check compares both agent+user DBs even when DB isn't scored.
        if (
            reward_info.db_check
            and reward_info.reward_basis
            and RewardType.DB in reward_info.reward_basis
        ):
            feedback_lines.append(
                f"\nDatabase Check: {'✓ PASS' if reward_info.db_check.db_match else '✗ FAIL'}"
            )

        # Add action check results
        if reward_info.action_checks:
            feedback_lines.append(f"\nAction Checks ({len(reward_info.action_checks)} total):")
            for check in reward_info.action_checks:
                status = "✓" if check.action_match else "✗"
                feedback_lines.append(f"  {status} {check.action.get_func_format()}")

        # Add communication check results
        if reward_info.communicate_checks:
            feedback_lines.append(
                f"\nCommunication Checks ({len(reward_info.communicate_checks)} total):"
            )
            for check in reward_info.communicate_checks:
                status = "✓" if check.met else "✗"
                feedback_lines.append(f"  {status} '{check.info}'")

        # Add environment assertion results
        if reward_info.env_assertions:
            feedback_lines.append(
                f"\nEnvironment Assertions ({len(reward_info.env_assertions)} total):"
            )
            for check in reward_info.env_assertions:
                status = "✓" if check.met else "✗"
                desc = (
                    check.env_assertion.message
                    or f"{check.env_assertion.func_name}({check.env_assertion.arguments})"
                )
                feedback_lines.append(f"  {status} {desc}")

        # Add additional info if available
        if reward_info.info:
            feedback_lines.append(f"\nAdditional Info: {reward_info.info}")

        # Convert to EvaluationResult
        return EvaluationResult(
            reward=reward_info.reward,
            done=True,
            content="\n".join(feedback_lines),
            info=get_evaluation_criteria(),
            isError=False,
        )

    except Exception as e:
        return EvaluationResult(
            reward=0.0,
            done=True,
            content=f"Evaluation error: {str(e)}",
            info=get_evaluation_criteria(),
            isError=True,
        )


# @evaluate.tool("get_evaluation_criteria")
def get_evaluation_criteria() -> Dict[str, Any]:
    """
    Get the evaluation criteria for the current task.

    Returns:
        Dictionary with evaluation criteria details
    """
    tau2_task = get_tau2_task()

    if tau2_task.task is None:
        return {"error": "No task set. Call setup_task first."}

    if tau2_task.task.evaluation_criteria is None:
        return {"message": "No evaluation criteria defined for this task"}

    criteria = tau2_task.task.evaluation_criteria

    return {
        "reward_basis": [basis.value for basis in criteria.reward_basis]
        if criteria.reward_basis
        else [],
        "actions": [action.model_dump() for action in criteria.actions] if criteria.actions else [],
        "communicate_info": criteria.communicate_info if criteria.communicate_info else [],
        "nl_assertions": criteria.nl_assertions if criteria.nl_assertions else [],
    }
