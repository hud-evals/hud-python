#!/usr/bin/env python3
"""Pre-release evaluation check script.

This script runs the hud-evals/test-diverse taskset with Claude and validates:
1. All tasks complete without errors
2. At least one task achieves a non-zero score
3. Overall success rate meets minimum threshold
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from hud.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class PreReleaseChecker:
    """Handles pre-release evaluation checks."""

    def __init__(
        self, dataset: str = "hud-evals/test-diverse", min_success_rate: float = 25.0
    ) -> None:
        self.dataset = dataset
        self.min_success_rate = min_success_rate
        self.results: list[Any] = []

    async def run_evaluation(self) -> bool:
        """Run the evaluation and return success status."""
        logger.info("Starting pre-release evaluation on %s", self.dataset)

        # Check required environment variables
        if not settings.api_key:
            logger.error("HUD_API_KEY environment variable not set")
            return False

        if not settings.anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY environment variable not set")
            return False

        try:
            # Import required modules
            from hud.datasets import load_tasks, run_dataset

            # Load tasks and apply tool filter to each task's environment
            task_list = load_tasks(self.dataset)
            for task in task_list:
                if task.env is not None:
                    # Only allow anthropic_computer tool for Claude
                    task.env._agent_include = ["anthropic_computer"]

            # Run the evaluation
            agent_params = {
                "model": "claude-sonnet-4-5",
                "verbose": False,
            }

            logger.info("Running evaluation...")
            self.results = await run_dataset(
                tasks=task_list,
                agent_type="claude",
                agent_params=agent_params,
                max_concurrent=25,
                max_steps=25,
            )

            return self._validate_results()

        except Exception as e:
            logger.exception("Evaluation failed with error: %s", e)
            return False

    def _validate_results(self) -> bool:
        """Validate the evaluation results."""
        if not self.results:
            logger.error("No results returned from evaluation")
            return False

        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if getattr(r, "reward", 0) > 0.7)
        error_tasks = sum(1 for r in self.results if getattr(r, "isError", False))

        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0

        # Log summary statistics
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info("Total tasks: %s", total_tasks)
        logger.info("Successful tasks: %s/%s (%.1f%%)", successful_tasks, total_tasks, success_rate)
        logger.info("Failed tasks: %s", total_tasks - successful_tasks)
        logger.info("Error tasks: %s", error_tasks)
        logger.info("=" * 60)

        # Check validation criteria
        validation_passed = True

        # Criterion 1: No tasks should have errors
        if error_tasks > 0:
            logger.error("❌ %s tasks encountered errors", error_tasks)
            validation_passed = False
        else:
            logger.info("✅ No tasks encountered errors")

        # Criterion 2: At least one task must succeed (non-zero score)
        if successful_tasks == 0:
            logger.error("❌ No tasks achieved a non-zero score")
            validation_passed = False
        else:
            logger.info("✅ %s tasks achieved non-zero scores", successful_tasks)

        # Criterion 3: Success rate must meet minimum threshold
        if success_rate < self.min_success_rate:
            logger.error(
                "❌ Success rate %.1f%% is below minimum threshold of %.1f%%",
                success_rate,
                self.min_success_rate,
            )
            validation_passed = False
        else:
            logger.info("✅ Success rate %.1f%% meets minimum threshold", success_rate)

        # Log individual task results for debugging
        if not validation_passed:
            logger.info("\nIndividual task results:")
            for i, result in enumerate(self.results):
                reward = getattr(result, "reward", 0)
                is_error = getattr(result, "isError", False)
                status = "ERROR" if is_error else f"Reward: {reward}"
                logger.info("  Task %s: %s", i + 1, status)

        return validation_passed


async def main() -> int:
    """Main entry point."""
    # Parse command line arguments if needed
    import argparse

    parser = argparse.ArgumentParser(description="Run pre-release evaluation checks")
    parser.add_argument(
        "--dataset",
        default="hud-evals/test-diverse",
        help="Dataset to evaluate (default: hud-evals/test-diverse)",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=50.0,
        help="Minimum success rate percentage required (default: 50.0)",
    )
    args = parser.parse_args()

    # Run the checker
    checker = PreReleaseChecker(dataset=args.dataset, min_success_rate=args.min_success_rate)
    success = await checker.run_evaluation()

    if success:
        logger.info("\n🎉 All pre-release checks passed!")
        return 0
    else:
        logger.error("\n❌ Pre-release checks failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
