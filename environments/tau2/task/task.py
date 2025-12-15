"""Tau2Task class to manage task and environment state."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from tau2.data_model.tasks import Task
from tau2.data_model.message import Message
from tau2.environment.environment import Environment


@dataclass
class Tau2Task:
    """Manages the state of the current tau2-bench task and environment."""

    domain: Optional[str] = None
    task_id: Optional[str] = None
    task: Optional[Task] = None
    tasks: List[Task] = field(default_factory=list)
    environment: Optional[Environment] = None
    messages: List[Message] = field(default_factory=list)
    solo_mode: bool = False

    def is_initialized(self) -> bool:
        """Check if a task and environment are initialized."""
        return self.task is not None and self.environment is not None

    def has_tasks_loaded(self) -> bool:
        """Check if tasks are loaded."""
        return len(self.tasks) > 0

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by ID from loaded tasks."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def set_task(self, task_id: str) -> bool:
        """
        Set the current task by ID.

        Returns:
            True if task was found and set, False otherwise
        """
        task = self.get_task_by_id(task_id)
        if task is None:
            return False
        self.task = task
        self.task_id = task_id
        return True

    def add_message(self, message: Message):
        """Add a message to the conversation history."""
        self.messages.append(message)

    def clear_messages(self):
        """Clear the message history."""
        self.messages = []

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get information about the current task.

        Returns:
            Dictionary with task information
        """
        if self.task is None:
            return {"error": "No task set"}

        return {
            "id": self.task.id,
            "domain": self.domain,
            "description": self.task.description.model_dump() if self.task.description else None,
            "user_scenario": self.task.user_scenario.model_dump() if self.task.user_scenario else None,
            "has_initial_state": self.task.initial_state is not None,
            "has_evaluation_criteria": self.task.evaluation_criteria is not None,
        }

    def get_policy(self) -> Optional[str]:
        """Get the policy from the environment."""
        if self.environment is None:
            return None
        return self.environment.get_policy()

    def reset(self):
        """Reset all state."""
        self.domain = None
        self.task_id = None
        self.task = None
        self.tasks = []
        self.environment = None
        self.messages = []
        self.solo_mode = False

