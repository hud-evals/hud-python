"""Example tasks for the argument-hints environment."""

from env import env, review_files

tasks = [
    review_files(
        prompt="Name three prime numbers under 20 and briefly say why each is prime.",
        attachments=[],
        criteria=[
            {"requirement": "Names exactly three numbers, all prime and under 20.", "weight": 2.0},
            {"requirement": "Explains primality for each number.", "weight": 1.0},
            {"requirement": "Includes a composite number presented as prime.", "weight": -2.0},
        ],
    ),
]

__all__ = ["env", "tasks"]
