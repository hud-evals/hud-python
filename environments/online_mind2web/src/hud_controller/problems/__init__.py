"""Problem definitions for remote browser environment."""

from .registry import ProblemRegistry, problem

# Import problem definitions to trigger registration
from .navigate_and_verify import NavigateAndVerifyProblem
from .form_interaction import FormFillAndSubmitProblem
from .search_interaction import GoogleSearchProblem
from .element_interaction import ButtonClickTestProblem

__all__ = [
    "ProblemRegistry",
    "problem",
]
