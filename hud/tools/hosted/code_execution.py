"""Provider-executed code execution tool."""

from __future__ import annotations

from typing import Any, ClassVar

from hud.tools.hosted.base import HostedTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType


class CodeExecutionTool(HostedTool):
    """Provider-executed code execution tool.

    When enabled, the model can generate and execute code in a sandboxed environment.
    Supported by Gemini (code_execution) and OpenAI (code_interpreter).

    Note: OpenAI's code_interpreter requires additional configuration and may have
    usage costs. Gemini's code_execution is included in standard API access.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(api_type="code_execution", hosted=True),
        AgentType.GEMINI_CUA: NativeToolSpec(api_type="code_execution", hosted=True),
        AgentType.OPENAI: NativeToolSpec(api_type="code_interpreter", hosted=True),
    }

    def __init__(self) -> None:
        """Initialize CodeExecutionTool."""
        super().__init__(
            name="code_execution",
            title="Code Execution",
            description="Execute code in a sandboxed environment",
        )

    @staticmethod
    def process_response(response: Any) -> dict[str, Any]:
        """Extract code execution results from the response.

        Args:
            response: Provider response containing code execution results

        Returns:
            Dictionary with code and output fields
        """
        # Gemini includes executable_code and code_execution_result in parts
        try:
            results: list[dict[str, Any]] = []

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts or []:
                        if hasattr(part, "executable_code") and part.executable_code:
                            results.append(
                                {
                                    "type": "code",
                                    "language": getattr(part.executable_code, "language", "python"),
                                    "code": part.executable_code.code,
                                }
                            )
                        if hasattr(part, "code_execution_result") and part.code_execution_result:
                            results.append(
                                {
                                    "type": "result",
                                    "outcome": getattr(
                                        part.code_execution_result, "outcome", "unknown"
                                    ),
                                    "output": part.code_execution_result.output,
                                }
                            )

            return {"executions": results} if results else {}
        except Exception:
            return {}
