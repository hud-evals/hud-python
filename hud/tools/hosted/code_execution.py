"""Provider-executed code execution tool."""

from __future__ import annotations

from typing import Any, ClassVar

from hud.tools.hosted.base import HostedTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType


class CodeExecutionTool(HostedTool):
    """Provider-executed code execution tool.

    When enabled, the model can generate and execute code in a sandboxed environment.

    Gemini: Works out of the box.
        env.add_tool(CodeExecutionTool())

    OpenAI: Requires container configuration.
        env.add_tool(CodeExecutionTool(container={"image": "python:3.12"}))
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(api_type="code_execution", hosted=True),
        AgentType.GEMINI_CUA: NativeToolSpec(api_type="code_execution", hosted=True),
        AgentType.OPENAI: NativeToolSpec(api_type="code_interpreter", hosted=True),
    }

    def __init__(self, container: dict[str, Any] | None = None) -> None:
        """Initialize CodeExecutionTool.

        Args:
            container: OpenAI container config for code_interpreter.
                       When provided, enables the tool for OpenAI agents.
        """
        instance_specs: NativeToolSpecs | None = None
        if container is not None:
            instance_specs = {
                AgentType.OPENAI: NativeToolSpec(
                    api_type="code_interpreter", hosted=True, extra={"container": container}
                ),
            }
        super().__init__(
            name="code_execution",
            title="Code Execution",
            description="Execute code in a sandboxed environment",
            native_specs=instance_specs,
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
