"""V5 public API import surface tests.

These tests are intentionally removal-focused: required public symbols must stay
available, but adding exports in these modules should not fail the suite.

Every symbol in the contract tables below should have concrete consumer
evidence from docs, examples, or reference environments. Do not add inferred
re-exports here just because they exist in the current package.
"""

from __future__ import annotations

from importlib import import_module

import pytest

TOP_LEVEL_DOCS_EXAMPLES_SURFACE = (
    "Chat",
    "Environment",
    "EvalContext",
    "eval",
)

TOP_LEVEL_ENVIRONMENT_SURFACE = (
    "Environment",
    "eval",
    "instrument",
    "trace",
)

TOP_LEVEL_EXPORTS = (
    "Chat",
    "Environment",
    "EvalContext",
    "eval",
    "instrument",
    "trace",
)


DOCS_EXAMPLES_PUBLIC_SURFACE: dict[str, tuple[str, ...]] = {
    "hud.agents": (
        "MCPAgent",
        "OpenAIAgent",
        "OpenAIChatAgent",
        "create_agent",
    ),
    "hud.agents.claude": ("ClaudeAgent",),
    "hud.native": (
        "BashGrader",
        "Grade",
        "Grader",
        "LLMJudgeGrader",
        "contains",
        "contains_all",
        "contains_any",
        "exact_match",
        "f1_score",
        "normalize",
        "numeric_match",
    ),
    "hud.server": (
        "MCPRouter",
        "MCPServer",
    ),
    "hud.services": (
        "Chat",
        "ChatService",
    ),
    "hud.tools": (
        "AgentTool",
        "AnthropicComputerTool",
        "BaseHub",
        "BaseTool",
        "BashTool",
        "EditTool",
        "GLMComputerTool",
        "GeminiComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
        "PlaywrightTool",
    ),
    "hud.types": (
        "AgentResponse",
        "AgentType",
        "MCPToolCall",
        "MCPToolResult",
        "Trace",
    ),
}


ENVIRONMENT_PUBLIC_SURFACE: dict[str, tuple[str, ...]] = {
    "hud.agents": (
        "MCPAgent",
        "OpenAIAgent",
        "OpenAIChatAgent",
        "create_agent",
    ),
    "hud.agents.claude": ("ClaudeAgent",),
    "hud.datasets": (
        "display_results",
        "load_tasks",
        "run_dataset",
        "run_single_task",
        "save_tasks",
        "submit_rollouts",
    ),
    "hud.environment": ("Environment",),
    "hud.server": (
        "MCPRouter",
        "MCPServer",
    ),
    "hud.services": ("ChatService",),
    "hud.tools": (
        "AgentTool",
        "AnthropicComputerTool",
        "BaseHub",
        "BaseTool",
        "BashTool",
        "EditTool",
        "HudComputerTool",
        "OpenAIComputerTool",
        "PlaywrightTool",
        "SubmitTool",
    ),
    "hud.tools.filesystem": (
        "GeminiGlobTool",
        "GeminiListTool",
        "GeminiReadManyTool",
        "GeminiReadTool",
        "GeminiSearchTool",
        "GlobTool",
        "GrepTool",
        "ListTool",
        "ReadTool",
    ),
    "hud.types": (
        "AgentType",
        "MCPToolCall",
        "MCPToolResult",
        "Trace",
        "TraceStep",
    ),
}


DOCS_EXAMPLES_DEEP_SURFACE: dict[str, tuple[str, ...]] = {
    "hud.eval.task": ("Task",),
    "hud.agents.gemini": ("GeminiAgent",),
    "hud.agents.openai": ("OpenAIAgent",),
    "hud.tools.coding": (
        "ApplyPatchTool",
        "EditTool",
        "ShellTool",
    ),
    "hud.tools.computer": (
        "AnthropicComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
    ),
    "hud.tools.executors": (
        "BaseExecutor",
        "PyAutoGUIExecutor",
        "XDOExecutor",
    ),
    "hud.tools.types": (
        "ContentResult",
        "EvaluationResult",
        "SubScore",
        "ToolError",
    ),
}


ENVIRONMENT_DEEP_SURFACE: dict[str, tuple[str, ...]] = {
    "hud.datasets.loader": ("resolve_taskset_id",),
    "hud.environment.connection": (
        "ConnectionConfig",
        "ConnectionType",
        "Connector",
    ),
    "hud.eval.manager": ("_send_job_enter",),
    "hud.eval.context": (
        "EvalContext",
        "get_current_trace_id",
        "set_trace_context",
    ),
    "hud.eval.task": ("Task",),
    "hud.datasets.utils": (
        "BatchRequest",
        "SingleTaskRequest",
        "submit_rollouts",
    ),
    "hud.native.graders": (
        "BashGrader",
        "Grade",
        "Grader",
    ),
    "hud.server.context": (
        "attach_context",
        "run_context_server",
    ),
    "hud.server.server": ("MCPServer",),
    "hud.settings": ("settings",),
    "hud.tools.base": (
        "BaseTool",
        "BaseHub",
    ),
    "hud.tools.agent": ("AgentTool",),
    "hud.agents.gemini": ("GeminiAgent",),
    "hud.agents.openai": ("OpenAIAgent",),
    "hud.tools.coding": (
        "ApplyPatchTool",
        "BashTool",
        "ClaudeBashSession",
        "EditTool",
        "GeminiEditTool",
        "GeminiShellTool",
        "GeminiWriteTool",
        "ShellTool",
    ),
    "hud.tools.coding.bash": (
        "BashTool",
        "ClaudeBashSession",
        "ContentResult",
        "ToolError",
    ),
    "hud.tools.coding.edit": (
        "Command",
        "EditTool",
    ),
    "hud.tools.coding.gemini_edit": ("GeminiEditTool",),
    "hud.tools.coding.gemini_shell": ("GeminiShellTool",),
    "hud.tools.coding.session": ("BashSession",),
    "hud.tools.coding.shell": (
        "BashSession",
        "ShellTool",
    ),
    "hud.tools.coding.utils": ("get_demote_preexec_fn",),
    "hud.tools.computer": (
        "AnthropicComputerTool",
        "GeminiComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
        "QwenComputerTool",
        "computer_settings",
    ),
    "hud.tools.computer.settings": ("computer_settings",),
    "hud.tools.computer.anthropic": ("AnthropicComputerTool",),
    "hud.tools.computer.hud": ("HudComputerTool",),
    "hud.tools.computer.openai": ("OpenAIComputerTool",),
    "hud.tools.executors": ("BaseExecutor",),
    "hud.tools.executors.base": ("BaseExecutor",),
    "hud.tools.jupyter": ("JupyterTool",),
    "hud.tools.playwright": ("PlaywrightTool",),
    "hud.tools.types": (
        "AgentAnswer",
        "ContentResult",
        "EvaluationResult",
        "SubScore",
        "ToolError",
    ),
    "hud.telemetry.exporter": ("queue_span",),
    "hud.telemetry.instrument": ("instrument",),
    "hud.tools.executors.pyautogui": ("PyAutoGUIExecutor",),
    "hud.tools.executors.xdo": ("XDOExecutor",),
}


DOCS_EXAMPLES_DEEP_MODULES: tuple[str, ...] = ()


ENVIRONMENT_DEEP_MODULES = (
    "hud.agents.base",
    "hud.telemetry.exporter",
)


DOCS_EXAMPLES_LAZY_PUBLIC_EXPORTS: dict[str, tuple[str, ...]] = {
    "hud.tools": (
        "AnthropicComputerTool",
        "GLMComputerTool",
        "GeminiComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
    ),
}


ENVIRONMENT_LAZY_PUBLIC_EXPORTS: dict[str, tuple[str, ...]] = {
    "hud.tools": (
        "AnthropicComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
    ),
    "hud.tools.computer": (
        "AnthropicComputerTool",
        "GeminiComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
        "QwenComputerTool",
    ),
}


def _merge_symbol_tables(
    *tables: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, set[str]] = {}
    for table in tables:
        for module_name, symbols in table.items():
            merged.setdefault(module_name, set()).update(symbols)
    return {module_name: tuple(sorted(symbols)) for module_name, symbols in sorted(merged.items())}


PUBLIC_SURFACE = _merge_symbol_tables(
    DOCS_EXAMPLES_PUBLIC_SURFACE,
    ENVIRONMENT_PUBLIC_SURFACE,
)
DEEP_SURFACE = _merge_symbol_tables(
    DOCS_EXAMPLES_DEEP_SURFACE,
    ENVIRONMENT_DEEP_SURFACE,
)
LAZY_PUBLIC_EXPORTS = _merge_symbol_tables(
    DOCS_EXAMPLES_LAZY_PUBLIC_EXPORTS,
    ENVIRONMENT_LAZY_PUBLIC_EXPORTS,
)
DEEP_MODULES = tuple(sorted(set(DOCS_EXAMPLES_DEEP_MODULES) | set(ENVIRONMENT_DEEP_MODULES)))


def assert_module_has_symbols(module_name: str, symbols: tuple[str, ...]) -> None:
    module = import_module(module_name)
    missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
    assert not missing, f"{module_name} missing public symbols: {missing}"


def test_hud_top_level_all_is_exact_v5_surface() -> None:
    import hud

    assert tuple(hud.__all__) == TOP_LEVEL_EXPORTS


def test_hud_top_level_exports_are_available() -> None:
    assert_module_has_symbols("hud", TOP_LEVEL_EXPORTS)


@pytest.mark.parametrize(("module_name", "symbols"), sorted(PUBLIC_SURFACE.items()))
def test_public_module_symbols_are_available(module_name: str, symbols: tuple[str, ...]) -> None:
    assert_module_has_symbols(module_name, symbols)


@pytest.mark.parametrize(("module_name", "symbols"), sorted(DEEP_SURFACE.items()))
def test_de_facto_public_deep_path_symbols_are_available(
    module_name: str,
    symbols: tuple[str, ...],
) -> None:
    assert_module_has_symbols(module_name, symbols)


@pytest.mark.parametrize("module_name", DEEP_MODULES)
def test_de_facto_public_deep_modules_are_importable(module_name: str) -> None:
    import_module(module_name)
