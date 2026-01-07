"""HUD Playground - Interactive web UI for multi-agent testing.

Launch with: hud playground
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def create_playground(
    config_dir: str = "agents/",
    jupyter_url: str | None = None,
    browser_hub: str | None = None,
    workspace: str = "./workspace",
) -> Any:
    """Create a Gradio-based playground for interactive agent testing.

    Args:
        config_dir: Directory containing agent YAML configs
        jupyter_url: Optional Jupyter MCP server URL
        browser_hub: Optional browser hub name (e.g., "hud-online-mind2web")
        workspace: Workspace directory for agent outputs

    Returns:
        Gradio ChatInterface instance
    """
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError(
            "Gradio is required for the playground. "
            "Install with: pip install hud-python[playground]"
        ) from e

    from hud.environment import Environment
    from hud.eval.context import EvalContext
    from hud.multi_agent import MultiAgentRunner

    # State
    runner: MultiAgentRunner | None = None
    env: Environment | None = None
    eval_ctx: EvalContext | None = None

    async def initialize() -> str:
        """Initialize the environment and runner."""
        nonlocal runner, env, eval_ctx

        # Check if any environment is configured
        has_environment = bool(jupyter_url or browser_hub)
        
        if not has_environment:
            raise RuntimeError(
                "No environment connected! The playground requires at least one environment.\n\n"
                "Please specify an environment using:\n"
                "  --jupyter-url <url>  - Connect to a Jupyter MCP server\n"
                "  --browser-hub <hub>  - Connect to a browser hub (e.g., 'hud-online-mind2web')\n\n"
                "Example:\n"
                "  hud playground --jupyter-url http://localhost:8765/mcp\n"
                "  hud playground --browser-hub hud-online-mind2web"
            )

        # Create environment
        env = Environment("playground")

        # Connect to jupyter if URL provided
        if jupyter_url:
            env.connect_url(jupyter_url, alias="jupyter")

        # Connect to browser hub if provided
        if browser_hub:
            env.connect_hub(browser_hub, alias="browser")

        # Create EvalContext from Environment (for tool access without tracing)
        eval_ctx = EvalContext.from_environment(
            env,
            name="playground",
            trace=True,  # Disable telemetry for playground
            quiet=False,   # Suppress trace link printing
        )

        # Enter the eval context (connects environment)
        await eval_ctx.__aenter__()

        # Get available tools for info
        tools = await eval_ctx.list_tools()
        tool_names = [t.name for t in tools]
        
        # Verify we actually have environment tools (not just sub-agent tools)
        if len(tool_names) == 0:
            raise RuntimeError(
                "Environment connected but no tools available!\n\n"
                "This usually means the MCP server failed to start or has no tools.\n"
                "Check that your environment URL is correct and the server is running."
            )

        # Create runner with EvalContext for full tool access
        runner = MultiAgentRunner(
            config_dir=Path(config_dir),
            ctx=eval_ctx,
            workspace=Path(workspace),
        )

        await runner.initialize()

        return f"Ready! Connected to {len(tool_names)} tools: {', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}"

    async def chat(message: str, history: list[list[str]]) -> Any:
        """Handle chat messages."""
        nonlocal runner

        # Initialize on first message
        if runner is None:
            yield "🔄 Initializing environment..."
            try:
                status = await initialize()
                yield f"✅ {status}\n\n🤖 Running agent..."
            except Exception as e:
                yield f"❌ Failed to initialize: {e}"
                return

        # Run the agent
        try:
            # FIX: Set the prompt on the eval context before running
            if eval_ctx is not None:
                eval_ctx.prompt = message

            # Type guard for linter
            assert runner is not None, "Runner should be initialized"

            # Set up progress callback
            progress_messages = []
            last_progress = ""

            def on_progress(update: str) -> None:
                """Callback for progress updates."""
                nonlocal last_progress
                progress_messages.append(update)
                last_progress = update

            # Set progress callback directly on eval context (runner will pick it up)
            if eval_ctx is not None:
                setattr(eval_ctx, 'progress_callback', on_progress)

            # Initial status
            yield "🤖 Running agent..."

            # Create task for runner
            import asyncio
            run_task = asyncio.create_task(runner.run(message, max_steps=100))

            # Poll for progress while running
            while not run_task.done():
                await asyncio.sleep(0.5)  # Check every 500ms
                if last_progress:
                    # Show all progress messages
                    progress_text = "\n".join([f"  ↳ {msg}" for msg in progress_messages[-5:]])  # Last 5 updates
                    yield f"🤖 Running agent...\n\n{progress_text}"

            # Get result
            result = await run_task

            # Format response
            output = result.output or "No output"

            # Add metrics
            response = f"{output}\n\n"
            response += "---\n"
            response += f"📊 **Steps:** {result.steps} | "
            response += f"**Duration:** {result.duration_ms:.0f}ms | "
            response += f"**Files:** {len(result.files)}"

            if result.logs_dir:
                response += f" | **Logs:** `{result.logs_dir}`"
            
            # Add trace ID link if available
            if eval_ctx and hasattr(eval_ctx, 'trace_id'):
                trace_id = eval_ctx.trace_id
                response += f"\n\n🔗 **Trace:** `{trace_id}`"
                response += f"\n\n<details><summary>View Trace (embed)</summary>\n\n"
                response += f'<iframe src="https://hud.ai/trace/{trace_id}" '
                response += f'width="100%" height="800" frameborder="0"></iframe>\n\n'
                response += f"</details>"

            if not result.success:
                response += f"\n\n⚠️ **Error:** {result.error}"

            yield response

        except Exception as e:
            yield f"❌ Agent error: {e}"

    async def cleanup() -> None:
        """Cleanup resources."""
        nonlocal eval_ctx
        if eval_ctx is not None:
            await eval_ctx.__aexit__(None, None, None)

    # Build environment info for description
    env_info_parts = []
    if jupyter_url:
        env_info_parts.append(f"Jupyter: {jupyter_url}")
    if browser_hub:
        env_info_parts.append(f"Browser: {browser_hub}")
    env_info = " | ".join(env_info_parts) if env_info_parts else "⚠️ No environments configured - will fail on first message"

    # Create Gradio interface
    demo = gr.ChatInterface(
        chat,
        title="🎮 HUD Agent Playground",
        description=f"Interactive multi-agent testing\n\n**Config:** `{config_dir}` | **Environments:** {env_info}",
        examples=[
            "Navigate to https://books.toscrape.com and extract the first 3 book titles",
            "Create a Python script that calculates Fibonacci numbers",
            "What tools are available?",
        ],
    )

    return demo


def launch_playground(
    config_dir: str = "agents/",
    jupyter_url: str | None = None,
    browser_hub: str | None = None,
    workspace: str = "./workspace",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the playground server.

    Args:
        config_dir: Directory containing agent YAML configs
        jupyter_url: Optional Jupyter MCP server URL
        browser_hub: Optional browser hub name
        workspace: Workspace directory for agent outputs
        port: Port to run the server on
        share: Whether to create a public Gradio link
    """
    # Use env vars as defaults if not provided
    jupyter_url = jupyter_url or os.getenv("HUD_DEV_URL")
    browser_hub = browser_hub or os.getenv("HUD_BROWSER_HUB")

    demo = create_playground(
        config_dir=config_dir,
        jupyter_url=jupyter_url,
        browser_hub=browser_hub,
        workspace=workspace,
    )

    demo.launch(
        server_port=port,
        share=share,
        show_error=True,
    )

