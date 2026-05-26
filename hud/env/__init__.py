"""HUD environment runtime.

::

    from hud.env import Capability, Env, Workspace

    async def amain():
        workspace = Workspace(root="/tmp/hud-coding")
        await workspace.start()                               # binds the SSH server

        env = Env(
            name="coding",
            capabilities=[
                # Workspace runs the daemon; env-author wires the URL + keys.
                Capability.ssh(
                    url=workspace.ssh_url,
                    host_pubkey=workspace.ssh_host_pubkey,
                    client_key_path=workspace.ssh_client_key_path,
                ),
            ],
        )

        @env.scenario(description="write fizzbuzz")
        async def fizzbuzz(*, n: int = 100):
            (workspace.root / "README.md").write_text(f"write fizzbuzz for n=1..{n}")
            _ = yield {"prompt": f"write fizzbuzz for n=1..{n}"}
            # plain Python — the agent's work landed under workspace.root via SFTP
            ok = (workspace.root / "fizzbuzz.py").exists()
            yield {"score": 1.0 if ok else 0.0}

        await env.serve(port=7000)

Other capabilities follow the same pattern — env-author runs the daemon
(Chromium, Xvnc, FastMCP, rosbridge_server) and constructs the capability
from its URL::

    Capability.cdp(url="ws://127.0.0.1:9222")
    Capability.rfb(url="rfb://127.0.0.1:5900")
    Capability.mcp(url="ws://127.0.0.1:9990/mcp")
    Capability.ros2(url="ws://127.0.0.1:9090")
"""

from .capability import Capability, Endpoint
from .env import Env
from .scenario import Scenario, ScenarioFn, ScenarioRunner
from .workspace import DEFAULT_SYSTEM_MOUNTS, Mount, MountKind, Workspace

__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Capability",
    "Endpoint",
    "Env",
    "Mount",
    "MountKind",
    "Scenario",
    "ScenarioFn",
    "ScenarioRunner",
    "Workspace",
]
