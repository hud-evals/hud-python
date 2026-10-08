import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock


def test_host_serves_mcp_without_a_transport_mode(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "sim" / "host.py"
    spec = importlib.util.spec_from_file_location("robot_gripper_host", path)
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)

    run_server = AsyncMock()
    simulator = ModuleType("sim.server")
    simulator.server = SimpleNamespace(run_async=run_server)
    package = ModuleType("sim")
    package.server = simulator
    monkeypatch.setitem(sys.modules, "sim", package)
    monkeypatch.setitem(sys.modules, "sim.server", simulator)
    monkeypatch.setenv("WORLDSIM_SIM_PORT", "8769")
    monkeypatch.delenv("WORLDSIM_SIM_SERVE", raising=False)
    monkeypatch.delenv("WORLDSIM_VIEWER", raising=False)

    host.main()

    run_server.assert_awaited_once_with(transport="http", host="127.0.0.1", port=8769, show_banner=False)
