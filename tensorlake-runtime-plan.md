# Tensorlake runtime provider plan

Add one module, `hud/eval/runtime/tensorlake.py`, modeled on the Daytona
provider, then register it in five places. The Tensorlake SDK already has every
primitive the provider needs, including a local TCP tunnel, so no new transport
work is required.

## The contract to satisfy

A runtime provider is any callable that takes a task and returns an async
context manager that yields a `Runtime`. The protocol is in
`hud/eval/runtime/core.py` (`Provider`). The yielded runtime carries a
`tcp://host:port` URL, a `params` dict with `provider` and `instance_id`, and
the effective `RuntimeConfig`. The client only dials `tcp://`
(`hud/clients/client.py`, `connect`), so a cloud provider must give the agent
process a local or public TCP address.

The Daytona provider in `hud/eval/runtime/daytona.py` does this in five steps.
Tensorlake maps onto each one:

| Step | Daytona call | Tensorlake call (`tensorlake.sandbox`) |
|---|---|---|
| Open client | `AsyncDaytona()` | `AsyncSandboxClient.for_cloud()`, reads `TENSORLAKE_API_KEY` |
| Create sandbox | `daytona.create(params, timeout=)` | `client.create_and_connect(image=, cpus=, memory_mb=, disk_mb=, gpu=GpuRequest, timeout_secs=, startup_timeout=)` |
| Start env server | `process.execute_session_command(run_async=True)` | `sandbox.start_process("sh", ["-c", cmd], working_dir=)` |
| Reach the port | asyncssh local forward | `await sandbox.create_tunnel(8765, local_port=0)` returns `AsyncTcpTunnel` with `local_port` |
| Teardown | `daytona.delete(sandbox)` | `await tunnel.close()` then `await sandbox.terminate()` |

`RuntimeResources` maps cleanly: `cpu` to `cpus` (floats allowed), `memory_mb`
and `storage_mb` to `memory_mb` and `disk_mb` with no GiB rounding, `gpu.count`
and one `gpu.type` to `GpuRequest(count, model=GpuModel(...))`. Both
`RuntimeLimits` fields are supported, unlike Daytona which rejects
`run_timeout_s`. Reject `compose` the same way Daytona does.

## Files to touch

- `hud/eval/runtime/tensorlake.py`: new `TensorlakeRuntime` class, SDK imported
  lazily inside `__call__` like the Daytona provider.
- `hud/eval/runtime/__init__.py` and `hud/eval/__init__.py`: export
  `TensorlakeRuntime`.
- `hud/cli/eval.py`: add `TENSORLAKE` to the `Placement` enum and the match;
  update the `--runtime` help text.
- `pyproject.toml`: add a `tensorlake = ["tensorlake>=0.5"]` extra beside
  `daytona`, then run `uv lock`.
- Tests: fake the `tensorlake.sandbox` module via
  `monkeypatch.setitem(sys.modules, ...)` as `_install_fake_daytona` does in
  `hud/eval/tests/test_docker_provider.py`. Add the enum to
  `hud/cli/tests/test_eval.py` and the error string that lists the choices.
- Docs: add a row and a section in `docs/v6/reference/runtime.mdx`, and the
  provider list in `docs/v6/internals/placement.mdx`.

## Things to decide or flag

- **Image names differ.** Tensorlake only accepts a registered Sandbox Image
  name, not a registry reference. The constructor takes `image_name`, and
  `runtime_config.image` means that name too. A "build if missing" path exists
  through `tensorlake.image.sandbox_builder.build_sandbox_image` and
  `import_sandbox_image`, but both are synchronous and heavy. Ship without it
  first.
- **No async log streaming.** `follow_output` returns a full list only after
  the process exits. Match Daytona's failure path instead: on a connection
  error, fetch `get_output(pid)` and attach it as an exception note.
- **SDK state.** `AsyncSandboxClient` emits a `FutureWarning` that it is in
  preview. The wheel is a native `abi3` build for macOS arm64, Linux x86_64 and
  aarch64, and Windows, and it pulls in `grpcio` and `protobuf`. Keeping it
  behind an extra matches the repo rule on optional dependencies.
- **Shared and warm pools.** Tensorlake has sandbox pools with warm containers.
  `create_and_connect(pool_id=)` could later back a fast path, but the basic
  provider does not need it.

## Status (2026-10-02)

Implemented on branch `feat/tensorlake_sandbox`, not yet committed:

- `hud/eval/runtime/tensorlake.py` with `TensorlakeRuntime`.
- Exports in `hud/eval/runtime/__init__.py` and `hud/eval/__init__.py`.
- CLI `--runtime tensorlake` in `hud/cli/eval.py`.
- `tensorlake` extra in `pyproject.toml`; CI type-check step installs it.
- Tests in `hud/eval/tests/test_docker_provider.py` and `hud/cli/tests/test_eval.py`.
- Docs: runtime reference, CLI reference, running-an-eval guide, placement
  and walkthrough internals.

Not done: a live run against a real Tensorlake sandbox, and the "build the
image if missing" path.
