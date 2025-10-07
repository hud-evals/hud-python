import socket
import asyncio
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvloop
import vllm.envs as envs
from fastapi import Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
    load_log_config,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.utils import FlexibleArgumentParser

from openai import AsyncOpenAI, NotFoundError
from httpx import Response

from hud.rl.logger import console
from hud.rl.utils import get_weights_path

__all__ = [
    "CheckpointWorker",
    "custom_build_async_engine_client",
    "custom_run_server_worker",
    "custom_run_server",
    "server",
]

class CheckpointWorker:
    """Extension of a vLLM worker that can hot-load checkpoints over RPC."""

    def update_weights(self, model_path: Path) -> None:
        """Update weights from a specified path pointing to a ``.pt`` file."""
        state_dict = torch.load(model_path, map_location="cpu", mmap=True)

        def weights_iterator():
            for key, value in state_dict.items():
                if not key:
                    continue
                yield key, value

        self.model_runner.model.load_weights(weights_iterator())

        # Post-processing is required for some architectures.
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(
            self.model_runner.model,
            self.model_runner.model_config,
            device,
        )


@asynccontextmanager
async def custom_build_async_engine_client(
    args: Namespace,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """Create an engine client that registers the checkpoint worker extension."""

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = "hud.rl.vllm.CheckpointWorker"

    async with build_async_engine_client_from_engine_args(
        engine_args,
        disable_frontend_multiprocessing=args.disable_frontend_multiprocessing,
        client_config=client_config,
    ) as engine:
        yield engine


async def custom_run_server_worker(
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    client_config: dict[str, Any] | None = None,
    **uvicorn_kwargs,
) -> None:
    """Run a single vLLM API server worker with custom endpoints."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified.
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with custom_build_async_engine_client(args, client_config) as engine_client:
        app = build_app(args)

        @app.post("/update_weights")
        async def _update_weights(request: Request):
            data = await request.json()
            model_path = data.get("model_path")
            await engine_client.collective_rpc("update_weights", args=(model_path,))
            return {"status": "ok"}

        @app.post("/reload_weights")
        async def _reload_weights(request: Request):
            await engine_client.collective_rpc("reload_weights")
            return {"status": "ok"}

        vllm_config = await engine_client.get_vllm_config()
        await init_app_state(engine_client, vllm_config, app.state, args)

        console.info_log(
            f"Starting vLLM API server {server_index} on {listen_address}",
        )
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    try:
        await shutdown_task
    finally:
        sock.close()


async def custom_run_server(args: Namespace, **uvicorn_kwargs) -> None:
    """Launch the custom server worker after setting up the listener socket."""

    listen_address, sock = setup_server(args)
    await custom_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


def server(config_or_args: Namespace | list[str], extra_args: list[str] | None = None) -> None:
    """Entry point mirroring `vllm serve` with our custom worker extension.

    Accepts either a ready Namespace (typically from Config.inference.to_vllm())
    optionally overlaying extra CLI flags, or a raw list of CLI tokens.
    """

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)

    if isinstance(config_or_args, Namespace):
        # Fill defaults and apply any extra args onto provided namespace
        args = parser.parse_args(args=extra_args or [], namespace=config_or_args)
    else:
        console.info_log(f"vLLM server args: {config_or_args}")
        args = parser.parse_args(args=config_or_args)

    validate_parsed_serve_args(args)

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            uvloop.run(custom_run_server(args))


# Client functions

async def check_health(client: AsyncOpenAI, interval: int = 1, log_interval: int = 10, timeout: int = 1800) -> None:
    wait_time = 0
    url = str(client.base_url).strip()[:-4] + "/health"
    console.debug_log(f"Starting pinging {url} to check health")
    while wait_time < timeout:
        try:
            await client.get(url, cast_to=Response, options={"max_retries": 0})
            console.debug_log(f"Inference pool is ready after {wait_time} seconds")
            return
        except NotFoundError:
            console.warning_log(f"The route {url} does not exist. Skipping health check.")
            return
        except Exception as e:
            if wait_time % log_interval == 0 and wait_time > 0:
                console.warning_log(f"Inference server was not reached after {wait_time} seconds (Error: {e})")
            await asyncio.sleep(interval)
            wait_time += interval
    msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
    console.error_log(msg)
    raise TimeoutError(msg)


async def check_has_model(client: AsyncOpenAI, model_name: str) -> None:
    console.debug_log(f"Checking if model {model_name} is in the inference pool")
    models = (await client.models.list()).data
    if not any(model.id == model_name for model in models):
        raise ValueError(f"Model {model_name} was not found in the inference pool")
    console.debug_log(f"Model {model_name} was found in the inference pool")


async def update_weights(client: AsyncOpenAI, path: Path, step: int) -> None:
    """Make a HTTP post request to the vLLM server to update the weights."""
    url = str(client.base_url).strip()[:-4] + "/update_weights"
    try:
        model_path = get_weights_path(path, step).absolute()
        console.debug_log(f"Sending request to {url} to update weights from {model_path}")
        await client.post(url, cast_to=Response, body={"model_path": model_path.as_posix()})
    except NotFoundError:
        console.warning_log(f"The route {url} does not exist. Skipping weight update.")
        return


async def reload_weights(client: AsyncOpenAI) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    url = str(client.base_url).strip()[:-4] + "/reload_weights"
    try:
        console.debug_log(f"Sending request to {url} to reload weights (reset to base model)")
        await client.post(url, cast_to=Response, body={})
    except NotFoundError:
        console.warning_log(f"The route {url} does not exist. Skipping weight reload.")
        return
    await client.post(url, cast_to=Response, body={})


def main() -> None:
    from hud.rl.config import VLLMConfig

    cfg, extra = VLLMConfig.from_argv(allow_extras=True)
    ns = cfg.to_vllm()
    server(ns, extra_args=extra)


if __name__ == "__main__":
    main()
