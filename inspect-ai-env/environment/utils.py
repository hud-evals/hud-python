import json
import asyncio
from typing import List

import hud


async def run_eval_and_log(trace_id: str, command: List[str]):
    """
    This is the background task. It creates its own trace, runs the
    subprocess, and pipes the output to the trace's log method.
    """
    with hud.trace(trace_id) as trace:
        try:
            await trace.log({"status": "starting", "command": command})

            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            async def log_stream(stream, stream_name):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    try:
                        # Best case: the process outputs structured JSON
                        log_data = json.loads(line)
                        await trace.log(log_data)
                    except json.JSONDecodeError:
                        # Fallback for plain text lines
                        await trace.log(
                            {"stream": stream_name, "message": line.decode().strip()}
                        )

            await asyncio.gather(
                log_stream(process.stdout, "STDOUT"),
                log_stream(process.stderr, "STDERR"),
            )

            await process.wait()
            await trace.log({"status": "finished", "return_code": process.returncode})

        except Exception as e:
            await trace.log({"status": "failed", "error": str(e)})
