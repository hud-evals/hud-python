import json
import asyncio
from typing import List

import json
import os
from unittest.mock import patch


class MockTrace:
    """
    A mock trace object that now correctly implements the async context manager protocol.
    """

    def __init__(self, trace_id):
        self.trace_id = trace_id
        self.filename = f"{self.trace_id}.log"

        # Clean up the log file from previous runs when a new trace starts
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def __enter__(self):
        print("Entering the 'with' block.")
        return self  # This value is assigned to 'cm' in the with statement

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the 'with' block.")
        if exc_type:
            print(f"An exception of type {exc_type} occurred.")
        # Perform cleanup actions here
        return False  # Return True to suppress the exception

    async def __aenter__(self):
        """
        This method is called when entering the 'async with' block.
        It should return the object that will be used as the context variable ('trace').
        """
        print(f"Starting trace '{self.trace_id}'. Logging to '{self.filename}'")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        This method is called when exiting the 'async with' block.
        It's used for cleanup. exc_type, exc_val, and exc_tb will contain
        exception information if one occurred inside the block.
        """
        print(f"Finished trace '{self.trace_id}'.")
        # We don't need any special cleanup, so we can just pass.
        pass

    async def log(self, data):
        """
        This is our mock implementation. It saves the log data to a file.
        """
        with open(self.filename, "a+") as f:
            f.write(json.dumps(data) + "\n")


# This is a placeholder for the actual 'hud' package
class MockHud:
    def trace(self, trace_id):
        return MockTrace(trace_id)


hud = MockHud()


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
