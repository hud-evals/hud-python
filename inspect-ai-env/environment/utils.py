import subprocess
from typing import List


def run_command(args: List[str]):
    """
    Runs a uv command with the given arguments and returns the captured output.
    """

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=True,  # This will raise a CalledProcessError if the command fails
    )
    return result.stdout.strip(), result.stderr.strip()
