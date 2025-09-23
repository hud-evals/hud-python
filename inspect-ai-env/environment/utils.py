import subprocess
import sys


def run_uv_command(args):
    """
    Runs a uv command with the given arguments and returns the captured output.
    """
    command = ["uv"] + args

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,  # This will raise a CalledProcessError if the command fails
    )
    return result.stdout.strip(), result.stderr.strip()
