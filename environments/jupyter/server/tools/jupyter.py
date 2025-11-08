import os
from hud.tools.jupyter import JupyterTool
from ..config import SOLUTIONS_PATH


class JupyterToolWithRecord(JupyterTool):
    """Jupyter Tool with code recording"""

    async def _execute(self, code: str, execution_timeout: int = 15) -> str:
        result = await super()._execute(code, execution_timeout)

        # Record code if no error
        is_error = (
            "-----" in result
            or "Error" in result
            or "Traceback" in result
            or "Execution timed out" in result
        )
        if not is_error:
            with open(os.path.join(SOLUTIONS_PATH, "1_solution.py"), "a") as f:
                f.write(code)
                f.write("\n\n")

        return result
