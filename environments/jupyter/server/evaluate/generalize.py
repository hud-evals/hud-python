import os, json, logging
import copy
from ..config import SOLUTIONS_PATH

logger = logging.getLogger(__name__)


def kmp(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text using KMP algorithm.
    Returns:
        List of starting positions where pattern is found in text
    """
    if not pattern or not text:
        return []

    n = len(text)
    m = len(pattern)
    pi = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j
    positions = []
    j = 0
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            positions.append(i - m + 1)
            j = pi[j - 1]

    return positions


def generalize_code(id: str):
    """Generalize solution code from instance 1 to instances 2 and 3."""
    # src_path = "/app/shared_data/1_solution.py"
    src_path = os.path.join(SOLUTIONS_PATH, "1_solution.py")

    # Read code from source file
    if not os.path.exists(src_path):
        logger.warning(f"Solution file not found: {src_path}")
        return {"error": f"Solution file not found: {src_path}"}

    with open(src_path, "r") as f:
        code = f.read()

    if not code.strip():
        return {"error": "Solution file is empty"}

    results = {}

    for i in range(2, 4):
        src_pattern = f"1_{id}_"
        tgt_pattern = f"{i}_{id}_"
        # tgt_path = f"/app/shared_data/{i}_solution.py"
        tgt_path = os.path.join(SOLUTIONS_PATH, f"{i}_solution.py")

        positions = kmp(code, src_pattern)

        # Convert to list for efficient in-place replacement (O(n) instead of O(k*n))
        code_list = list(code)
        for pos in positions:
            for j, char in enumerate(tgt_pattern):
                code_list[pos + j] = char
        new_code = "".join(code_list)

        # Write to target file
        with open(tgt_path, "w") as f:
            f.write(new_code)

        results[f"instance_{i}"] = {
            "path": tgt_path,
            "replacements": len(positions),
            "pattern": f"{src_pattern} -> {tgt_pattern}",
        }

    return results
