"""Generate graded variants of the calc debugging task.

Each function is self-contained — no function calls another — so an injected bug
fails only its own tests. That keeps difficulty a function of *how many* bugs are
present rather than of which dependency chain happened to break.

A variant is a deterministic subset of BUGS of size k, k cycling 2/3/4.
"""

from __future__ import annotations

HEADER = '"""A tiny stats helper."""\n\nfrom collections import Counter\n'

FUNCS: dict[str, dict[str, str]] = {
    "mean": {
        "correct": """
def mean(xs):
    return sum(xs) / len(xs)
""",
        "buggy": """
def mean(xs):
    return sum(xs) // len(xs)
""",
        "tests": """
def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5
""",
    },
    "median": {
        "correct": """
def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
""",
        "buggy": """
def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]
""",
        "tests": """
def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5
""",
    },
    "variance": {
        "correct": """
def variance(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
""",
        "buggy": """
def variance(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)
""",
        "tests": """
def test_variance_is_sample_variance():
    assert variance([1, 2, 3, 4]) == pytest.approx(1.6666666666666667)
""",
    },
    "stdev": {
        "correct": """
def stdev(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
""",
        "buggy": """
def stdev(xs):
    m = sum(xs) / len(xs)
    return sum(abs(x - m) for x in xs) / len(xs)
""",
        "tests": """
def test_stdev_is_sample_stdev():
    assert stdev([1, 2, 3, 4]) == pytest.approx(1.2909944487358056)
""",
    },
    "percentile": {
        "correct": """
def percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return float(s[lo])
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
""",
        "buggy": """
def percentile(xs, p):
    s = sorted(xs)
    return s[int(round((len(s) - 1) * p))]
""",
        "tests": """
def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.25) == pytest.approx(1.75)


def test_percentile_endpoints():
    assert percentile([1, 2, 3, 4], 0.0) == pytest.approx(1.0)
    assert percentile([1, 2, 3, 4], 1.0) == pytest.approx(4.0)
""",
    },
    "mode": {
        "correct": """
def mode(xs):
    counts = Counter(xs)
    top = max(counts.values())
    return min(x for x, c in counts.items() if c == top)
""",
        "buggy": """
def mode(xs):
    return Counter(xs).most_common(1)[0][0]
""",
        "tests": """
def test_mode_breaks_ties_by_smallest():
    assert mode([3, 1, 1, 3]) == 1
""",
    },
    "spread": {
        "correct": """
def spread(xs):
    return max(xs) - min(xs)
""",
        "buggy": """
def spread(xs):
    return max(xs) - min(xs) + 1
""",
        "tests": """
def test_spread():
    assert spread([1, 2, 3, 4]) == 3
""",
    },
    "cv": {
        "correct": """
def cv(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return sd / m
""",
        "buggy": """
def cv(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return m / sd
""",
        "tests": """
def test_cv():
    assert cv([1, 2, 3, 4]) == pytest.approx(0.5163977794943222)
""",
    },
}

NAMES = list(FUNCS)
K_CYCLE = (4, 5, 6)


def broken_for(variant: int) -> list[str]:
    """Deterministic subset of function names to break for *variant*."""
    k = K_CYCLE[variant % len(K_CYCLE)]
    start = (variant * 3) % len(NAMES)
    return [NAMES[(start + i) % len(NAMES)] for i in range(k)]


def build(variant: int) -> tuple[str, str, list[str]]:
    """Return (calc.py source, test_calc.py source, names broken)."""
    broken = broken_for(variant)
    calc = HEADER + "".join(
        "\n" + FUNCS[name]["buggy" if name in broken else "correct"].strip("\n") + "\n"
        for name in NAMES
    )
    tests = (
        "import pytest\n\nfrom calc import "
        + ", ".join(sorted(NAMES))
        + "\n"
        + "".join("\n" + FUNCS[name]["tests"].strip("\n") + "\n" for name in NAMES)
    )
    return calc, tests, broken


def test_count() -> int:
    return sum(f["tests"].count("def test_") for f in FUNCS.values())
