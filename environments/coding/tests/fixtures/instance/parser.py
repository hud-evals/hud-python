"""Fixture stand-in for an official per-instance parser: reports no tests."""

import json
import sys

if __name__ == "__main__":
    with open(sys.argv[3], "w") as f:
        json.dump({"tests": []}, f)
