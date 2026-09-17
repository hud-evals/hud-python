"""``ASTCodeGrader`` — evaluate Python code AST and enforce anti-cheat policies."""

from __future__ import annotations

import ast
import logging
from typing import Any

from .base import Grader
from .results import SubScore

logger = logging.getLogger(__name__)


class ASTCodeGrader(Grader):
    """Static Python AST evaluator for code generation rollouts and anti-cheat policies.

    Validates Python syntax without executing untrusted code, and optionally verifies
    required function/class definitions while enforcing anti-cheat rules (e.g. disallowed
    imports such as 'subprocess', 'os', 'sys' or forbidden calls such as 'eval', 'exec').
    """

    name = "ASTCodeGrader"

    @classmethod
    async def compute_score(
        cls,
        code: str | None = None,
        required_functions: list[str] | None = None,
        required_classes: list[str] | None = None,
        disallowed_imports: list[str] | None = None,
        disallowed_calls: list[str] | None = None,
        **kwargs: Any,
    ) -> SubScore:
        """Parse ``code`` into an AST and enforce syntactic/anti-cheat constraints."""
        if code is None:
            raise ValueError("ASTCodeGrader requires code")
        del kwargs

        # 1. Syntax parse
        try:
            tree = ast.parse(code)
        except SyntaxError as err:
            logger.debug("ASTCodeGrader syntax error on line %s: %s", err.lineno, err.msg)
            return SubScore(
                name=cls.name,
                value=0.0,
                info={
                    "valid_syntax": False,
                    "error": str(err.msg),
                    "lineno": err.lineno,
                    "offset": err.offset,
                },
            )

        # 2. Extract defined functions, classes, imports, and calls
        defined_funcs = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        defined_classes = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }

        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])

        calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)

        violations: list[str] = []
        missing_definitions: list[str] = []

        if required_functions:
            missing_definitions.extend(
                f"Missing required function: {fn}"
                for fn in required_functions
                if fn not in defined_funcs
            )

        if required_classes:
            missing_definitions.extend(
                f"Missing required class: {cls_name}"
                for cls_name in required_classes
                if cls_name not in defined_classes
            )

        if disallowed_imports:
            violations.extend(
                f"Disallowed import detected: {imp}"
                for imp in disallowed_imports
                if imp in imports
            )

        if disallowed_calls:
            violations.extend(
                f"Disallowed call detected: {call_name}"
                for call_name in disallowed_calls
                if call_name in calls
            )

        passed = len(violations) == 0 and len(missing_definitions) == 0

        return SubScore(
            name=cls.name,
            value=1.0 if passed else 0.0,
            info={
                "valid_syntax": True,
                "passed": passed,
                "defined_functions": sorted(defined_funcs),
                "defined_classes": sorted(defined_classes),
                "imports": sorted(imports),
                "violations": violations,
                "missing_definitions": missing_definitions,
            },
        )


def is_valid_python(code: str) -> float:
    """Return 1.0 if ``code`` parses as valid Python syntax without execution, else 0.0."""
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0


__all__ = ["ASTCodeGrader", "is_valid_python"]
