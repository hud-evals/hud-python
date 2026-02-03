"""Pre-deploy validation for HUD environments.

Catches common issues before uploading to avoid wasted build time.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime


@dataclass
class ValidationIssue:
    """A validation issue found during pre-deploy checks."""

    severity: str  # "error" or "warning"
    message: str
    file: str | None = None
    hint: str | None = None


def validate_pyproject_references(directory: Path) -> list[ValidationIssue]:
    """Check that files referenced in pyproject.toml exist.

    Validates:
    - license (file = "LICENSE" or similar)
    - readme
    - include/exclude patterns that reference specific files

    Args:
        directory: Environment directory

    Returns:
        List of validation issues
    """
    issues: list[ValidationIssue] = []
    pyproject_path = directory / "pyproject.toml"

    if not pyproject_path.exists():
        return issues

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"Failed to parse pyproject.toml: {e}",
                file="pyproject.toml",
            )
        )
        return issues

    project = data.get("project", {})

    # Check license file reference
    license_info = project.get("license")
    if isinstance(license_info, dict):
        license_file = license_info.get("file")
        if license_file:
            license_path = directory / license_file
            if not license_path.exists():
                hint = (
                    f"Create a {license_file} file or remove the "
                    "license.file reference from pyproject.toml"
                )
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"License file not found: {license_file}",
                        file="pyproject.toml",
                        hint=hint,
                    )
                )

    # Check readme file reference
    readme = project.get("readme")
    if isinstance(readme, str):
        readme_path = directory / readme
        if not readme_path.exists():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Readme file not found: {readme}",
                    file="pyproject.toml",
                    hint=f"Create a {readme} file or remove the readme reference",
                )
            )
    elif isinstance(readme, dict):
        readme_file = readme.get("file")
        if readme_file:
            readme_path = directory / readme_file
            if not readme_path.exists():
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Readme file not found: {readme_file}",
                        file="pyproject.toml",
                        hint=f"Create a {readme_file} file or remove the readme.file reference",
                    )
                )

    # Check hatch/hatchling build includes
    tool = data.get("tool", {})
    hatch_build = tool.get("hatch", {}).get("build", {}).get("targets", {})

    for target_name, target_config in hatch_build.items():
        if isinstance(target_config, dict):
            includes = target_config.get("include", [])
            for pattern in includes:
                # Only check non-glob patterns
                is_literal = isinstance(pattern, str) and "*" not in pattern and "?" not in pattern
                if is_literal:
                    include_path = directory / pattern
                    if not include_path.exists():
                        hint = f"Referenced in [tool.hatch.build.targets.{target_name}].include"
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                message=f"Included file/dir not found: {pattern}",
                                file="pyproject.toml",
                                hint=hint,
                            )
                        )

    return issues


def validate_dockerfile(directory: Path) -> list[ValidationIssue]:
    """Validate Dockerfile for common issues.

    Checks:
    - COPY commands reference files that exist
    - uv sync / pip install ordering issues with pyproject.toml references

    Args:
        directory: Environment directory

    Returns:
        List of validation issues
    """
    issues: list[ValidationIssue] = []

    # Find Dockerfile
    dockerfile_path = directory / "Dockerfile.hud"
    if not dockerfile_path.exists():
        dockerfile_path = directory / "Dockerfile"
    if not dockerfile_path.exists():
        return issues

    try:
        content = dockerfile_path.read_text()
    except Exception:
        return issues

    # Track what files have been copied (for ordering validation)
    copied_files: set[str] = set()
    has_uv_sync_before_full_copy = False

    # Check for common Dockerfile issues
    lines = content.split("\n")
    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        # Track COPY commands
        if line.upper().startswith("COPY "):
            parts = line.split()
            if len(parts) >= 3:
                # Find sources (skip flags, last arg is destination)
                src_idx = 1
                while src_idx < len(parts) - 1 and parts[src_idx].startswith("--"):
                    src_idx += 1

                # All args except last are sources
                for src in parts[src_idx:-1]:
                    if src == ".":
                        copied_files.add("__ALL__")
                    else:
                        # Normalize path: remove leading ./ and trailing / or *
                        normalized = src.lstrip("./").rstrip("/").rstrip("*")
                        copied_files.add(normalized)

        # Check for uv sync or pip install before full COPY
        line_lower = line.lower()
        is_install_cmd = "uv sync" in line_lower or "pip install" in line_lower
        if is_install_cmd and "__ALL__" not in copied_files:
            has_uv_sync_before_full_copy = True

    # If uv sync runs before COPY . ., check pyproject.toml references
    if has_uv_sync_before_full_copy and (directory / "pyproject.toml").exists():
        issues.extend(_check_pyproject_copy_order(directory, copied_files, dockerfile_path.name))

    return issues


def _check_pyproject_copy_order(
    directory: Path,
    copied_files: set[str],
    dockerfile_name: str,
) -> list[ValidationIssue]:
    """Check if pyproject.toml references files that aren't copied before install."""
    issues: list[ValidationIssue] = []
    pyproject_path = directory / "pyproject.toml"

    try:
        import tomllib

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})

        # Check if LICENSE is referenced but not copied early
        license_info = project.get("license")
        if isinstance(license_info, dict):
            license_file = license_info.get("file", "")
            if license_file:
                # Normalize path for comparison
                normalized_license = license_file.lstrip("./")
                if normalized_license not in copied_files:
                    hint = (
                        f"Add 'COPY {license_file} ./' before the RUN command "
                        "that installs dependencies"
                    )
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message="LICENSE file not copied before uv sync/pip install",
                            file=dockerfile_name,
                            hint=hint,
                        )
                    )

        # Check if README is referenced but not copied early
        readme = project.get("readme")
        if isinstance(readme, str):
            # Normalize path for comparison
            normalized_readme = readme.lstrip("./")
            if normalized_readme not in copied_files:
                hint = f"Add 'COPY {readme} ./' before the RUN command, or builds may fail"
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message="README not copied before uv sync/pip install",
                        file=dockerfile_name,
                        hint=hint,
                    )
                )
    except Exception:  # noqa: S110 - best effort validation, errors are expected
        pass  # Best effort - tomllib may not parse all files

    return issues


def validate_environment(directory: Path) -> list[ValidationIssue]:
    """Run all pre-deploy validations on an environment directory.

    Args:
        directory: Environment directory

    Returns:
        List of all validation issues found
    """
    issues: list[ValidationIssue] = []

    # Run all validators
    issues.extend(validate_pyproject_references(directory))
    issues.extend(validate_dockerfile(directory))

    return issues


def format_validation_issues(issues: list[ValidationIssue]) -> str:
    """Format validation issues for display.

    Args:
        issues: List of validation issues

    Returns:
        Formatted string for display
    """
    if not issues:
        return ""

    lines: list[str] = []

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if errors:
        lines.append(f"Found {len(errors)} error(s):")
        for issue in errors:
            file_info = f" ({issue.file})" if issue.file else ""
            lines.append(f"  ✗ {issue.message}{file_info}")
            if issue.hint:
                lines.append(f"    Hint: {issue.hint}")

    if warnings:
        lines.append(f"Found {len(warnings)} warning(s):")
        for issue in warnings:
            file_info = f" ({issue.file})" if issue.file else ""
            lines.append(f"  ⚠ {issue.message}{file_info}")
            if issue.hint:
                lines.append(f"    Hint: {issue.hint}")

    return "\n".join(lines)
