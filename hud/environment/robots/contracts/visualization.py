"""Terminal visualization for contract matching results."""

from __future__ import annotations

from .adaptation import IntegrationReview, integration_review
from .matching import Feature, match, match_actions, pair_observations


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _lbl(name: str | None, feature: dict | None) -> str:
    if not feature:
        return "(none)"
    kind = feature.get("type") or feature.get("state_type", "?")
    shape = feature.get("shape", "")
    return f"{name} [{kind} {shape}]"


def _rows(
    pairs: list[tuple[Feature, Feature]],
    arrow: str,
    *,
    indent: str,
    env_code: str,
    model_code: str,
) -> list[str]:
    lefts = [_lbl(en, ef) for (en, ef), _ in pairs]
    rights = [_lbl(mn, mf) for _, (mn, mf) in pairs]
    width = max((len(label) for label in lefts), default=0)
    return [
        f"{indent}{_c(f'{left:<{width}}', env_code)} {_c(arrow, '90')} {_c(right, model_code)}"
        for left, right in zip(lefts, rights, strict=True)
    ]


def format_integration_review(review: IntegrationReview) -> list[str]:
    """Render an integration review block for terminal output."""
    lines = [_c("  integration review:", "1;90")]
    lines.append(_c("    matched:", "90"))
    lines.extend(f"      · {item}" for item in review.scope)
    if review.problems:
        lines.append(_c("    problems:", "91"))
        for gap in review.problems:
            lines.append(f"      [{gap.category}] {gap.issue}")
            lines.append(_c(f"        spec: {gap.spec}", "90"))
    else:
        lines.append(_c("    problems: (none)", "90"))
    return lines


def render_match(
    model: dict,
    env: dict,
    *,
    model_name: str = "model",
    env_name: str = "env",
    integration: bool = False,
) -> str:
    robot_type = env.get("robot_type", "?")
    decision_variables = match(model, robot_type)
    head = _c(
        f"robot: env {env_name!r} ({robot_type}) <-> model {model_name!r}",
        "1;36",
    )
    if decision_variables is None:
        robots = list(model.get("robot_type_variables", {}))
        return f"{head}\n  {_c('NO MATCH', '1;31')} {_c(f'(model robots: {robots})', '90')}"

    lines = [
        head,
        f"  {_c('MATCH', '1;32')} | decision_variables={decision_variables or '{}'}",
        _c("  observations (env -> model):", "1;34"),
        *_rows(
            pair_observations(env, model, robot_type),
            "->",
            indent="    ",
            env_code="34",
            model_code="36",
        ),
    ]

    action = match_actions(env, model, robot_type)
    lines.append(_c("  action (env <- model):", "1;33"))
    if action.matched:
        lines.append(_c(f"    mode={action.mode!r} [{action.signature}]", "33"))
        lines.extend(
            _rows(list(action.pairs), "<-", indent="      ", env_code="33", model_code="35")
        )
    else:
        lines.append(
            _c(
                f"    model modes {list(action.available_signatures)} "
                f"-> env wants [{action.signature}]  MISSING",
                "1;31",
            )
        )

    if integration:
        review = integration_review(env, model, decision_variables=decision_variables)
        if review is not None:
            lines.extend(format_integration_review(review))
    return "\n".join(lines)
