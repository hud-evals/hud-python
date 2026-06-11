"""Contract tooling: match a model contract against an env contract.

A *contract* is the JSON schema a robot env advertises with its ``robot``
capability — robot type, control rate, and every observation/action feature
(dtype/shape/names/stats plus semantic fields like ``state_type``, ``frame``,
``units``). Model contracts describe the same things from the policy's side.
The contract format is defined in ``spec_v0.md`` co-located in this package.

This package is the **advisory** wiring check used at preflight time:

- :func:`~hud.environment.robots.contracts.matching.match` — robot_type gate
  (v0: support is the top-level ``robot_type``; returns a plain bool).
- :func:`~hud.environment.robots.contracts.matching.pair_observations` /
  :func:`~hud.environment.robots.contracts.matching.match_actions` — feature pairing.
- :func:`~hud.environment.robots.contracts.adaptation.integration_review` — gap
  analysis (dtype/shape/frame/units/control_rate mismatches). Reports problems;
  does not generate adapters.
- :func:`~hud.environment.robots.contracts.visualization.render_match` — terminal
  wiring diagram.

The v0 contract schema is the single-space form: one embodiment (``robot_type``),
one ``role == "action"`` feature set plus observations per contract. A model or
env with several action/observation forms ships one contract per form. Every
feature is rank ≥ 1 (scalars use ``[1]``). The retired multi-mode schema
(``action_modes`` / ``observation_modes`` / ``robot_type_variables``) lives only
as archived documentation in the demos ``contracts/experiments/`` corpus; this
package does not load it.
"""

from __future__ import annotations

from .adaptation import Gap, IntegrationReview, integration_review
from .matching import (
    ActionMatch,
    Feature,
    action_signature,
    list_actions,
    match,
    match_actions,
    pair_observations,
    split_observations,
)
from .visualization import format_integration_review, render_match

__all__ = [
    "ActionMatch",
    "Feature",
    "Gap",
    "IntegrationReview",
    "action_signature",
    "format_integration_review",
    "integration_review",
    "list_actions",
    "match",
    "match_actions",
    "pair_observations",
    "render_match",
    "split_observations",
]
