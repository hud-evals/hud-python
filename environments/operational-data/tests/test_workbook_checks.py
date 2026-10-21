"""Saved numeric evidence retains partial credit alongside narrative criteria."""

from pathlib import Path

import pytest
from hud.graders import SubScore
from openpyxl import Workbook

import agent_workspace
import workflow
from data_sources.base import ActionLog
from workbook_checks import WorkbookAlternative, WorkbookCheck, WorkbookTable, check_workbook


@pytest.fixture
def check() -> WorkbookCheck:
    return WorkbookCheck(
        file="reconciliation.xlsx",
        alternatives=[
            WorkbookAlternative(
                tables=[
                    WorkbookTable(
                        sheet="Invoices",
                        keys=["Invoice"],
                        rows=[{"Invoice": "INV-1", "Pay": 125}, {"Invoice": "INV-2", "Pay": None}],
                        totals={"Pay": 125},
                    )
                ]
            )
        ],
    )


def _saved(root: Path, rows: list[list[object]]) -> agent_workspace.Deliverable:
    target = root / "outputs" / "reconciliation.xlsx"
    target.parent.mkdir(parents=True)
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Invoices"
    sheet.append(["Invoice", "Pay"])
    for row in rows:
        sheet.append(row)
    workbook.save(target)
    workbook.close()
    return agent_workspace.read_deliverable(root, target)


@pytest.mark.parametrize(
    ("rows", "passes"),
    [
        ([["INV-1", 125], ["INV-2", None]], True),
        ([["INV-1", "125"], ["INV-2", None]], False),
        ([["INV-1", 125], ["INV-2", 0]], False),
        ([["INV-1", 125]], False),
        ([["INV-1", 125], ["INV-2", None], ["INV-1", 125]], False),
    ],
)
def test_checks_use_saved_typed_values_and_complete_row_keys(
    check: WorkbookCheck, tmp_path: Path, rows: list[list[object]], passes: bool
):
    saved = _saved(tmp_path, rows)
    assert saved.status == "saved"
    assert (not check_workbook(check, [saved])) is passes


@pytest.mark.parametrize("saved", [False, True])
async def test_missing_workbooks_keep_only_supported_narrative_credit(
    check: WorkbookCheck, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: bool
):
    async def judge(**kwargs: object) -> SubScore:
        assert kwargs["max_tokens"] == 16384
        return SubScore(
            name="judge",
            value=1.0,
            children=[SubScore(name="Explains the decision.", value=1.0)],
        )

    monkeypatch.setattr(workflow.LLMJudgeGrader, "compute_score", judge)
    deliverables = [_saved(tmp_path, [["INV-1", 125], ["INV-2", None]])] if saved else []
    result = await workflow.grade(
        "The payment decision is explained.",
        "Reconcile the invoices and save the workbook.",
        [
            workflow.Criterion(requirement="Explains the decision.", weight=1),
            workflow.Criterion(
                requirement="Saves the required numeric rows.", weight=3, workbook=check
            ),
        ],
        ActionLog(),
        deliverables,
    )
    assert result.reward == pytest.approx(1.0 if saved else 0.25)
