# File Analysis

A document-analysis environment for tasks built from prompts, uploaded files,
and weighted grading criteria. Files are staged in an isolated workspace where
the agent can inspect them with shell tools and write a final report.

The featured task is a supplier-payment reconciliation. It combines invoices,
bank payments, receiving records, and vendor terms, then grades the resulting
report for correct calculations, evidence, and recommended follow-up.

The environment provides:

- an isolated shell workspace with network access disabled
- file tracking for the agent's report
- criteria-based grading with an LLM judge

Deploy from this directory with:

```bash
hud deploy
```
