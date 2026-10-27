"""A source that fails as it loads: a container serving it exits before it serves."""

raise ImportError("broken fixture: no module named bugs")
