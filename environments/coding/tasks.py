"""Sample tasks: four bugs in the sample repo, plus one SDLC variant.

Each row parameterizes a template from ``env.py`` with the 3-branch
convention — ``{task}_baseline`` (starting state), ``{task}_test`` (hidden
tests), ``{task}_golden`` (reference fix) — as refs in the repo the env
serves (``REPO_URL``, default
https://github.com/hud-evals/coding-template-sample)::

    hud eval tasks.py claude --task-ids sentry-fix -y --runtime local
    hud eval tasks.py claude --full

SWE-bench Pro rows live in ``swe_tasks.py``.
"""

from env import coding_task, env, sdlc_task  # noqa: F401  (env re-exported for `hud eval tasks.py`)

TEST_COMMAND = "python3 -m pytest -q {test_files}"


def _bug(slug: str, task_id: str, description: str, test_files: list[str]):
    task = coding_task(
        description=description,
        test_command=TEST_COMMAND,
        base_ref=f"origin/{task_id}_baseline",
        test_ref=f"origin/{task_id}_test",
        golden_ref=f"origin/{task_id}_golden",
        test_files=test_files,
    )
    task.slug = slug
    return task


tasks = [
    _bug(
        "sentry-fix",
        "sentry_fix",
        "Fix a crash in the user profile endpoint.\n\n"
        "The user profile service crashes with a KeyError for certain users. Some users\n"
        "have incomplete profile data — their `profile` field may be None or missing\n"
        "entirely. The service works fine for users with complete profiles but fails for\n"
        "others. Investigate and fix the error handling in the user service.\n\n"
        "Expected behavior when a user has no profile (the `profile` field is None or\n"
        "absent): fall back to the user's top-level `name` for the display name and use an\n"
        "empty string for the bio. Users with a complete profile keep their existing\n"
        "`display_name` and `bio`.",
        ["test_user_service.py"],
    ),
    _bug(
        "notif-bug",
        "notif_bug",
        "Fix the broken notification system.\n\n"
        "The notification system is completely silent — no notifications are generated when\n"
        "tasks are created, assigned, or completed. The event handlers are registered and\n"
        "the notification service is initialized, but events never reach their handlers.\n"
        "Investigate the event routing pipeline and fix the issue.",
        ["test_notifications.py"],
    ),
    _bug(
        "settings-v2",
        "settings_v2",
        "Fix disappearing fields in API responses.\n\n"
        "API responses for the settings and user endpoints are randomly dropping fields\n"
        "that have values like 0, false, or empty string. Direct key lookups work fine,\n"
        "but when responses are serialized to JSON, certain valid fields disappear. The\n"
        "issue affects multiple endpoints and seems related to how data is iterated over\n"
        "during serialization.",
        ["test_settings.py"],
    ),
    _bug(
        "webhook-bug",
        "webhook_bug",
        "Fix inconsistent webhook notification channels.\n\n"
        "Webhook notifications work correctly on the first request for a given event type,\n"
        "but subsequent requests for the same event type produce incorrect or duplicated\n"
        "notification channels. The issue gets worse with repeated requests — channels\n"
        "accumulate and sort order changes unexpectedly.",
        ["test_notifications.py"],
    ),
]


# The SDLC variant of sentry-fix: same bug, but the task arrives as a GitHub
# issue and the deliverable is a pushed branch with a pull request.
_sentry_fix_pr = sdlc_task(
    description=(
        "Issue #42 in the tracker reports a crash in the user profile endpoint. "
        "Read the issue with the github tools, fix the bug, and ship the fix "
        "through the normal review workflow."
    ),
    test_command=TEST_COMMAND,
    base_ref="origin/sentry_fix_baseline",
    test_ref="origin/sentry_fix_test",
    golden_ref="origin/sentry_fix_golden",
    test_files=["test_user_service.py"],
    issues=[
        {
            "number": 42,
            "title": "KeyError crash on user profile endpoint",
            "body": (
                "Some users crash the profile endpoint with a KeyError — their `profile` "
                "field can be None or missing. Expected: fall back to the top-level `name` "
                "for the display name and an empty string for the bio; users with a "
                "complete profile keep their existing `display_name` and `bio`."
            ),
            "labels": ["bug"],
        }
    ],
)
_sentry_fix_pr.slug = "sentry-fix-pr"
tasks.append(_sentry_fix_pr)
