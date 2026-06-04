"""``HUDConsole`` — smoke-exercise the output methods + check the pure formatters.

These mostly assert "doesn't raise" (output goes to a Rich console), which still
exercises the formatting branches; the ``format_*`` / ``prefix`` helpers return values
we can assert directly.
"""

from __future__ import annotations

from hud.utils.hud_console import HUDConsole


def test_output_methods_do_not_raise() -> None:
    c = HUDConsole()
    c.header("Title")
    c.section_title("Section")
    c.success("ok")
    c.error("bad")
    c.warning("warn")
    c.info("info")
    c.print("plain")
    c.dim_info("key", "value")
    c.link("https://example.com")
    c.json_config('{"a": 1}')
    c.progress_message("working")
    c.phase(1, "Phase one")
    c.command(["hud", "dev", "env:env"])
    c.hint("a hint")
    c.detail("detail")
    c.flow("flow")
    c.note("note")
    c.render_support_hint()
    c.symbol("*", "symbolic")


def test_verbose_toggles_debug_logging() -> None:
    c = HUDConsole()
    c.set_verbose(True)
    c.debug("debug visible")
    c.debug_log("debug log")
    c.info_log("info")
    c.progress_log("progress")
    c.success_log("done")
    c.warning_log("warn")
    c.error_log("err")
    c.set_verbose(False)
    c.debug("debug hidden")  # no-op when not verbose


def test_format_helpers_return_strings() -> None:
    c = HUDConsole()
    assert isinstance(c.format_tool_call("bash", {"command": "ls"}), str)
    assert isinstance(c.format_tool_result("output text"), str)
    assert isinstance(c.format_tool_result("error text", is_error=True), str)
    assert isinstance(c.prefix, str)


def test_render_exception_does_not_raise() -> None:
    c = HUDConsole()
    try:
        raise ValueError("boom")
    except ValueError as exc:
        c.render_exception(exc)


def test_progress_context_updates() -> None:
    c = HUDConsole()
    with c.progress("starting") as p:
        p.update("step 1")
        p.update("step 2")
