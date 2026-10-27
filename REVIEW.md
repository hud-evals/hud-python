# Reviewing hud-python

Review adversarially: assume the change breaks something and find where. `AGENTS.md`
holds the rules a change must meet (Code Quality Bar, Architecture And Simplification,
Testing); a change that breaks one is a finding. This guide is how to look.

## Behavior

- Trace each changed contract from a real producer to a real consumer. Ask who
  calls the code, with what inputs, and what they do with its output.
- Look for inputs a real producer emits that the change mishandles, and for
  outputs the change emits that a real consumer cannot read.
- Find the cause. A fix that retries, widens a timeout, suppresses a check, or
  special-cases the reported input treats a symptom; the finding is where the
  fault is.
- Check every place the same rule applies. A fix for one instance that leaves its
  siblings broken is a finding.
- Name the consumers of anything public. Environments and services pin `hud` to a
  release range, so a changed signature, return type, or default inside that range
  breaks them. Compatibility kept for a consumer nobody can name is also a finding.
- When you find a break, give the input or sequence that triggers it.

## Size and completeness

- Ask whether a much smaller change would do the same. A component, flag, or layer
  that nothing needs today is a finding, and so is a diff that grew in every
  review round without its design being revisited.
- A change finishes what it starts. Code it made dead, and a caveat or follow-up
  left where the fix should be, are findings.

## Proof

- "Fixed" rests on the real path having run: a test through the public boundary
  that fails with the fix reverted, or the lane that exercises the change. A
  passing build or type check is not proof.
- A declared dependency range is one CI exercises. A floor below the version the
  code needs is a finding.

## Tests

The rules are under "Testing" in `AGENTS.md`. Treat each of these as a finding:

- A test that patches a `hud` name, reads a private attribute, or asserts that a
  fake was called the way it was set up.
- A test that asserts prompt text, rendered output or argv.
- A test that only re-checks the author's change, or that would pass with the
  code under test deleted.
- A regression test that would still pass with its fix reverted.

## Prose

Comments, docstrings, docs, commit messages and PR descriptions say what is and
what the reader needs. A comment that restates the code or the file's purpose, or
that carries history, measured counts, or the reasoning behind a version choice,
is a finding: that belongs in the commit message.

## Isolation and credentials

For changes to `hud/environment`, the runtimes, egress, or Harbor, ask what an
agent inside the sandbox could now do: read the host's credentials, reach the
control channel, run code outside the sandbox, or keep a process alive past
teardown. Anything it newly could is a finding.

## Responding to review

Review comments, a bot's included, are evidence. Verify each against the code,
fix what the change introduced at its root, and answer the rest with the reason.
When a second finding lands on the same invariant, list every path through it and
fix them in one pass.
