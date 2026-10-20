# Reviewing hud-python

Review adversarially: assume the change breaks something and find where. The
tests pin what a user would notice, so the review is where a broken seam or an
unintended behavior change gets caught.

## Behavior

- Trace each changed contract from a real producer to a real consumer. Ask who
  calls the code, with what inputs, and what they do with its output.
- Look for inputs a real producer emits that the change mishandles, and for
  outputs the change emits that a real consumer cannot read.
- When you find a break, give the input or sequence that triggers it.

## Tests

The rules are under "Testing" in `AGENTS.md`. Treat each of these as a finding:

- A test that patches a `hud` name, reads a private attribute, or asserts that a
  fake was called the way it was set up.
- A test that asserts prompt text, rendered output or argv.
- A test that only re-checks the author's change, or that would pass with the
  code under test deleted.
- A regression test that would still pass with its fix reverted.

## Isolation and credentials

For changes to `hud/environment`, the runtimes, egress, or Harbor, ask what an
agent inside the sandbox could now do: read the host's credentials, reach the
control channel, run code outside the sandbox, or keep a process alive past
teardown. Anything it newly could is a finding.
