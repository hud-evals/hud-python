# Reviewing hud-python

Review adversarially: assume the change breaks something and find where. The
tests cover contracts at public boundaries, so the review is where a broken
seam, a missing scenario, or an unintended behavior change gets caught.

## Behavior

- Trace each changed contract from a real producer to a real consumer. Ask who
  calls the code, with what inputs, and what they do with its output.
- Look for inputs a real producer emits that the change mishandles, and for
  outputs the change emits that a real consumer cannot read.
- When you find a break, give the scenario row that fails because of it.

## Tests

The rules are under "Testing" in `AGENTS.md`. Treat each of these as a finding:

- A test that patches a `hud` name, reads a private attribute, hand-builds what
  a producer emits, or asserts that a mock was called the way it was set up.
- A bug fix without a scenario row that fails before the fix.
- A changed inline snapshot that the change does not explain. Every snapshot
  change is a behavior change; one updated just to make a test pass hides one.
- A new test that only re-checks the author's change without adding a contract.
  That belongs in a temporary script.
- A scenario that could pass with the code under test deleted.

## Isolation and credentials

For changes to `hud/environment`, the runtimes, egress, or Harbor, ask what an
agent inside the sandbox could now do: read the host's credentials, reach the
control channel, run code outside the sandbox, or keep a process alive past
teardown. Anything it newly could is a finding.
