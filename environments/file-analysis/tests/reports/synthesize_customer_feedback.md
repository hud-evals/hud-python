# Onboarding decision brief

1. Fix workspace naming first. F-01 and F-05 entered names with spaces and failed
at the very first step. events.csv EV-102/EV-502 record invalid_slug; session-notes.md
F-01 says "I don't know which bit it wants me to fix." F-05: "It says name, not web address."
Use a friendly display-name field, derive a valid slug automatically with preview,
and show inline actionable validation adjacent to the field before submission.
2. Make invitation prerequisites actionable. F-02/F-08 reached disabled Invite
with tooltip-only email verification gating (events.csv EV-202/EV-802).
session-notes.md: F-02 "Did that click do anything?"; F-08 "I thought I was already set up."
Show a visible verification explanation and verify/resend action, preserving the
intended invite draft so the user can resume. No invitation was sent in either case.

participants.csv and events.csv: workspace6/8 succeeded and2/8 failed overall;
first-time4/6 succeeded and2/6 failed; returning2/2 succeeded. Invitation3/5
succeeded,2/5 failed among observable attempts; first-time2/4 succeeded,2/4
failed. F-01/F-05 never attempted invitations. F-07 disconnected (EV-702), so
invitation outcome is unknown, excluded from that denominator and not a UI failure.

Lesser issues: F-04 found no saved view through global search, overlooking the
Collections label (EV-403/EV-404); investigate the label in follow-up, but this
was one observed later-task failure. F-06 completed every task and asked for
dark mode, a preference without an observed color/contrast failure. Neither
outweighs the two early blockers.

Validate revised naming and invitation flow with eight new first-time and four
returning users, with verified/unverified email coverage. Repeat workspace,
invitation and saved-view tasks; record completion without moderator help,
validation errors, successful invite actions, task time, and help requests.
Predefine targets: all12 create a workspace without slug errors and all eligible
invitation attempts find verification/resend and complete the invite action;
record failures and time distribution rather than claiming proof from a threshold.
The study-brief.md convenience sample of8 moderated sessions is exploratory:
it cannot establish population conversion lift, causality or revenue impact.
F-07's missing outcome and recruitment bias remain uncertain. No change has
shipped and no performance improvement has been measured.
