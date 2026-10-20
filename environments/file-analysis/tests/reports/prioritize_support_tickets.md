# Support handoff, 2026-10-20 15:00 UTC

| Rank | Ticket | Priority | Verified impact | Deadline UTC | First-response status | Owner |
|---|---|---|---|---|---|---|
|1|S-406 Vale Health enterprise EUR96000|P0|private-report exposure|2026-10-20 14:55|5min overdue|security on-call|
|2|S-401 Cedar Logistics enterprise EUR180000|P1|export blocked without workaround|2026-10-20 13:10|1h50 overdue|incident commander + engineering on-call|
|3|S-405 Juniper Co growth EUR36000|P2|import degraded; retry works|2026-10-20 14:30|30min overdue|support engineer|
|4|S-403 Bellwether Learning growth EUR42000|P2|intermittent login; private-window workaround|2026-10-20 17:30|14:00 response met SLA; unresolved|support engineer|
|5|S-404 Marlin Labs enterprise EUR240000|P3|feature request; dashboards work|2026-10-21 09:00|not overdue|customer support|
|6|S-402 Pebble Studio free EUR0|P4|cosmetic typo; no impairment|2026-10-23 08:00|not overdue|product backlog|

response-policy.md defines elapsed 24/7 first-response targets, not resolution.
tickets.csv supplies open/response timestamps; accounts.csv supplies context.
Unanswered S-405 comes before already-answered S-403 within P2.

investigation-notes.md E-406 confirms signed-out access to a private report,
so despite the normal customer label request immediate security containment
and an acknowledgment, preserving private reproduction evidence without copying
the report's data. Ask security for scope/mitigation status; none has occurred.
E-401 confirms blocked exports and failed retry, so escalate outage and send a
safe status update with next update time once agreed; ask engineering for scope.
E-405 confirms idempotent retries without duplicate rows: communicate this
workaround and investigate failures, without claiming resolution.
E-403 rejects the customer's breach claim: expired session, no exposure reproduced,
fresh private-window login works. Response already sent14:00. Track the open
intermittent defect and request new reproduction details if it persists.
E-404 says existing dashboards work; tomorrow's 2026-10-21 renewal is important
account context but cannot override impact policy. Acknowledge feature request,
collect requirements, and seek roadmap review without promising delivery.
E-402 is a spelling defect, not the urgent functional issue claimed. Log the
cosmetic correction and normal acknowledgment path.
These are recommendations only: no messages sent, access changed or fixes claimed.
