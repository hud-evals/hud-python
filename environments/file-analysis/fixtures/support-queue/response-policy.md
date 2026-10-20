# Folio Dock support response policy

All companies, incidents, people and amounts in this exercise are fictional.
Snapshot: 2026-10-20 15:00 UTC. All SLAs use elapsed clock time, 24/7, starting
at opened_at. They measure FIRST human response, not resolution. Use the
verified investigation notes when customer labels conflict with observed impact.

| priority | observed impact | first response target | primary owner |
|---|---|---|---|
| P0 | confirmed unintended access to customer data | 15 minutes | security on-call |
| P1 | core workflow blocked without a usable workaround | 1 hour | incident commander + engineering on-call |
| P2 | core workflow degraded or blocked with a usable workaround | 4 hours | support engineer |
| P3 | how-to or feature request without a current service impairment | 24 hours | customer support |
| P4 | cosmetic defect or preference without functional impact | 72 hours | product backlog |

Order P0 before P1 before P2 before P3 before P4. Within the same priority,
items with no first response come before items already answered; then use the
earlier first-response deadline. Account size, renewal dates, and customer
urgency labels provide context but NEVER override the impact rules.
A first response already sent meets its SLA only if sent by its deadline.
Do not imply that a response resolves the incident. The queue is a snapshot;
no incident is resolved unless the notes explicitly say so.

Prepare recommendations only. Do not send replies, expose private records,
change account access, or claim to have fixed anything.
