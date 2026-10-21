# Dispatch report format

Save both files in `outputs/`.

## dispatch-plan.xlsx

Use these four worksheets and headers:

| Worksheet | Headers |
| --- | --- |
| Dispatch | `order_id`, `decision`, `kit`, `quantity`, `warehouse`, `dispatch_utc`, `priority_points`, `contribution_usd`, `reason`, `evidence` |
| Inventory | `warehouse`, `sku`, `snapshot_id`, `gross`, `reserved`, `available`, `consumed`, `remaining` |
| Capacity | `warehouse`, `early_minutes`, `early_minutes_limit`, `daily_minutes`, `daily_minutes_limit`, `early_cartons`, `early_cartons_limit`, `late_cartons`, `late_cartons_limit` |
| Evidence | A readable audit trail identifying the records and policy supporting material decisions. |

`Dispatch` must contain one row for every candidate order. Use SHIP or DEFER. Show actual planned quantities, points and contribution; deferred rows have zero for those values and blank warehouse/time. Use full UTC timestamps.

`Inventory` must contain one row per warehouse and component SKU, showing the reconciled stock, reservations, consumption and balance. `Capacity` must contain one row per warehouse, showing actual usage and applicable limits. Both worksheets must reconcile to the dispatch rows.

Use numeric cells for quantities, minutes, points and USD. If you use formulas, include correct visible values or explicit numeric subtotals. Evidence should cite source record IDs and explain the basis for admitting or excluding material records.

## release-note.md

State the knowledge cutoff, chosen orders and schedule, total kits/points/contribution, and remaining constrained stock and capacity. Explain each deferral and any approval relied upon. Justify optimality with a reproducible enumeration, bound or equivalent checkable argument. Identify operational follow-ups.

Any feasible plan attaining the policy's optimum is acceptable. No particular presentation style is required.
