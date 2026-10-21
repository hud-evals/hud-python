# Settlement report format

Save both files in `outputs/`.

## reconciliation.xlsx

Use a worksheet named `Reconciliation`, with one row per unique invoice ID, including invoices outside the proposed batch. Use these headers; additional useful columns are allowed:

`invoice_id`, `schedule`, `corrected_gross`, `net_settled`, `open_before_discount`, `discount`, `recommended_cash`, `decision`, `evidence`.

- `schedule`: the applicable commercial schedule.
- `corrected_gross`: corrected goods less accepted credits, plus allowed freight and tax, before a new early-payment discount.
- `net_settled`: net bank cash settled as of the review cutoff.
- `open_before_discount`: corrected gross less net settled cash.
- `discount`: the net-goods discount proposed for this batch.
- `recommended_cash`: proposed cash after the discount and its tax effect.
- `decision`: PAY, FUTURE DUE, SETTLED, OVERPAID, HOLD PENDING, HOLD RECEIPT or VOID.
- `evidence`: supporting agreement/amendment clause, ERP row, credit and bank record identifiers.

Use numeric currency cells rounded to cents. For VOID invoices, show zero liability, open balance, discount and proposed cash. Where acceptance is missing, leave corrected_gross, open_before_discount and discount blank, propose zero cash and explain what is missing. Retain overpayments as negative open balances with zero proposed cash. Apply the agreements' accounting and payment rules.

Include separately labeled totals for recommended cash, known signed open balances before new discounts, and unresolved ERP exposure. The known-open total includes future-due and held-pending balances and signed overpayments; do not combine unknown exposure with it.

## decision.md

State the batch recommendation and totals, reconciliation method, material exceptions and evidence supporting the decisions. Explain which records you used or excluded and why. Identify matters procurement or treasury must resolve. The memo and workbook must agree.
