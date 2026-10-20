# Supplier reconciliation as of 2026-10-06 (EUR)

| Invoice | Invoiced | Paid | Valid | Recoverable |
|---|---:|---:|---:|---:|
| NORTH-1048 | 900.00 | 1800.00 | 900.00 | 900.00 |
| ORBIT-771 | 648.00 | 648.00 | 583.20 | 64.80 |
| LUMA-309 | 864.00 | 864.00 | 691.20 | 172.80 |
| CLEAR-552 | 600.00 | 600.00 | 600.00 | 0.00 |
| Total | 3012.00 | 3912.00 | 2774.40 | 1137.60 |

NORTH: invoice-north-1048.md and accepted receipt DEMO-R101 in receiving-log.csv
support EUR900. bank-payments.csv DEMO-P001 and DEMO-P004 are two distinct
settled payments, not a reversal. Verify bank records and request EUR900 refund.
ORBIT: invoice-orbit-771.md and Orbit's signed vendor-terms.csv give unconditional
10% off 12 x45 = 540 net: 486 + 97.20 VAT = 583.20. DEMO-P002 paid648;
request discount credit/refund of64.80. No credit exists yet.
LUMA: invoice-luma-309.md, DEMO-R103 in receiving-log.csv and vendor-terms.csv
show six returned damaged stands with no replacement. 24 x24 =576 net;
VAT115.20; valid691.20. DEMO-P003 paid864; request172.80 credit/refund
(6 x24 x1.20) attaching receipt evidence.
CLEAR: invoice-clear-552.md and DEMO-R104 confirm five accepted visits. Terms
allow installments: DEMO-P005400 + DEMO-P006200 =600 in bank-payments.csv,
so clean, not duplicate. All refunds/credits are pending, none received.
No additional payments should be made. This report does not move funds or
contact suppliers.
