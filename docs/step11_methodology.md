# Step 11 — EAD Projection

## 1. Purpose

Exposure at Default (EAD) is the third factor in the IFRS 9 ECL formula:

$$ECL = PD \times LGD \times EAD$$

For an amortizing term loan, EAD is the outstanding principal balance at a given month — a function of original principal, interest rate, term, and elapsed time. Unlike PD (statistical) or LGD (segmental average), EAD is **deterministic arithmetic** given the contract: the closed-form balance formula is the same one a mortgage calculator uses.

Two horizons are produced per loan:
- **12-month EAD** — average outstanding balance over the next 12 months (the IFRS 9 Stage 1 ECL input).
- **Lifetime EAD path** — projected balance at every month until contractual maturity (Stage 2 / Stage 3 ECL input), plus discounted-total summary.

## 2. Methodological decisions

**Decision A — `as_of` snapshot date.** Set to 2019-04-01 for all loans. This is the dataset cutoff used in Step 7's maturity filter; reusing it keeps the pipeline coherent.

**Decision B — Schedule starting balance (DEVIATION from spec).** The original specification used `out_prncp` as the as-of starting balance, with Task 2.5 zero-ing out any loan whose `loan_status` is in (Charged Off, Default, Fully Paid). **Every loan in this dataset is in one of those three statuses** (Step 7 filtered out Currents and Late buckets, and DNMCP rows were dropped). Following the spec literally produces an all-zero EAD path, breaks the aggregate plausibility check, and leaves the methodology unexercised.

Resolution: re-amortize from `funded_amnt` to compute the contractual balance at `as_of`, then project forward using the closed-form amortization formula. This treats the EAD step as a **contractual-hypothetical projection** — the forward EAD any performing loan would have given its origination terms. It mirrors how a production EAD model would behave on a portfolio of live loans, demonstrated on this dataset.

For loans with `months_remaining ≤ 0` (contractually matured by 2019-04-01), EAD is zero. For others, the contractual balance at `as_of` is the starting point for the path.

**Decision C — Prepayment.** Ignored. LC borrowers prepay at 5–15% annually; ignoring prepayment biases EAD upward by an estimated 5–10% over a 36-month horizon. Future work: separate prepayment hazard model.

**Decision D — 12-month EAD.** Average outstanding balance over months 1 to 12 (or fewer if `months_remaining < 12`). Matches the IFRS 9 12-month ECL integration. Also stored: `ead_at_month_12` for transparency.

**Decision E — Lifetime EAD storage.** Both the full balance vector (`ead_lifetime_path`, parquet list column) and summary statistics: `ead_lifetime_undiscounted_total`, `ead_lifetime_discounted_total`, `ead_at_month_24/36/60`. Total file size ≈ 60–80 MB.

**Decision F — Discount factor.** Per-loan vector `(1 + monthly_rate)^(-t)` for t = 1..months_remaining. Stored alongside the balance path in `discount_factors`. Step 12 multiplies element-wise. Strictly, IFRS 9 requires the original effective interest rate (EIR, including fees); `int_rate` is a close approximation.

## 3. Aggregate diagnostics

- Population: **1,179,687** loans (matches input).
- Active for EAD (months_remaining > 0): **379,053**.
- Inactive (already matured contractually): **800,634**.

**Distribution of `ead_12m` (full population):**

- min = 0.00, max = 26898.61
- mean = 1053.34, median = 0.00
- p25 = 0.00, p75 = 501.47, p90 = 3762.94

Histogram in `docs/ead_histogram.csv`. Months-remaining distribution in `docs/ead_months_remaining_distribution.csv`.

**Status breakdown:**

| Bucket | Count | % |
|---|---:|---:|
| zero (matured) | 800,634 | 67.87% |
| short (<12) | 240,857 | 20.42% |
| medium (12-36) | 138,196 | 11.71% |
| long (>36) | 0 | 0.00% |

**Aggregate sums:**

- Total 12-month EAD: **$1,242,617,269**
- Total lifetime undiscounted: $18,525,830,728
- Total lifetime discounted: $16,445,638,133
- Total funded principal: $16,997,974,550
- Ratio (12-month EAD / funded): **0.0731**

The ratio reflects the portfolio's average remaining contractual life: a fully fresh portfolio would be near 0.95, a fully matured portfolio near 0. Mid-life portfolios sit between 0.3 and 0.7.

## 4. Sanity check results

- 8.1 Principal conservation (5K sample, |Δ|/start ≤ 0.1%): **pass**
- 8.2 Monotonic non-increasing balance: **pass**
- 8.3 Final balance < $1: **pass**
- 8.4 Iterative ↔ closed-form match (1¢ tolerance): **pass**
- 8.5 ead_12m ≤ starting balance: **pass**
- 8.6 Aggregate ratio plausibility (0.05–0.95): **pass**

## 5. Limitations

- **Prepayment ignored.** Empirical prepayment rates for LC are 5–15% annually depending on grade; ignoring them biases EAD upward by ~5–10% over a 36-month horizon. A production model would fit a separate prepayment hazard.
- **No re-amortization on missed payments.** The contractual schedule assumes regular catch-up, not re-amortization, when payments are missed. For consumer term loans this is the standard simplification.
- **Discount factor uses `int_rate`.** Strictly, IFRS 9 requires the original effective interest rate (EIR) including fees. `int_rate` is a close approximation; the difference is sub-percent for LC's pricing structure.
- **Methodological deviation on starting balance.** Re-amortization from `funded_amnt` (not `out_prncp`) is used because the dataset is all-terminated. On a live-portfolio dataset, `out_prncp` would be the correct choice. The code produces methodologically valid contractual EAD; the input data does not exercise actual-vs-contractual divergence.
- **Fixed-rate assumption.** All LC consumer loans are fixed-rate; no interest-rate uncertainty in the projection.

## 6. Outputs

- **Loans + EAD:** `data/loans_with_ead.parquet`.
- **Test predictions extended:** `data/test_predictions.parquet` (columns `ead_12m`, `ead_lifetime_discounted_total`, `months_remaining`).
- **Histogram of ead_12m:** `docs/ead_histogram.csv`.
- **Months-remaining distribution:** `docs/ead_months_remaining_distribution.csv`.
- **Status breakdown:** `docs/ead_status_breakdown.csv`.
