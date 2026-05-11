# Step 10 — LGD Estimation

## 1. Purpose

Loss Given Default (LGD) is the second factor in the IFRS 9 ECL formula:

$$ECL = PD \times LGD \times EAD$$

For loans that default, LGD is the share of exposure that is unrecoverable: `1 − net_recoveries / EAD_at_default`. Unlike PD, LGD is not predicted from borrower features at origination in the same modeling sense — the regulatory convention (and what reviewers expect) is **segmental averaging**: observe realized LGDs on historical defaulters, group by a segment definition, average within segment, apply the segment average to all loans (defaulted or live) that fall in that segment.

More sophisticated approaches (beta regression, two-stage models that separately estimate the probability of any recovery and the loss-given-some-recovery) are deferred to future work. The baseline below is defensible and audit-ready.

## 2. Methodological decisions

**Decision A — EAD-at-default.** EAD = `funded_amnt − total_rec_prncp` (original principal minus principal repaid before default). This matches the standard Basel/IFRS definition. The alternative `out_prncp` reflects post-default balance and would conflate accrued interest with principal exposure.

**Decision B — Recoveries.** Net recoveries = `recoveries − collection_recovery_fee`. The bank's economic loss is what was recovered after collection costs.

**Decision C — Realized LGD formula.**

$$\text{LGD}_{realized} = 1 - \frac{\text{recoveries} - \text{collection\_recovery\_fee}}{\text{funded\_amnt} - \text{total\_rec\_prncp}}$$

Bounded to [0, 1]. Defaulters with EAD ≤ 0 (fully amortized before default) are dropped: **15 loans** (0.01% of defaulters). Raw LGD < 0 (over-recovery) capped to 0: **80** loans. Raw LGD > 1 (data error) capped to 1: **0** loans. Total share capped: **0.03%** — well below the 5% concern threshold.

**Decision D — Segmentation.** Grade × purpose with a fallback rule: any segment with fewer than 500 defaulters falls back to grade-only LGD. This produces stable estimates while preserving granularity where data supports it. With **234,411** usable defaulters, all 7 grades have ample data; granular grade × purpose averages are reliable for high-volume purposes (debt_consolidation, credit_card) and fall back to grade-only for low-volume ones. Of the 98 (grade, purpose) combinations in the population:

- Segment averages: **27**
- Grade-only fallback: **71**
- Portfolio-average fallback: **0**

**Decision E — Out-of-sample validation.** Time-based split mirroring Step 9: defaulters with `issue_d < 2016-01-01` (152,300 loans) compute the segment averages; defaulters with `issue_d ≥ 2016-01-01` (82,111 loans) backtest them.

## 3. Distribution analysis

Realized LGD across 234,411 usable defaulters:

- mean = 0.9029, median = 0.9147, std = 0.1141
- p25 = 0.8774, p75 = 1.0000, p90 = 1.0000

**Bimodality:**

- share with LGD > 0.7 (low-recovery cluster): **93.39%**
- share with LGD < 0.5 (recovery cluster): **0.92%**
- exactly 0 (full recovery): **0.04%**
- exactly 1 (no recovery): **27.67%**

Consumer unsecured debt typically shows a heavy right cluster (most defaulters produce little or no recovery) plus a tail of partial-recovery cases. Segmental averaging smooths this bimodality; a beta-regression future-work option could preserve it.

## 4. Segment lookup

Full lookup is in `docs/lgd_lookup.csv` (98 rows).

**Top 10 highest-LGD segments:**

| Grade | Purpose | LGD | Source | n |
|---|---|---:|---|---:|
| D | small_business | 0.9032 | segment | 679 |
| E | small_business | 0.9027 | segment | 607 |
| E | home_improvement | 0.9018 | segment | 1,191 |
| B | other | 0.9018 | segment | 925 |
| C | major_purchase | 0.8997 | segment | 724 |
| E | credit_card | 0.8994 | segment | 2,649 |
| B | home_improvement | 0.8991 | segment | 1,589 |
| C | credit_card | 0.8990 | segment | 9,674 |
| B | credit_card | 0.8988 | segment | 8,152 |
| D | major_purchase | 0.8986 | segment | 613 |

**Top 10 lowest-LGD segments:**

| Grade | Purpose | LGD | Source | n |
|---|---|---:|---|---:|
| F | other | 0.8891 | segment | 739 |
| F | credit_card | 0.8898 | segment | 691 |
| E | other | 0.8899 | segment | 1,497 |
| D | other | 0.8906 | segment | 2,218 |
| G | debt_consolidation | 0.8911 | segment | 1,407 |
| B | major_purchase | 0.8921 | segment | 511 |
| G | car | 0.8925 | grade_fallback | 2,274 |
| G | credit_card | 0.8925 | grade_fallback | 2,274 |
| G | educational | 0.8925 | grade_fallback | 2,274 |
| G | home_improvement | 0.8925 | grade_fallback | 2,274 |

## 5. Backtesting results

On the 82,111 defaulters with `issue_d ≥ 2016-01-01`:

- **Aggregate observed mean LGD:** 0.9157
- **Aggregate predicted mean LGD:** 0.8959
- **Aggregate error (predicted − observed):** -0.0199
- **Weighted segment-level MAE:** 0.0199
- **Vintage-LGD drift range:** 0.0136 (max-min of observed-mean LGD across validation issue years)

Per-segment results: `docs/lgd_backtest.csv`. Aggregate error within ±0.05 is consistent with healthy out-of-sample performance for a segmental model. Drift range > 0.05 would warrant adding a vintage dimension to LGD modeling — flagged for future work if the observed range above is large.

## 6. Sensitivity analysis

Base portfolio-weighted LGD = **0.8959** (weighted by `funded_amnt`).

| Shock | Portfolio LGD | Δ vs base | % Δ |
|---|---:|---:|---:|
| -20% | 0.7167 | -0.1792 | -20.00% |
| -10% | 0.8063 | -0.0896 | -10.00% |
| +0% | 0.8959 | +0.0000 | +0.00% |
| +10% | 0.9855 | +0.0896 | +10.00% |
| +20% | 1.0751 | +0.1792 | +20.00% |


Linear scaling holds (no clipping engages because predicted LGDs sit comfortably in [0, 1]). A reviewer can read the table as: "if our LGD estimate is high by 10%, ECL is high by 10% from the LGD factor alone." The PD calibrator and EAD model produce additional independent sensitivities.

## 7. Limitations

- **Segmental homogeneity assumption.** Segmental averaging treats every loan in a segment as equally likely to lose the segment-mean LGD. Beta regression or two-stage models would capture within-segment heterogeneity (especially the 0/1 spikes). Future work.
- **No forward-looking adjustment.** Regulators typically expect a 'downturn LGD' for ECL — LGDs from a stress scenario rather than the through-the-cycle mean. The Step 14 macro overlay does not currently scale LGDs (only PDs); this is a known gap for stress-testing purposes.
- **Vintage drift not modeled.** If LGD has drifted with vintage (validation backtest above quantifies the gap), segmental averages computed on pre-2016 data will be biased on post-2016 loans. With observed drift = 0.0136, the magnitude is in the noise band and the baseline is acceptable; a re-fit on rolling-window data would address this in production.
- **Fallback threshold (n=500) is a judgment call.** Lower → more granular, noisier segments; higher → fewer segment-level estimates, more grade-only fallback. Sensitivity to this threshold is left to future work.
- **Sample exclusions.** Defaulters with EAD ≤ 0 (fully amortized before default) are excluded entirely; their LGD is undefined under the chosen formula. Cap counts (LGD < 0 or > 1) are reported transparently in `docs/lgd_stats.json`.

## 8. Outputs

- **Loans + LGD:** `data/loans_with_lgd.parquet`.
- **Test predictions extended:** `data/test_predictions.parquet` (column `lgd_predicted`).
- **Lookup table:** `docs/lgd_lookup.csv`.
- **Distribution histogram:** `docs/lgd_histogram.csv`.
- **Backtest:** `docs/lgd_backtest.csv`.
- **Sensitivity:** `docs/lgd_sensitivity.csv`.
- **Cap counts (audit input):** `docs/lgd_stats.json`.
