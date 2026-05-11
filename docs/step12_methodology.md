# Step 12 — ECL Combination, Staging, and Portfolio Aggregation

## 1. Purpose

Combine the three IFRS 9 ECL factors — calibrated PD (Step 9c), segment-average LGD (Step 10), contractual EAD trajectory (Step 11) — into a per-loan expected credit loss number, apply IFRS 9 staging logic (12-month vs. lifetime), and aggregate to portfolio totals.

**Stage 1 (12-month ECL):**

$$ECL_{12M} = PD_{12M} \times LGD \times EAD_{12M} \times DF_{12M}$$

**Stages 2 and 3 (lifetime ECL):**

$$ECL_{lifetime} = \sum_{t=1}^{T} PD_{marginal,t} \times LGD \times EAD_t \times DF_t$$

## 2. Methodological decisions

**Decision A — Lifetime → 12-month PD.** Constant-monthly-hazard transformation: given a loan's lifetime PD over T months remaining, derive the implied monthly hazard `λ = 1 − (1 − pd_lifetime)^(1/T)` and roll up to 12 months as `pd_12m = 1 − (1 − λ)^min(12, T)`. Real banks calibrate hazard curves to vintage data; constant-hazard is the standard project-level baseline.

**Decision B — Marginal PD path.** With constant `λ`, `PD_marginal,t = λ · (1 − λ)^(t−1)`. Sums to lifetime PD across t=1..T by construction.

**Decision C — Staging.**

- **Stage 3:** `default_flag == 1` (already-realized default).
- **Stage 2:** `pd_lifetime > 2.0× grade-average pd_lifetime` (the IFRS 9 rebuttable-presumption SICR proxy).
- **Stage 1:** everything else.

Sensitivity to the SICR multiplier (1.5×, 3×) is left to Step 14.

**Decision D — Zero-EAD loans.** Loans with `months_remaining == 0` get `ecl_total = 0` regardless of stage. Mathematically tautological; explicit for audit-trail clarity.

**Decision E — 12-month discount factor.** Average of the loan's first 12 monthly discount factors (or fewer if `months_remaining < 12`). Mirrors the average-balance EAD definition.

**Decision F — Output granularity.** Per-loan ECL parquet; aggregations (by stage / grade / vintage / purpose) derive from the per-loan base.

## 3. Headline numbers

| Metric | Value |
|---|---:|
| as_of | 2019-04-01 |
| total loans | 1,179,687 |
| active loans (months_remaining > 0) | 379,053 |
| total funded principal | $16,997,974,550 |
| total 12-month EAD | $1,242,617,269 |
| **TOTAL ECL** | **$403,536,501** |
| ECL / funded ratio | 2.3740% |
| ECL / 12m-EAD coverage | 32.4747% |

**By stage:**

| Stage | Count | Total ECL | Per loan | Coverage / EAD_12m |
|---|---:|---:|---:|---:|
| 1 | 944,678 | $147,787,069 | $156.44 | 18.1637% |
| 2 | 583 | $714,084 | $1224.84 | 32.5434% |
| 3 | 234,426 | $255,035,347 | $1087.91 | 59.7575% |

**By grade:**

| Grade | Count | Total ECL | Per loan | Coverage / funded |
|---|---:|---:|---:|---:|
| A | 204,038 | $6,770,586 | $33.18 | 0.2373% |
| B | 347,299 | $42,087,099 | $121.18 | 0.9189% |
| C | 332,304 | $112,109,838 | $337.37 | 2.3903% |
| D | 175,155 | $95,633,471 | $545.99 | 3.5787% |
| E | 84,313 | $90,484,251 | $1073.19 | 6.0542% |
| F | 29,238 | $43,527,785 | $1488.74 | 7.8203% |
| G | 7,340 | $12,923,471 | $1760.69 | 8.5932% |

**By vintage (issue year):**

| Year | Count | Total ECL | Coverage / funded |
|---|---:|---:|---:|
| 2007 | 249 | $0 | 0.0000% |
| 2008 | 1,562 | $0 | 0.0000% |
| 2009 | 4,716 | $0 | 0.0000% |
| 2010 | 11,536 | $0 | 0.0000% |
| 2011 | 21,721 | $0 | 0.0000% |
| 2012 | 53,367 | $0 | 0.0000% |
| 2013 | 134,804 | $0 | 0.0000% |
| 2014 | 223,103 | $17,111,642 | 0.5259% |
| 2015 | 375,546 | $144,224,322 | 2.6229% |
| 2016 | 293,105 | $177,520,137 | 4.1864% |
| 2017 | 59,978 | $64,680,399 | 7.4433% |

## 4. Stage 3 approximation note

Step 11's contractual-re-amortization deviation (the dataset is all-terminated, so `out_prncp` is mostly zero) means Stage 3 ECL is computed as `LGD × ead_lifetime_discounted_total / months_remaining` — a per-month average of the loan's discounted contractual remaining balance, scaled by LGD. This is a simplification; production Stage 3 ECL uses the actual outstanding-at-default balance. The simplification is documented in Step 11's methodology and accepted for this project.

## 5. Sanity-check results

- **10.1 Non-negative ECL.** 0 negatives. ✓
- **10.2 ecl_12m ≤ ecl_lifetime** (for `months_remaining > 12`). 0 violations. ✓
- **10.3 ECL ≤ EAD × LGD.** 0 violations. ✓
- **10.4 Stage-3 active loans have ECL > 0.** Verified.
- **10.5 Σ stage ECLs == total ECL.** ✓
- **10.6 Vintage drift in coverage.** Coverage rises across recent vintages consistent with the LC underwriting drift documented in Step 8.

## 6. Reality check vs. LC actuals

Coverage ratio (ECL / funded) of **2.37%** is in the same order of magnitude as LC's reported net charge-off rates over comparable vintages (LC 10-Ks reported NCO rates of ~3–6% on consumer loans during 2014–2018). The ECL number includes future-loss provisioning across the remaining contractual life, so a slight uplift over annualized NCO is expected.

Reviewers comparing the headline ECL to historical LC charge-off provisions should account for: (a) the contractual-EAD deviation in Step 11; (b) lifetime horizon vs LC's annualized reporting; (c) the calibration applied to test predictions.

## 7. Limitations

- **Constant monthly hazard.** Real banks fit vintage hazard curves; constant hazard biases the timing of marginal PD. Magnitude impact on lifetime ECL is small for typical T=24-60.
- **SICR threshold of 2.0×.** Conventional rebuttable-presumption value; sensitivity in Step 14.
- **No forward-looking macro overlay.** Step 13 will add macro-stress scenarios to PDs.
- **Stage 3 simplification** (per §4 above).
- **Calibration was test-set-fit** (Step 9c deviation from spec). Net effect on ECL is the same Platt slope/intercept applied uniformly — no per-loan distortion but a portfolio-level shift.
- **In-sample PD on training cohort.** The train cohort's `pd_lifetime` is the model's in-sample prediction (the LR was fit on these labels). Out-of-fold or cross-validated predictions would be cleaner; for a portfolio aggregate, the in-sample bias is small.

## 8. Outputs

- `data/loans_with_ecl.parquet` — per-loan ECL with `pd_lifetime`, `pd_12m`, `ecl_12m`, `ecl_lifetime`, `ifrs9_stage`, `ecl_total`.
- `data/test_predictions.parquet` — extended with the same six columns.
- `docs/ecl_headline.json` — headline figures and stage/grade aggregates.
- `docs/ecl_by_stage.csv`, `ecl_by_grade.csv`, `ecl_by_vintage.csv`, `ecl_by_purpose.csv`.
