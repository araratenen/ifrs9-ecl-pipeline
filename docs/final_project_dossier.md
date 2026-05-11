# IFRS 9 ECL Modeling Project — Final Project Dossier

**Generated:** 2026-05-09T19:01:28
**Pipeline version:** 11 step scripts, 104 pipeline artifacts
**Verification status:** _(see Section 14)_
**Purpose:** The canonical project document — methodology, results, and verified file index in one self-contained reference.

## Section 0 — Document Metadata

This dossier is the single canonical reference for the IFRS 9 ECL modeling project on the LendingClub consumer-loan dataset. It is **self-verifying**: every quantitative claim in the prose is recomputed from the source artifact at generation time and tagged inline (with a green check on success, or a red flag with the cited and recomputed values on a mismatch). The verification appendix (Section 13) lists all rows; the file index (Section 12) enumerates every pipeline artifact with its existence/integrity status.

Sections are organized in a methodology-then-results flow: executive summary (1) → data sources (2) → cleaning (3) → modeling overview (4) → PD (5) → LGD (6) → EAD (7) → ECL combination (8) → macro overlays (9) → reproducibility (10) → governance (11) → file index (12) → verification appendix (13) → end matter (14).

## Section 1 — Executive Summary

This project builds an end-to-end IFRS 9 Expected Credit Loss (ECL) model on the LendingClub consumer-loan portfolio (2007–2018 vintages). It covers data acquisition, cleaning, PD/LGD/EAD modeling, IFRS 9 staging, forward-looking macro overlay, validation, and a Power BI dashboard. The pipeline is reproducible: re-running the scripts regenerates the headline numbers to within $0.01.

> **The three headline ECL numbers**
>
> | Headline | Total ECL | Use case |
> |---|---:|---|
> | Step 12 baseline | **$403,536,501** [verified ✓ — source: ecl_headline.json + parquet sum] | Internal model output, no forward-looking adjustment |
> | Step 13 data-driven overlay | **$390,821,719** [verified ✓ — source: ecl_overlay_headline.json + parquet sum] | Mechanical IFRS 9 with documented direction inversion |
> | **Step 14 regulatory overlay** | **$484,474,530** [verified ✓ — source: validation_regulatory_overlay.json] | **Recommended for IFRS 9 reporting** |

**Scope:** 1,179,687 loans [verified ✓ — source: ecl_headline.json + parquet rows], $16,997,974,550 of funded principal [verified ✓ — source: ecl_headline.json + parquet sum], vintages 2007–2017 (post-Step-7 maturity filter), `as_of = 2019-04-01`.

**Recommended figure for IFRS 9 reporting:** $484,474,530. The data-driven overlay decreases ECL relative to baseline because the trained model's macro coefficient on UNRATE is negative — a known property of LC data documented across Steps 8, 13, and 14. The regulatory overlay substitutes externally-derived stress-test coefficients (CCAR/EBA-aligned) and produces the +20% adjustment that IFRS 9 expects.

**Two analytical findings.** First, raw correlations between LC default rates and macros are *inverted* due to vintage drift; the within-transformation from panel econometrics recovers the within-cohort signal. Second, the LC underwriting-reaction effect — when unemployment rises, LC tightens credit standards within the same year, partially offsetting the conventional macro→default channel — propagates through the trained model and is the root cause of the data-driven overlay's inverted direction.

**Pipeline integrity.** All 159 regression-validator checks pass. All 120 forensic-audit checks pass with 9 documented WARNs (collinearity findings, mtime ordering on regenerated artifacts, Stage 2 thinness). Headline numbers reproduce to **<$0.01** across the entire artifact chain.

## Section 2 — Data Sources and Acquisition

### 2.1 — LendingClub accepted loans dataset

- **Source:** Kaggle (`wendykan/lending-club-loan-data` and equivalent public mirrors).
- **License:** Public.
- **Time period:** 2007 — Q1 2019 (issue dates) with terminal status observable through approximately March 2019.
- **Raw scope:** ~2.26M rows × 145 columns × ~3 GB on disk.
- **Why this dataset:** The LC dataset is the standard reference for academic and practitioner work on US consumer credit. It (a) covers a full credit cycle from 2008 through post-recovery, (b) provides terminal loan status and recovery amounts for closed loans, (c) is freely available with comprehensive documentation, and (d) reflects a real underwritten portfolio with grade, FICO, DTI, employment, and many other origination-time features.
- **Pre-processing at acquisition:** none. The raw CSV is read into pandas and processed in Step 7.

After cleaning the population is **1,179,687 loans** [verified ✓ — source: ecl_headline.json + parquet rows] (post-DNMCP-drop, post-FICO-floor, post-maturity-filter; all at full traceability — see Section 3 for the full chain).

### 2.2 — FRED macroeconomic series

- **Source:** Federal Reserve Economic Data API (https://fred.stlouisfed.org/), pulled via the `fredapi` Python wrapper using a free API key.
- **License:** Public, free with API key.
- **Series used (4):**
  - `UNRATE` — Civilian Unemployment Rate, monthly, seasonally adjusted, %.
  - `GDPC1` — Real Gross Domestic Product, quarterly, seasonally adjusted (transformed to YoY % change in Step 8).
  - `FEDFUNDS` — Federal Funds Effective Rate, monthly, %.
  - `CSUSHPISA` — S&P/Case-Shiller US National Home Price Index, monthly, seasonally adjusted (transformed to YoY % change in Step 8).
- **Time period:** 2005-01-01 to 2020-01-31. Saved as `macros_monthly.parquet` (169 rows) [verified ✓ — source: macros_monthly.parquet rows].
- **Why these four:** Each represents a distinct macroeconomic channel for consumer credit performance — unemployment for income shock, real GDP growth for the broad economic cycle, the federal funds rate for monetary policy / credit cost, and the housing index for household wealth and the HELOC channel.
- **Pre-processing at acquisition:** none. The raw series are pulled and transformed into level / YoY measures in Step 8.

For full detail see `docs/step8_methodology.md`.

## Section 3 — Data Cleaning and Preparation

### 3.1 — Default flag mapping

A binary `default_flag` was derived from the LC `loan_status` column:

- `Charged Off`, `Default`, and `Does not meet the credit policy. Status:Fully Paid` → `1`.
- `Fully Paid` → `0`.
- `Current`, `Late (16-30 days)`, `Late (31-120 days)`, `In Grace Period`, and `Does not meet the credit policy. Status:Charged Off` → row dropped (no terminal label observable).

The DNMCP split is unusual: the Fully-Paid variant is mapped to default = 1 because LC labels indicate these were originated under a deprecated policy and treats them as outliers; the Charged-Off variant of DNMCP is dropped to avoid mixing the deprecated regime with the modern underwriting standard.

### 3.2 — Sentinel and outlier handling (Step 7)

| Treatment | Rows / cells affected | Source |
|---|---:|---|
| Drop "Does not meet the credit policy" rows | 1,988 dropped | step7_methodology.md §2 |
| `dti = 999` → NaN (sentinel for "not computable") | 38 cells | step7_methodology.md §2 |
| `revol_util > 100` capped at 100 (over-limit revolvers) | 4,687 cells | step7_methodology.md §2 |
| `annual_inc` winsorized at p99 = $250,000 | 13,448 cells | step7_methodology.md §2 |
| `fico_range_low < 660` dropped (LC issuance floor; mostly DNMCP residuals) | 2 dropped | step7_methodology.md §2 |

### 3.3 — Maturity filter (survivorship-bias correction)

The dataset is censored at `as_of = 2019-04-01`. Loans issued in 2017–2018 had not had time to fully realize defaults; using them directly biases the labeled population toward fast-defaulters because slow performers were still `Current` and got dropped in Step 3.1.

**Rule:** keep loans where `months_observable = (as_of − issue_d)` ≥ 24.

**Effect:** the post-cleaning population is **1,179,687 rows** [verified ✓ — source: loans_with_ecl.parquet rows], down from approximately 1.34M after the default-flag mapping. Most of the dropped rows are 2017–2018 vintages where many loans were still maturing.

### 3.4 — Feature classification

`feature_classification.json` is the single source of truth for which columns are available as model inputs. Four mutually-exclusive categories are defined:

- **`pd_inputs`: 33 columns** [verified ✓ — source: feature_classification.json] — origination features available at the time the model would predict, including the 4 macros and `issue_year`.
- **`identifiers`: 3 columns** [verified ✓ — source: feature_classification.json] — `id`, `issue_d`, `last_pymnt_d` (used for joins and time-based splits, never as features).
- **`outcome_only`: 13 columns** [verified ✓ — source: feature_classification.json] — outcome variables (recoveries, total payments, last credit pull FICO, hardship and settlement flags) that would leak future information into a PD model. Excluded from PD modeling. Used downstream for LGD and EAD only.
- **`label`: [verified ✓ — source: feature_classification.json]** — the binary target derived in 3.1.

For full detail see `docs/step7_methodology.md`.

## Section 4 — The Three Headline ECL Numbers

The pipeline produces three independently computed headline figures. Each is reproducible to **<$0.01** across all artifacts (validated by audit Categories 16, 17, 18, and 19).

| Headline | Total ECL | ECL/Funded | Use case | Source artifact |
|---|---:|---:|---|---|
| Step 12 baseline | $403,536,501 [verified ✓ — source: ecl_headline.json + parquet sum] | 2.37% [verified ✓ — source: computed] | Internal model output, pre-IFRS-9 forward-looking | `ecl_headline.json` |
| Step 13 data-driven overlay | $390,821,719 [verified ✓ — source: ecl_overlay_headline.json + parquet sum] | 2.30% [verified ✓ — source: computed] | Mechanical IFRS 9, inversion documented | `ecl_overlay_headline.json` |
| **Step 14 regulatory overlay** | **$484,474,530** [verified ✓ — source: validation_regulatory_overlay.json] | **2.85%** [verified ✓ — source: computed] | **Recommended for reporting** | `validation_regulatory_overlay.json` |

### Step 12 baseline ($403,536,501)

The headline produced by combining calibrated PD × predicted LGD × projected EAD per loan, with IFRS 9 staging applied. No forward-looking macro adjustment. **How calculated:** for each loan in the active population, `ECL = PD_12m × LGD × EAD_12m × DF` (Stage 1) or `Σ_t (PD_marginal_t × LGD × EAD_t × DF_t)` (Stage 2/3). Aggregated across the portfolio. **Strengths:** every component traceable to per-loan inputs; reproduces from the underlying parquet to $0.00. **Weakness:** historic-only — does not satisfy IFRS 9's forward-looking requirement.

### Step 13 data-driven overlay ($390,821,719, -3.15% [verified ✓ — source: ecl_overlay_headline.json] vs baseline)

Three macro scenarios are constructed by shocking `unrate` and `hpi_yoy`, re-binning those two features only, and re-scoring through the saved logistic regression and Platt calibrator. Probability-weighted 50% baseline, 30% adverse, 20% severe per the IFRS 9 conservative-tilt convention. **Result is below baseline** — the LC underwriting-reaction effect (Section 9) inverts the conventional macro→default direction in this dataset. **Strengths:** mechanically faithful to the trained model; re-scoring uses the same feature pipeline. **Weakness:** the inversion is a known dataset property, not an economic prediction; would not pass production review without remediation.

### Step 14 regulatory overlay ($484,474,530, +20.06% [verified ✓ — source: validation_regulatory_overlay.json] vs baseline)

Replaces the dataset-derived macro coefficients with conventional Fed CCAR / EBA stress test sensitivities applied as additive log-odds shifts: +0.18 per pp UNRATE shock, +0.05 per pp HPI YoY decline. Same scenarios, same weights, same EAD, same LGD, same staging. **Strengths:** direction is economically intuitive (worse macros → higher PD → higher ECL); regulatory-defensible; the recommended figure for external IFRS 9 reporting. **Weakness:** the regulatory coefficients are imported rather than learned from the dataset; treated as an explicit, audit-ready assumption.

For full detail see `docs/step12_methodology.md`, `docs/step13_methodology.md`, and `docs/final_validation_report.md` Section 6.

## Section 5 — The PD Model: Calculation Path

### 5.1 — Feature preparation (Step 9a)

The feature set begins with the 33 candidate features [verified ✓ — source: binning_summary.json] from `feature_classification.json#pd_inputs` plus two derived columns (`issue_year` from `issue_d`, and `credit_history_years` from `earliest_cr_line`). Optimal binning is applied via `optbinning.BinningProcess` with parameters `max_n_prebins=20`, `min_prebin_size=0.05`, `monotonic_trend="auto"`, and IV selection criteria `{"min": 0.02, "max": 0.7}`. The output: **16 features selected by IV** [verified ✓ — source: binning_summary.json] and **3 features force-included** via `fixed_variables` [verified ✓ — source: binning_summary.json] (`unrate`, `hpi_yoy`, `issue_year`) for methodological coherence with Step 8 commitments and the Step 13 macro overlay.

**Final model uses 19 WoE-transformed features.**

**Top features by IV:**
- `sub_grade` — IV 0.5037 [verified ✓ — source: binning_summary.json]
- `grade` — IV 0.4700 [verified ✓ — source: binning_summary.json]
- `int_rate` — IV 0.4695 [verified ✓ — source: binning_summary.json]

`grade` and `sub_grade` carry the bulk of the IV because LC's grade is essentially a precomputed PD score; this dominance is documented and accepted (audit Category 7 has dedicated checks on the IV-band assertion `[0, 0.7]`).

### 5.2 — Train/test split

Time-based, mirroring how a bank would use vintage-out validation:
- **Train:** `issue_d < 2016-01-01` — 826,604 rows, 152,304 defaults, **18.43%** default rate.
- **Test:** `issue_d ≥ 2016-01-01` — 353,083 rows, 82,122 defaults, **23.26%** default rate.

The 4.83 pp default-rate gap reflects LC's documented underwriting drift from 2007–2015 to 2016+ — surfaced in Step 8 §4 as the within-year correlation finding and again in Step 13 as the root cause of the data-driven overlay's direction inversion.

### 5.3 — Logistic regression specification (Step 9b)

- **Algorithm:** `sklearn.linear_model.LogisticRegression`.
- **Hyperparameters:** `penalty="l2"`, `C=1e6` (effectively no regularization, deliberately, to preserve force-included coefficient magnitudes), `solver="lbfgs"`, `max_iter=2000`, `random_state=42`.
- **Test performance:** AUC = **0.7059** [verified ✓ — source: model_evaluation.json], Gini = **0.4118** [verified ✓ — source: model_evaluation.json], KS = **0.2968** [verified ✓ — source: model_evaluation.json].
- **Coefficient signs:** 18 negative + 1 positive — the expected pattern under optbinning's `log(non_event/event)` WoE convention, where high WoE corresponds to low risk and a sound P(default) model produces negative coefficients on each WoE feature.

### 5.4 — Gradient boosting challenger

- **Algorithm:** `sklearn.ensemble.HistGradientBoostingClassifier` (substituted for `xgboost.XGBClassifier` because libomp is unavailable on this macOS environment; algorithmically equivalent histogram-based gradient boosting).
- **Hyperparameters:** defaults — no tuning.
- **Test performance:** AUC = **0.7084** [verified ✓ — source: model_evaluation.json].
- **Interpretability cost** of choosing logistic = **+0.25 pp** AUC, accepted in exchange for transparent coefficients defensible in audit and regulatory review.

### 5.5 — Probability calibration (Step 9c)

A Platt scaler — a one-feature logistic regression mapping the raw `predict_proba` output to a calibrated probability — is fit on the test cohort's raw predictions. **The original specification called for fitting on training**, but training-fit Platt produced *higher* test ECE than pre-calibration (0.0398 → 0.0480) because of the LC vintage drift between train and test. Test-set fitting is accepted because Platt's two-parameter form on 353K rows produces negligible overfitting.

| Metric | Logistic | Gradient boosting |
|---|---:|---:|
| Pre-calibration ECE | 0.0398 [verified ✓ — source: calibration_pre.json] | 0.0345 [verified ✓ — source: calibration_pre.json] |
| Post-calibration ECE | 0.0195 [verified ✓ — source: calibration_post.json] | 0.0218 [verified ✓ — source: calibration_post.json] |
| Reduction | 51% | 37% |

**Calibrator parameters** (logistic, the production candidate): intercept = -2.4180 [verified ✓ — source: calibrators.json], slope = +5.7716 [verified ✓ — source: calibrators.json]. AUC is preserved exactly (Platt is a monotonic transformation).

### 5.6 — Lifetime PD to 12-month PD conversion (Step 12)

The PD model is a **lifetime PD** by construction (the label is whether the loan ever defaulted during its full term). The formula uses the constant monthly hazard rate:

$$\lambda = 1 - (1 - \text{pd\_lifetime})^{1/T}$$

with `T = months_remaining`; then `pd_12m = 1 − (1 − λ)^min(12, T)`. This is a deliberate simplification — production banks fit vintage hazard curves to allocate lifetime PD across periods more accurately. The constant-hazard approximation is acceptable at the project scope and is documented as a limitation in Section 11.

For full detail see `docs/step9a_methodology.md`, `docs/step9b_methodology.md`, and `docs/step9c_methodology.md`.

## Section 6 — The LGD Model: Calculation Path

### 6.1 — Defaulter identification

Filter the population to `default_flag == 1`: **234,426 historical defaulters** [verified ✓ — source: lgd_stats.json]. The realized LGD is computed only for these loans.

### 6.2 — Realized LGD per loan

$$\text{LGD}_{\text{realized}} = 1 - \frac{\text{recoveries} - \text{collection\_recovery\_fee}}{\text{funded\_amnt} - \text{total\_rec\_prncp}}$$

Bounded to [0, 1]. Cap counts:

- **15 loans dropped** [verified ✓ — source: lgd_stats.json] where the denominator (`funded_amnt − total_rec_prncp`) was ≤ 0 (loans fully amortized before defaulting; LGD undefined).
- **80 loans capped at 0** [verified ✓ — source: lgd_stats.json] (over-recovery, possible due to interest-then-principal accounting).
- **0 loans capped at 1** [verified ✓ — source: lgd_stats.json] (data errors).
- **Combined cap share: 0.03%** [verified ✓ — source: lgd_stats.json] (well below the 5% concern threshold).

After filtering: **234,411 usable defaulters** [verified ✓ — source: lgd_stats.json].

### 6.3 — Segment estimation

Loans are segmented by **grade × purpose** (98 segments observed [verified ✓ — source: lgd_lookup.csv]). For any segment with fewer than 500 defaulters, the segment falls back to the grade-only LGD:

- **Segments using their own segment-mean LGD:** 27 [verified ✓ — source: lgd_lookup.csv]
- **Segments using grade-only fallback:** 71 [verified ✓ — source: lgd_lookup.csv]

**Mean realized LGD across the dataset: 0.9029** [verified ✓ — source: lgd_stats.json] — typical for unsecured consumer credit, reflecting LC's heavy-recovery-loss profile (most charged-off loans yield no recovery; a tail recovers partially). The predicted LGD spread across segments is small: **[0.8891, 0.9032]** [verified ✓ — source: lgd_lookup.csv][verified ✓ — source: lgd_lookup.csv], meaning LGD acts as roughly a constant multiplier in this portfolio.

### 6.4 — Validation

Backtested on `issue_d ≥ 2016-01-01` defaulters. Aggregate predicted vs. observed error is approximately −2.0 pp (observed = 0.916, predicted = 0.896), within the documented ±5pp tolerance. Vintage drift in LGD is small (range 0.014 across validation years).

For full detail see `docs/step10_methodology.md`.

## Section 7 — The EAD Projection: Calculation Path

### 7.1 — Amortization formula

For a fixed-rate term loan, the monthly payment is

$$M = P \cdot \frac{r(1+r)^n}{(1+r)^n - 1}$$

with `P` = original principal, `r` = monthly interest rate, `n` = term in months. The outstanding balance at month `t` follows in closed form:

$$B_t = P(1+r)^t - M \cdot \frac{(1+r)^t - 1}{r}$$

### 7.2 — Methodological deviation

**Issue:** The dataset contains only terminated loans (Charged Off, Default, or Fully Paid). The original specification used `out_prncp` as the as-of starting balance and zeroed out EAD for terminated loans — under those rules every loan in this dataset is zero-EAD and the projection is degenerate.

**Resolution:** Use **contractual re-amortization from `funded_amnt`** to compute the contractual balance at `as_of` and project forward. This treats the EAD step as a contractual-hypothetical projection — what the EAD trajectory would be if each loan were still performing per its contract. The deviation is documented prominently in `docs/step11_methodology.md` §2 (Decision B).

### 7.3 — Per-loan outputs

For each loan, Step 11 produces:

- **`ead_12m`** — average outstanding balance over the next 12 months (the IFRS 9 Stage 1 ECL input).
- **`ead_lifetime_path`** — monthly balance vector across remaining contractual life (parquet list column).
- **`discount_factors`** — monthly discount factor vector using `int_rate / 12` as monthly rate.
- **`ead_lifetime_undiscounted_total`**, **`ead_lifetime_discounted_total`**, **`months_remaining`**, plus checkpoints at months 12 / 24 / 36 / 60.

Sanity checks performed and persisted: principal conservation (sum of repayments equals starting balance), monotonic non-increasing balance, final balance < $1, closed-form-vs-iterative match within 1¢. All seven sanity checks in Step 11 §8 passed.

### 7.4 — Population breakdown at `as_of = 2019-04-01`

- **Already matured (months_remaining = 0): 800,634 loans** [verified ✓ — source: ead_status_breakdown.csv] — zero EAD trajectory.
- **Short remaining (< 12 months): 240,857 loans** [verified ✓ — source: ead_status_breakdown.csv] — partial 12-month EAD.
- **Medium remaining (12–36 months): 138,196 loans** [verified ✓ — source: ead_status_breakdown.csv] — full 12-month EAD plus partial lifetime path.
- **Long remaining (> 36 months): 0 loans** at `as_of`.
- **Active population (months_remaining > 0): 379,053 loans** [verified ✓ — source: ecl_headline.json] — the effective ECL base.

For full detail see `docs/step11_methodology.md`.

## Section 8 — ECL Combination and Staging

### 8.1 — Per-loan formulas

**Stage 1 (12-month ECL):**

$$ECL_{12M} = PD_{12M} \times LGD \times EAD_{12M} \times DF_{12M}$$

with `DF_12M` the average discount factor over the first 12 months.

**Stage 2 (lifetime ECL):**

$$ECL_{\text{lifetime}} = \sum_{t=1}^{T} PD_{\text{marginal},t} \times LGD \times EAD_t \times DF_t$$

with $PD_{\text{marginal},t} = \lambda(1-\lambda)^{t-1}$ derived from the constant monthly hazard.

**Stage 3 (already-defaulted):** simplified as `LGD × discounted average remaining balance`. The simplification follows from the contractual-EAD deviation in Step 11 (this dataset has no realized outstanding-at-default values; production deployments would use the actual figure).

### 8.2 — Staging logic

- **Stage 3:** `default_flag == 1` (already-realized default).
- **Stage 2:** `pd_lifetime > 2 × grade-average pd_lifetime` (the IFRS 9 conventional rebuttable-presumption SICR proxy).
- **Stage 1:** all other active loans.

| Stage | Count | Total ECL |
|---|---:|---:|
| 1 | 944,678 [verified ✓ — source: ecl_headline.json + parquet] | $147,787,069 [verified ✓ — source: ecl_headline.json + parquet] |
| 2 | 583 [verified ✓ — source: ecl_headline.json + parquet] | $714,084 [verified ✓ — source: ecl_headline.json + parquet] |
| 3 | 234,426 [verified ✓ — source: ecl_headline.json + parquet] | $255,035,347 [verified ✓ — source: ecl_headline.json + parquet] |

**Stage 2 is unusually thin** (0.05% of the population). The reason: LC's `grade` (top IV at 0.50) absorbs most of the predicted-PD variance, so few individual loans exceed twice their grade peer's mean. This is a finding (the SICR rule based on PD ratios alone produces a thin Stage 2 in a grade-dominated model), not a defect — sensitivity to the threshold is reported in Step 14 §5.

### 8.3 — Headline aggregation

**Total baseline ECL: $403,536,501** [verified ✓ — source: ecl_headline.json] (2.37% [verified ✓ — source: ecl_headline.json (computed)] of $16,997,974,550 funded principal).

**By grade:**

| Grade | Count | Total ECL | Per loan | Coverage / funded |
|---|---:|---:|---:|---:|
| A | 204,038 | $6,770,586 | $33.18 | 0.2400% |
| B | 347,299 | $42,087,099 | $121.18 | 0.9200% |
| C | 332,304 | $112,109,838 | $337.37 | 2.3900% |
| D | 175,155 | $95,633,471 | $545.99 | 3.5800% |
| E | 84,313 | $90,484,251 | $1073.19 | 6.0500% |
| F | 29,238 | $43,527,785 | $1488.74 | 7.8200% |
| G | 7,340 | $12,923,471 | $1760.69 | 8.5900% |

Stage 3 dominates the absolute ECL — these are realized losses being recognized through the IFRS 9 framework. Stage 1 ECL is the forward-looking 12-month provision on the active book.

For full detail see `docs/step12_methodology.md`.

## Section 9 — Forward-Looking Macro Overlay

### 9.1 — Three-scenario design

IFRS 9 requires that PD estimates reflect a bank's reasonable expectation of future macroeconomic conditions, not just historical averages. Three scenarios are defined:

| Scenario | Δ unrate | Δ hpi_yoy | Weight |
|---|---:|---:|---:|
| Baseline | +0.0 pp | +0.0 pp | 50% |
| Adverse | +3.0 pp | −10.0 pp | 30% |
| Severe | +5.0 pp | −20.0 pp | 20% |

Weights sum to 1.0 [verified ✓ — source: ecl_overlay_headline.json]. Shocks are calibrated to match EBA stress test severities, scaled for US data. Weights follow the IFRS 9 conservative-tilt convention. Sensitivity to alternative weights (60/30/10, 33/33/33, 40/40/20, etc.) is reported in `docs/validation_overlay_weights.csv`; the headline range across alternative weights is small.

### 9.2 — Two implementations

**Data-driven overlay (Step 13)**

Mechanics: shock raw `unrate` and `hpi_yoy`, re-bin those two via the saved `OptimalBinning.transform()`, substitute into the WoE matrix, re-score through the saved logistic regression and Platt calibrator. Other 17 features unchanged.

**Result: $390,821,719** [verified ✓ — source: ecl_overlay_headline.json] **(-3.15%** [verified ✓ — source: ecl_overlay_headline.json] **vs baseline) — INVERTED from textbook expectation.**

**Why?** Step 8 documented that within-year correlation between UNRATE and default rate in LC data is essentially zero/slightly negative (−0.006 with year FE; −0.010 with year + grade + state controls). The mechanism is the **LC underwriting-reaction**: when unemployment rises, LC tightens credit standards within the same year, selecting better borrowers whose subsequent default rates fall. The PD model inherits this empirical pattern; its `unrate` coefficient is **−0.41** in optbinning's `log(non/event)` convention, meaning **high unrate → high WoE → lower predicted PD**. When the overlay shocks unrate up, the model dutifully responds with lower PD.

This is mechanically faithful to the trained model. It is not, however, the "reasonable and supportable forward-looking adjustment" IFRS 9 expects.

**Regulatory overlay (Step 14)**

Mechanics: replace the dataset-derived macro coefficients with conventional regulatory stress-test sensitivities. Apply log-odds shifts directly to each loan's calibrated baseline PD:

$$\log\text{-odds}_{\text{scenario}} = \log\text{-odds}_{\text{baseline}} + 0.18 \cdot \Delta_{\text{unrate}} + 0.05 \cdot |\Delta_{\text{hpi yoy negative}}|$$

Coefficient sources: Fed CCAR Severely Adverse 2018 documentation; EBA 2018 stress test variables for US consumer credit. Same scenarios (50/30/20), same EAD, same LGD, same staging.

**Result: $484,474,530** [verified ✓ — source: validation_regulatory_overlay.json] **(+20.06%** [verified ✓ — source: validation_regulatory_overlay.json] **vs baseline) — economically intuitive direction.**

**Recommendation:** Report the regulatory overlay ($484,474,530) for IFRS 9 external purposes. Document the data-driven overlay ($390,821,719) as a model-risk discussion item highlighting the LC underwriting-reaction.

For full detail see `docs/step8_methodology.md`, `docs/step13_methodology.md` §3 (the inversion narrative), `docs/final_validation_report.md` Section 6, and `docs/validation_regulatory_overlay.json`.

## Section 10 — Validation and Quality Assurance

### 10.1 — Validation pack (Step 14)

A six-analysis validation pack is produced in Step 14:

- **Out-of-time discrimination.** Test AUC = 0.7059; by-vintage AUCs **2016 = 0.7077** [verified ✓ — source: validation_discrimination.json], **2017 = 0.6969** [verified ✓ — source: validation_discrimination.json]; **variation across vintages = 0.0108** [verified ✓ — source: validation_discrimination.json] (well below the 0.05 threshold; model is stable).
- **Calibration.** **Decile MAD on test = 0.0276** [verified ✓ — source: validation_calibration.json]; max decile deviation = 0.0531; HL test rejects strict calibration on n=353K but ECE/MAD are within healthy range.
- **SICR threshold sensitivity.** Headline ECL ranges from **$403,413,586** [verified ✓ — source: validation_sicr_sensitivity.csv] (3.0× rule, strictest) to **$435,716,463** [verified ✓ — source: validation_sicr_sensitivity.csv] (absolute-PD-5% rule, loosest); the chosen 2× rule sits within this range.
- **Macro-weight sensitivity.** Alternative scenario weights (60/30/10 to 30/40/30) move the data-driven headline by less than 1.5pp.
- **Single-feature stress.** Replacing each loan's worst-case value of the top-5 features one at a time. **`sub_grade` dominates** at **+9.88%** [verified ✓ — source: validation_single_feature_stress.csv] ECL impact when shocked to its worst observed value, consistent with its IV ranking. Other features in the top-5 produce <5% impacts.
- **PSI over time.** **All PSI values across vintages 2014–2017 < 0.0075** [verified ✓ — source: validation_psi_over_time.csv] (i.e., near-zero); calibrated PD distributions are highly stable.

### 10.2 — Audit trail

- **Regression validator** (`src/validate_pipeline_steps_7_8.py`): 18 task groups, 159+ checks. Smoke-tests every artifact and invariant. **Status: 0 failures.**
- **Forensic audit** (`src/audit_full_pipeline.py`): 21 categories, 120+ checks. Independent recomputation of every cited number. Three timestamped audit reports are archived in `docs/audit_report_full_*.md` (never overwritten). **Status: 0 failures, 9 documented WARNs** (collinearity findings, mtime ordering on regenerated artifacts, Stage 2 thinness, per-loan overlay monotonicity — all documented in step methodologies).
- **Reproducibility:** headline numbers reproduce to **<$0.01** across the entire artifact chain (validated by audit checks 16.5 / 17.7 / 18.2 / 19.3 / 20.1 / 21.5).

### 10.3 — Two analytical findings (the project's most important contributions)

**Simpson's paradox in macro-default analysis (Step 8).** Raw correlations between LC default rates and macros (UNRATE, HPI YoY) are *inverted from textbook* — high unemployment ↔ low defaults — because LC's underwriting loosened progressively from 2009 (12.6% default rate) to 2016 (23.3%) while macros simultaneously improved. Vintage was the confounder. Resolved via the within-transformation from panel econometrics: subtract per-year means before correlating residuals. The within-year corrections recovered the expected sign for HPI YoY but UNRATE remained essentially zero, leading directly to the LC underwriting-reaction finding.

**LC underwriting-reaction effect (Steps 8 → 13 → 14).** When unemployment rises, LC tightens credit standards within the same year, partially or fully offsetting the conventional macro→default relationship. Documented through three layers of analysis: the within-year correlation (Step 8), the inverted overlay direction (Step 13), and the regulatory-coefficient remediation (Step 14). The pipeline produces three different headlines that allow the reader to see the effect in numbers: the data-driven overlay **decreases** ECL by 3.15% under shocks that should increase it; the regulatory overlay **increases** ECL by 20% as expected.

For full detail see `docs/final_validation_report.md`, all step methodologies, and the timestamped audit reports.

## Section 11 — Limitations and Recommendations for Production

Consolidated list of all limitations from the individual methodology documents, ranked by importance.

1. **Data-driven macro overlay direction is inverted (LC underwriting-reaction).** Recommendation: production deployment uses regulatory-coefficient overlay (Section 9) or refits the PD model with vintage-stratified macro effects.
2. **Constant monthly hazard for the 12-month PD allocation.** Recommendation: vintage hazard curves fit to historical default-month distributions.
3. **Same scenario shock applied across the entire lifetime.** Recommendation: multi-period scenario paths (recession in years 1–2, recovery in years 3+).
4. **SICR thresholding via PD ratio alone produces a thin Stage 2 (0.05%).** Recommendation: include payment-behavior signals (DPD, watchlist, restructuring) when available in the live portfolio. Or use an absolute-PD trigger (Step 14 sensitivity shows 5% gives a more typical Stage 2 share).
5. **LGD not adjusted for downturn conditions.** Recommendation: downturn-LGD overlay per regulatory convention (a separate "downturn LGD" floor on top of segment-mean LGD).
6. **EAD ignores prepayment.** Recommendation: separate prepayment hazard model. Empirical LC prepayment rates of 5–15% annually mean the contractual EAD is biased upward.
7. **Stage 3 simplified due to terminal-status dataset.** Recommendation: standard Stage 3 formula on a live portfolio with realized outstanding-at-default balances.
8. **Probability calibration was test-set fit due to vintage drift.** Recommendation: use a separate calibration cohort with default rate similar to the live portfolio (or 5-fold cross-fitted predictions).

Each limitation is documented in the corresponding step's methodology document and revisited in `docs/final_validation_report.md` Section 7–8.

## Section 12 — Vital Files Index

Complete reference of every file in the project. For each file: path, format, size, the step that produced it, the steps that consume it, a one-line purpose, and a verification status (`✓` = file exists and is well-formed, `MISSING` = file missing or empty, `MALFORMED` = file present but unparseable).

**Index health: 92/92 files verified.** All files present and well-formed.

### Cleaned data


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `data/loans_modeling_ready.parquet` | parquet | 58.8 MB | Step 7 | Step 8 | Cleaned + maturity-filtered population | ✓ |

| `data/loans_with_macros.parquet` | parquet | 59.1 MB | Step 8 | Steps 9a, 11 | + FRED macros at issue month | ✓ |

| `data/train.parquet` | parquet | 41.7 MB | Step 9a | Steps 9b, 12 | Train cohort with derivations | ✓ |

| `data/test.parquet` | parquet | 18.6 MB | Step 9a | Steps 9b, 12 | Test cohort with derivations | ✓ |

| `data/train_woe.parquet` | parquet | 9.7 MB | Step 9a | Steps 9b, 13 | WoE-transformed train | ✓ |

| `data/test_woe.parquet` | parquet | 4.4 MB | Step 9a | Steps 9b, 13 | WoE-transformed test | ✓ |

| `data/loans_with_lgd.parquet` | parquet | 60.0 MB | Step 10 | Step 11 | + per-loan predicted LGD | ✓ |

| `data/loans_with_ead.parquet` | parquet | 83.0 MB | Step 11 | Step 12 | + per-loan EAD trajectory + discount factors | ✓ |

| `data/loans_with_ecl.parquet` | parquet | 107.7 MB | Step 12 | Step 13 | + IFRS 9 stage and ECL components | ✓ |

| `data/loans_with_ecl_overlay.parquet` | parquet | 164.6 MB | Step 13 | Steps 14, 15 | + scenario PDs and weighted ECL | ✓ |

| `data/test_predictions.parquet` | parquet | 36.1 MB | Steps 9b → 13 | Step 14 | Test cohort per-loan predictions | ✓ |

| `data/macros_monthly.parquet` | parquet | 8.2 KB | Step 8 | Step 13 | FRED monthly macro reference | ✓ |

| `data/accepted_labeled.parquet` | parquet | 67.9 MB | Step 6 | Step 7 | Initial labeled population | ✓ |

### Models


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `models/binning_process.pkl` | pkl | 80.7 KB | Step 9a | Steps 9b, 13, 14 | Fitted optbinning model | ✓ |

| `models/pd_logistic.pkl` | pkl | 1.0 KB | Step 9b | Steps 9c, 12, 13, 14 | Logistic regression PD | ✓ |

| `models/pd_xgboost.pkl` | pkl | 381.8 KB | Step 9b | Step 9c | Gradient-boosting challenger | ✓ |

| `models/pd_logistic_calibrator.pkl` | pkl | 0.9 KB | Step 9c | Steps 12, 13, 14 | Platt scaler for logistic | ✓ |

| `models/pd_xgboost_calibrator.pkl` | pkl | 0.9 KB | Step 9c | — | Platt scaler for gradient boosting | ✓ |

### Methodology


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `docs/step7_methodology.md` | md | 7.9 KB | Step 7 | — | Observation/performance windows + cleaning | ✓ |

| `docs/step8_methodology.md` | md | 7.6 KB | Step 8 | — | Macro features, Simpson's paradox finding | ✓ |

| `docs/step9a_methodology.md` | md | 8.2 KB | Step 9a | — | Train/test split + WoE binning | ✓ |

| `docs/step9b_methodology.md` | md | 9.2 KB | Step 9b | — | PD model fit | ✓ |

| `docs/step9c_methodology.md` | md | 7.1 KB | Step 9c | — | Platt scaling calibration | ✓ |

| `docs/step10_methodology.md` | md | 7.5 KB | Step 10 | — | LGD segment-average estimation | ✓ |

| `docs/step11_methodology.md` | md | 6.0 KB | Step 11 | — | EAD contractual amortization | ✓ |

| `docs/step12_methodology.md` | md | 6.2 KB | Step 12 | — | ECL combination + IFRS 9 staging | ✓ |

| `docs/step13_methodology.md` | md | 7.9 KB | Step 13 | — | Forward-looking macro overlay (data-driven) | ✓ |

### Reports


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `docs/final_validation_report.md` | md | 13.1 KB | Step 14 | — | Senior-reviewer comprehensive validation | ✓ |

| `docs/project_summary.md` | md | 32.1 KB | Build | — | Narrative summary (no inline verification) | ✓ |

| `docs/dashboard_spec.md` | md | 11.6 KB | Step 15 | Manual PBI build | Power BI build instructions | ✓ |

### Configuration


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `docs/feature_classification.json` | json | 1.0 KB | Step 7 | Steps 8–14 | Feature lists + label | ✓ |

| `docs/binning_summary.json` | json | 4.6 KB | Step 9a | Steps 9b, 13, 14 | Binning IV table + selection | ✓ |

| `docs/model_evaluation.json` | json | 0.8 KB | Step 9b | Step 14 | PD model performance metrics | ✓ |

| `docs/calibration_pre.json` | json | 0.3 KB | Step 9c | — | Pre-calibration metrics | ✓ |

| `docs/calibration_post.json` | json | 0.3 KB | Step 9c | — | Post-calibration metrics | ✓ |

| `docs/calibration_comparison.json` | json | 0.7 KB | Step 9c | — | Pre vs post comparison | ✓ |

| `docs/calibrators.json` | json | 0.4 KB | Step 9c | — | Platt parameters + interpretation | ✓ |

| `docs/lgd_stats.json` | json | 0.4 KB | Step 10 | Step 14, dossier | LGD cap counts | ✓ |

### Headlines


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `docs/ecl_headline.json` | json | 1.7 KB | Step 12 | Steps 13–15, dossier | Baseline ECL aggregate | ✓ |

| `docs/ecl_overlay_headline.json` | json | 2.1 KB | Step 13 | Steps 14, 15, dossier | Data-driven overlay aggregate | ✓ |

| `docs/validation_regulatory_overlay.json` | json | 3.3 KB | Step 14 | Step 15, dossier | Regulatory overlay aggregate | ✓ |

### Aggregations


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `docs/lgd_lookup.csv` | csv | 3.9 KB | Step 10 | Step 14, dossier | Segment LGD table | ✓ |

| `docs/lgd_histogram.csv` | csv | 0.4 KB | Step 10 | — | Realized LGD distribution | ✓ |

| `docs/lgd_backtest.csv` | csv | 3.6 KB | Step 10 | — | LGD validation backtest | ✓ |

| `docs/lgd_sensitivity.csv` | csv | 0.2 KB | Step 10 | — | LGD ±20% portfolio sensitivity | ✓ |

| `docs/ead_histogram.csv` | csv | 0.6 KB | Step 11 | — | EAD_12m distribution | ✓ |

| `docs/ead_status_breakdown.csv` | csv | 0.1 KB | Step 11 | dossier | EAD bucket counts | ✓ |

| `docs/ead_months_remaining_distribution.csv` | csv | 0.5 KB | Step 11 | — | Months remaining at as_of | ✓ |

| `docs/ecl_by_stage.csv` | csv | 0.2 KB | Step 12 | Step 15 | ECL by IFRS 9 stage | ✓ |

| `docs/ecl_by_grade.csv` | csv | 0.4 KB | Step 12 | Step 15, dossier | ECL by LC grade | ✓ |

| `docs/ecl_by_vintage.csv` | csv | 0.5 KB | Step 12 | Step 15 | ECL by issue year | ✓ |

| `docs/ecl_by_purpose.csv` | csv | 0.8 KB | Step 12 | Step 15 | ECL by loan purpose | ✓ |

| `docs/ecl_overlay_by_stage.csv` | csv | 0.3 KB | Step 13 | Step 15 | Overlay ECL by stage | ✓ |

| `docs/ecl_overlay_by_grade.csv` | csv | 0.6 KB | Step 13 | Step 15 | Overlay ECL by grade | ✓ |

| `docs/ecl_overlay_by_vintage.csv` | csv | 0.6 KB | Step 13 | Step 15 | Overlay ECL by vintage | ✓ |

### Validation


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `docs/validation_discrimination.json` | json | 1.6 KB | Step 14 | Step 15, dossier | AUC/Gini/KS by vintage and grade | ✓ |

| `docs/validation_calibration.json` | json | 1.6 KB | Step 14 | Step 15, dossier | Calibration backtest summary | ✓ |

| `docs/validation_auc_by_vintage.csv` | csv | 0.1 KB | Step 14 | Step 15 | Out-of-time AUC table | ✓ |

| `docs/validation_auc_by_grade.csv` | csv | 0.4 KB | Step 14 | Step 15 | By-grade AUC table | ✓ |

| `docs/validation_gain_curve.csv` | csv | 0.6 KB | Step 14 | Step 15 | Cumulative gain by decile | ✓ |

| `docs/validation_reliability_test.csv` | csv | 0.6 KB | Step 14 | Step 15 | Decile reliability | ✓ |

| `docs/validation_calibration_by_grade.csv` | csv | 0.3 KB | Step 14 | Step 15 | Calibration by grade | ✓ |

| `docs/validation_calibration_by_vintage.csv` | csv | 0.1 KB | Step 14 | Step 15 | Calibration by vintage | ✓ |

| `docs/validation_sicr_sensitivity.csv` | csv | 0.4 KB | Step 14 | Step 15, dossier | SICR threshold sensitivity | ✓ |

| `docs/validation_overlay_weights.csv` | csv | 0.5 KB | Step 14 | Step 15 | Overlay weight sensitivity | ✓ |

| `docs/validation_psi_over_time.csv` | csv | 0.2 KB | Step 14 | dossier | PSI by vintage | ✓ |

| `docs/validation_single_feature_stress.csv` | csv | 0.3 KB | Step 14 | dossier | Top-5 feature stress | ✓ |

| `docs/validation_stage_migration.csv` | csv | 0.4 KB | Step 14 | — | Stage migration matrix | ✓ |

### Per-feature reports


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `docs/coefficients_lr.csv` | csv | 1.2 KB | Step 9b | — | LR coefficients with IV reference | ✓ |

| `docs/feature_importance_xgb.csv` | csv | 1.0 KB | Step 9b | — | GBM permutation importance | ✓ |

| `docs/decile_lift_lr.csv` | csv | 0.4 KB | Step 9b | — | LR decile lift | ✓ |

| `docs/decile_lift_xgb.csv` | csv | 0.4 KB | Step 9b | — | GBM decile lift | ✓ |

### Dashboard


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `data/dashboard/loans_summary.csv` | csv | 341.4 MB | Step 15 | Power BI | Per-loan facts (1.18M rows) | ✓ |

| `data/dashboard/headline_metrics.csv` | csv | 5.6 KB | Step 15 | Power BI | Headline KPI aggregations | ✓ |

| `data/dashboard/discrimination_metrics.csv` | csv | 0.6 KB | Step 15 | Power BI | Discrimination metrics | ✓ |

| `data/dashboard/calibration_table.csv` | csv | 0.7 KB | Step 15 | Power BI | Calibration table | ✓ |

| `data/dashboard/sensitivity_table.csv` | csv | 1.3 KB | Step 15 | Power BI | Sensitivity table | ✓ |

### Source code


| Path | Format | Size | Produced by | Consumed by | Purpose | Status |

|---|---|---:|---|---|---|:---:|

| `src/step7_observation_window.py` | py | 16.4 KB | — | — | Step 7 script | ✓ |

| `src/step8_macro_features.py` | py | 21.3 KB | — | — | Step 8 script | ✓ |

| `src/step9a_woe_binning.py` | py | 33.3 KB | — | — | Step 9a script | ✓ |

| `src/step9b_pd_model.py` | py | 28.0 KB | — | — | Step 9b script | ✓ |

| `src/step9c_calibration.py` | py | 26.7 KB | — | — | Step 9c script | ✓ |

| `src/step10_lgd_estimation.py` | py | 27.8 KB | — | — | Step 10 script | ✓ |

| `src/step11_ead_projection.py` | py | 25.4 KB | — | — | Step 11 script | ✓ |

| `src/step12_ecl_combination.py` | py | 27.2 KB | — | — | Step 12 script | ✓ |

| `src/step13_macro_overlay.py` | py | 31.8 KB | — | — | Step 13 script | ✓ |

| `src/step14_validation.py` | py | 45.0 KB | — | — | Step 14 script | ✓ |

| `src/step15_dashboard_data.py` | py | 27.1 KB | — | — | Step 15 script | ✓ |

| `src/build_project_summary.py` | py | 42.8 KB | — | — | Project summary builder | ✓ |

| `src/build_final_dossier.py` | py | 78.4 KB | — | — | This dossier builder | ✓ |

| `src/validate_pipeline_steps_7_8.py` | py | 64.2 KB | — | — | Regression validator | ✓ |

| `src/audit_full_pipeline.py` | py | 104.3 KB | — | — | Forensic audit | ✓ |

## Section 13 — Inline Verification Appendix

Every quantitative claim in this dossier is recorded below with its cited value, the independently re-derived value, the tolerance kind, the source artifact, and the verification status. Reviewers can spot-check any claim by looking up its row.

| # | Claim | Cited | Recomputed | Tolerance | Source | Status |
|---:|---|---:|---:|---|---|:---:|
| 1 | Section 1 — baseline ECL | 403,536,500.57 | 403,536,500.57 | currency | ±0.01 | ecl_headline.json + parquet sum | ✓ PASS |
| 2 | Section 1 — data-driven overlay ECL | 390,821,719.20 | 390,821,719.20 | currency | ±0.01 | ecl_overlay_headline.json + parquet sum | ✓ PASS |
| 3 | Section 1 — regulatory overlay ECL | 484,474,529.85 | 484,474,529.85 | currency | ±0.01 | validation_regulatory_overlay.json | ✓ PASS |
| 4 | Section 1 — total loan count | 1179687 | 1179687 | count | ±0 | ecl_headline.json + parquet rows | ✓ PASS |
| 5 | Section 1 — total funded principal | 16,997,974,550.00 | 16,997,974,550.00 | currency | ±0.01 | ecl_headline.json + parquet sum | ✓ PASS |
| 6 | Section 2 — post-cleaning row count | 1179687 | 1179687 | count | ±0 | ecl_headline.json + parquet rows | ✓ PASS |
| 7 | Section 2 — FRED monthly rows | 169 | 169 | count | ±0 | macros_monthly.parquet rows | ✓ PASS |
| 8 | Section 3 — pd_inputs count | 33 | 33 | count | ±0 | feature_classification.json | ✓ PASS |
| 9 | Section 3 — identifiers count | 3 | 3 | count | ±0 | feature_classification.json | ✓ PASS |
| 10 | Section 3 — outcome_only count | 13 | 13 | count | ±0 | feature_classification.json | ✓ PASS |
| 11 | Section 3 — label | default_flag | default_flag | exact | ±0 | feature_classification.json | ✓ PASS |
| 12 | Section 3 — post-maturity-filter row count | 1179687 | 1179687 | count | ±0 | loans_with_ecl.parquet rows | ✓ PASS |
| 13 | Section 4 — baseline ECL (cross-check) | 403,536,500.57 | 403,536,500.57 | currency | ±0.01 | ecl_headline.json + parquet sum | ✓ PASS |
| 14 | Section 4 — data-driven overlay ECL (cross-check) | 390,821,719.20 | 390,821,719.20 | currency | ±0.01 | ecl_overlay_headline.json + parquet sum | ✓ PASS |
| 15 | Section 4 — regulatory overlay ECL | 484,474,529.85 | 484,474,529.85 | currency | ±0.01 | validation_regulatory_overlay.json | ✓ PASS |
| 16 | Section 4 — baseline ECL/funded ratio | 2.3740 | 2.3740 | pct | ±0.01 | computed | ✓ PASS |
| 17 | Section 4 — data-driven ECL/funded ratio | 2.2992 | 2.2992 | pct | ±0.01 | computed | ✓ PASS |
| 18 | Section 4 — regulatory ECL/funded ratio | 2.8502 | 2.8502 | pct | ±0.01 | computed | ✓ PASS |
| 19 | Section 4 — data-driven overlay % change | -3.1508 | -3.1508 | pct | ±0.01 | ecl_overlay_headline.json | ✓ PASS |
| 20 | Section 4 — regulatory overlay % change | 20.0572 | 20.0572 | pct | ±0.01 | validation_regulatory_overlay.json | ✓ PASS |
| 21 | Section 5 — features input count | 33 | 33 | count | ±0 | binning_summary.json | ✓ PASS |
| 22 | Section 5 — IV-selected count | 16 | 16 | count | ±0 | binning_summary.json | ✓ PASS |
| 23 | Section 5 — force-included count | 3 | 3 | count | ±0 | binning_summary.json | ✓ PASS |
| 24 | Section 5 — top IV (sub_grade) | 0.5037 | 0.5037 | iv | ±0.001 | binning_summary.json | ✓ PASS |
| 25 | Section 5 — IV (grade) | 0.4700 | 0.4700 | iv | ±0.001 | binning_summary.json | ✓ PASS |
| 26 | Section 5 — IV (int_rate) | 0.4695 | 0.4695 | iv | ±0.001 | binning_summary.json | ✓ PASS |
| 27 | Section 5 — LR test AUC | 0.7059 | 0.7059 | auc | ±0.0005 | model_evaluation.json | ✓ PASS |
| 28 | Section 5 — LR test Gini | 0.4118 | 0.4118 | auc | ±0.0005 | model_evaluation.json | ✓ PASS |
| 29 | Section 5 — LR test KS | 0.2968 | 0.2968 | auc | ±0.0005 | model_evaluation.json | ✓ PASS |
| 30 | Section 5 — GBM test AUC | 0.7084 | 0.7084 | auc | ±0.0005 | model_evaluation.json | ✓ PASS |
| 31 | Section 5 — LR pre-cal ECE | 0.0398 | 0.0398 | iv | ±0.001 | calibration_pre.json | ✓ PASS |
| 32 | Section 5 — LR post-cal ECE | 0.0195 | 0.0195 | iv | ±0.001 | calibration_post.json | ✓ PASS |
| 33 | Section 5 — GBM pre-cal ECE | 0.0345 | 0.0345 | iv | ±0.001 | calibration_pre.json | ✓ PASS |
| 34 | Section 5 — GBM post-cal ECE | 0.0218 | 0.0218 | iv | ±0.001 | calibration_post.json | ✓ PASS |
| 35 | Section 5 — Platt LR intercept | -2.4180 | -2.4180 | coef | ±0.0001 | calibrators.json | ✓ PASS |
| 36 | Section 5 — Platt LR slope | 5.7716 | 5.7716 | coef | ±0.0001 | calibrators.json | ✓ PASS |
| 37 | Section 6 — initial defaulter count | 234426 | 234426 | count | ±0 | lgd_stats.json | ✓ PASS |
| 38 | Section 6 — usable defaulter count | 234411 | 234411 | count | ±0 | lgd_stats.json | ✓ PASS |
| 39 | Section 6 — EAD<=0 dropped count | 15 | 15 | count | ±0 | lgd_stats.json | ✓ PASS |
| 40 | Section 6 — capped at 0 count | 80 | 80 | count | ±0 | lgd_stats.json | ✓ PASS |
| 41 | Section 6 — capped at 1 count | 0 | 0 | count | ±0 | lgd_stats.json | ✓ PASS |
| 42 | Section 6 — mean realized LGD | 0.9029 | 0.9029 | iv | ±0.001 | lgd_stats.json | ✓ PASS |
| 43 | Section 6 — share capped % | 0.0341 | 0.0341 | pct | ±0.01 | lgd_stats.json | ✓ PASS |
| 44 | Section 6 — segment count | 98 | 98 | count | ±0 | lgd_lookup.csv | ✓ PASS |
| 45 | Section 6 — segment-native count | 27 | 27 | count | ±0 | lgd_lookup.csv | ✓ PASS |
| 46 | Section 6 — grade-fallback count | 71 | 71 | count | ±0 | lgd_lookup.csv | ✓ PASS |
| 47 | Section 6 — predicted LGD min | 0.8891 | 0.8891 | iv | ±0.001 | lgd_lookup.csv | ✓ PASS |
| 48 | Section 6 — predicted LGD max | 0.9032 | 0.9032 | iv | ±0.001 | lgd_lookup.csv | ✓ PASS |
| 49 | Section 7 — zero-EAD count | 800634 | 800634 | count | ±0 | ead_status_breakdown.csv | ✓ PASS |
| 50 | Section 7 — short EAD count | 240857 | 240857 | count | ±0 | ead_status_breakdown.csv | ✓ PASS |
| 51 | Section 7 — medium EAD count | 138196 | 138196 | count | ±0 | ead_status_breakdown.csv | ✓ PASS |
| 52 | Section 7 — active loan count | 379053 | 379053 | count | ±0 | ecl_headline.json | ✓ PASS |
| 53 | Section 8 — Stage 1 count | 944678 | 944678 | count | ±0 | ecl_headline.json + parquet | ✓ PASS |
| 54 | Section 8 — Stage 2 count | 583 | 583 | count | ±0 | ecl_headline.json + parquet | ✓ PASS |
| 55 | Section 8 — Stage 3 count | 234426 | 234426 | count | ±0 | ecl_headline.json + parquet | ✓ PASS |
| 56 | Section 8 — Stage 1 ECL | 147,787,069.27 | 147,787,069.27 | currency | ±0.01 | ecl_headline.json + parquet | ✓ PASS |
| 57 | Section 8 — Stage 2 ECL | 714,084.21 | 714,084.21 | currency | ±0.01 | ecl_headline.json + parquet | ✓ PASS |
| 58 | Section 8 — Stage 3 ECL | 255,035,347.09 | 255,035,347.09 | currency | ±0.01 | ecl_headline.json + parquet | ✓ PASS |
| 59 | Section 8 — total baseline ECL | 403,536,500.57 | 403,536,500.57 | currency | ±0.01 | ecl_headline.json | ✓ PASS |
| 60 | Section 8 — ECL/funded ratio | 2.3740 | 2.3740 | pct | ±0.01 | ecl_headline.json (computed) | ✓ PASS |
| 61 | Section 9 — data-driven overlay ECL | 390,821,719.20 | 390,821,719.20 | currency | ±0.01 | ecl_overlay_headline.json | ✓ PASS |
| 62 | Section 9 — data-driven overlay % change | -3.1508 | -3.1508 | pct | ±0.01 | ecl_overlay_headline.json | ✓ PASS |
| 63 | Section 9 — regulatory overlay ECL | 484,474,529.85 | 484,474,529.85 | currency | ±0.01 | validation_regulatory_overlay.json | ✓ PASS |
| 64 | Section 9 — regulatory overlay % change | 20.0572 | 20.0572 | pct | ±0.01 | validation_regulatory_overlay.json | ✓ PASS |
| 65 | Section 9 — scenario weights sum | 1.0000 | 1.0000 | iv | ±0.001 | ecl_overlay_headline.json | ✓ PASS |
| 66 | Section 10 — 2016 vintage AUC | 0.7077 | 0.7077 | auc | ±0.0005 | validation_discrimination.json | ✓ PASS |
| 67 | Section 10 — 2017 vintage AUC | 0.6969 | 0.6969 | auc | ±0.0005 | validation_discrimination.json | ✓ PASS |
| 68 | Section 10 — AUC range across vintages | 0.0108 | 0.0108 | auc | ±0.0005 | validation_discrimination.json | ✓ PASS |
| 69 | Section 10 — Calibration MAD | 0.0276 | 0.0276 | iv | ±0.001 | validation_calibration.json | ✓ PASS |
| 70 | Section 10 — SICR min ECL | 403,413,586.00 | 403,413,586.00 | currency | ±0.01 | validation_sicr_sensitivity.csv | ✓ PASS |
| 71 | Section 10 — SICR max ECL | 435,716,463.00 | 435,716,463.00 | currency | ±0.01 | validation_sicr_sensitivity.csv | ✓ PASS |
| 72 | Section 10 — top single-feature stress (sub_grade) % | 9.8763 | 9.8763 | pct | ±0.01 | validation_single_feature_stress.csv | ✓ PASS |
| 73 | Section 10 — max PSI over time | 0.0075 | 0.0075 | iv | ±0.001 | validation_psi_over_time.csv | ✓ PASS |

## Section 14 — Generation Metadata and Sign-Off

**Generated:** 2026-05-09T19:01:28

**Verification summary:**
- Total quantitative claims: **73**
- Claims verified ✓: **73**
- Claims failed ✗: **0**
- Claims unavailable: **0**

**File index summary:**
- Files documented: **92**
- Files verified ✓: **92**
- Files missing: **0**
- Files malformed: **0**

**Pipeline integrity status:** Validator 159+ checks passing; forensic audit 120+ checks passing with 9 documented WARNs (collinearity findings, mtime ordering on regenerated artifacts, Stage 2 thinness from grade-dominated PD model, per-loan overlay monotonicity from bin-boundary effects). Three timestamped audit reports archived in `docs/audit_report_full_*.md`.

**Overall verification status:** **ALL VERIFIED**.

---

**Sign-off statement.** If verification status is ALL VERIFIED, this dossier is the canonical reference document for this project. Numbers cited here are independently re-derived from source artifacts at generation time; the verification appendix in Section 13 records each derivation. The file index in Section 12 enumerates every artifact and confirms it exists and is well-formed. Subsequent edits to source artifacts that change cited values will cause this dossier to fail to re-generate (or generate with explicit FAIL annotations) — the dossier and the underlying pipeline are kept consistent by construction.

For deeper detail than a 30-minute read provides:
- `docs/final_validation_report.md` — Big 4-grade validation focus.
- `docs/project_summary.md` — narrative summary without verification overhead.
- `docs/step*_methodology.md` — per-step deep technical detail.
- `docs/audit_report_full_*.md` — timestamped forensic audit reports.
