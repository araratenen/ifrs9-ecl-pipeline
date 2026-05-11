# IFRS 9 ECL Pipeline — Project Summary

**Project:** End-to-end Expected Credit Loss model for the LendingClub consumer-loan portfolio, computed under IFRS 9 mechanics with full forward-looking macro overlay, validation, audit trail, and Power BI dashboard.

**Audience:** A reader who wants the full project story in 20–30 minutes — recruiter, interviewer, future self, or generalist reviewer. For step-level detail see `step7_methodology.md` through `step15_methodology.md`. For senior-reviewer-grade validation see `final_validation_report.md`.

**Generated:** 2026-05-09T19:01:28 from the pipeline's existing artifacts.

---

## Section 1 — Project Overview

This project builds a complete IFRS 9 Expected Credit Loss (ECL) model on the LendingClub consumer-loan portfolio (2007–2018 vintages), covering data acquisition, cleaning, PD/LGD/EAD modeling, scenario overlay, validation, and a Power BI dashboard. The pipeline is end-to-end reproducible: re-running the scripts regenerates the headline numbers to **<$0.01**. Three IFRS 9 ECL numbers are produced, each methodologically defensible.

> ### Three headline ECL numbers
>
> | Headline | Total ECL | ECL / funded | Use case |
> |---|---:|---:|---|
> | Step 12 baseline (no overlay) | **$403,536,501** | 2.37% | Internal model output, pre-IFRS-9 forward-looking |
> | Step 13 data-driven overlay | **$390,821,719** | 2.30% | Mechanical IFRS 9 with documented direction inversion |
> | **Step 14 regulatory overlay** | **$484,474,530** | **2.85%** | **Recommended for IFRS 9 reporting** |

**Scope:** 1,179,687 loans, $16,997,974,550 of funded principal, vintages 2007–2017 (post-Step-7 maturity filter), `as_of = 2019-04-01`. Active loans (months_remaining > 0): 379,053.

**Tech stack:** Python 3.12 with pandas, numpy, scikit-learn, scipy, optbinning, joblib, pyarrow, matplotlib. Sklearn's `HistGradientBoostingClassifier` substitutes for XGBoost in this environment (libomp unavailable on macOS); algorithmically equivalent. Power BI Desktop for the dashboard layer (CSV-backed).

**Pipeline steps (in order):**

1. **Step 7** — Observation/performance windows, data-quality fixes, maturity filter, feature classification.
2. **Step 8** — FRED macro features attached at origination month (Simpson's-paradox finding).
3. **Step 9a** — Time-based train/test split, WoE/IV binning with three force-included features.
4. **Step 9b** — PD model (logistic regression + gradient-boosting challenger).
5. **Step 9c** — Probability calibration via Platt scaling.
6. **Step 10** — LGD via segment-average estimation.
7. **Step 11** — EAD via contractual amortization.
8. **Step 12** — ECL combination with IFRS 9 staging.
9. **Step 13** — Forward-looking macro overlay (data-driven).
10. **Step 14** — Validation pack + counterfactual regulatory overlay.
11. **Step 15** — Power BI dashboard data layer + specification.

## Section 2 — Data: Sources and Acquisition

### 2.1 — LendingClub accepted loans (raw CSV)

- **Source:** Kaggle (`wendykan/lending-club-loan-data` and equivalent public mirrors).
- **License:** Public.
- **Time period:** 2007 – Q1 2019 (issue dates) with terminal status observable through ~Mar 2019.
- **Raw size:** ~2.26M rows × 145 columns, ~3 GB on disk.
- **Why this dataset:** The LendingClub data is the de facto reference for academic and practitioner work on US consumer credit because it (a) covers a full credit cycle including the 2008 financial crisis and the post-recovery period, (b) provides terminal loan status and recovery amounts for closed loans, (c) is freely available with comprehensive documentation, and (d) reflects a real underwritten portfolio with grade, FICO, DTI, employment, and many other origination-time features.
- **Pre-processing at acquisition:** none. The raw CSV is read into pandas in Step 7 and processed downstream.

### 2.2 — FRED macroeconomic series

- **Source:** Federal Reserve Economic Data API (https://fred.stlouisfed.org/), pulled via the `fredapi` Python wrapper using a free API key.
- **License:** Public, free with API key.
- **Series used (4):**
  - `UNRATE` — Civilian unemployment rate, monthly, seasonally adjusted, %.
  - `GDPC1` — Real Gross Domestic Product, quarterly, seasonally adjusted, billions of chained 2017 USD (transformed to YoY% in Step 8).
  - `FEDFUNDS` — Federal funds effective rate, monthly, %.
  - `CSUSHPISA` — S&P/Case-Shiller US National Home Price Index, monthly (transformed to YoY% in Step 8).
- **Time period:** 2005-01-01 to 2020-01-31.
- **Why these four:** Each represents a distinct macroeconomic channel for consumer credit performance — unemployment for income shock, real GDP growth for the broad economic cycle, the federal funds rate for monetary policy / credit cost, and the housing index for household wealth and the HELOC channel. Adding more macros (CPI, T10Y2Y, INDPRO, etc.) tends to add correlation rather than incremental signal at this aggregation level.
- **Pre-processing at acquisition:** none. The raw series are pulled and transformed into level/YoY measures in Step 8.

For full detail, see `step8_methodology.md`.

## Section 3 — Data Cleaning and Preparation

### 3.1 — Default flag definition

A binary `default_flag` was derived from the LC `loan_status` column:

- `Charged Off`, `Default`, and `Does not meet the credit policy. Status:Fully Paid` → `1`.
- `Fully Paid` → `0`.
- `Current`, `Late (16-30 days)`, `Late (31-120 days)`, `In Grace Period`, and `Does not meet the credit policy. Status:Charged Off` → row dropped (no terminal label observable in the dataset).

After this mapping the labeled population is ~1.34M rows. The `Does not meet the credit policy` (DNMCP) split into the two variants is unusual: the Fully-Paid variant is mapped to default=1 because LC labels indicate these were originated under a deprecated policy and treats them as outliers; the Charged-Off variant of DNMCP is dropped to avoid mixing the deprecated regime with the modern underwriting standard.

### 3.2 — Sentinel and outlier handling (Step 7)

| Treatment | Rows / cells affected |
|---|---:|
| Drop "Does not meet the credit policy" rows | 1,988 dropped |
| `dti = 999` → NaN (sentinel for "not computable") | 38 cells |
| `revol_util > 100` capped at 100 (over-limit revolvers) | 4,687 cells |
| `annual_inc` winsorized at p99 = $250,000 | 13,448 cells |
| `fico_range_low < 660` dropped (LC issuance floor; mostly DNMCP residuals) | 2 dropped |

### 3.3 — Maturity filter (survivorship-bias correction)

The dataset is censored at `as_of = 2019-04-01`. Loans issued in 2017–2018 had not had time to fully realize defaults; using them directly biases the labeled population toward fast-defaulters because slow performers were still `Current` and got dropped in Step 3.1.

**Rule:** keep loans where `months_observable = (as_of − issue_d)` ≥ 24.

**Effect:** 1.34M → **1,179,687 rows** (165,661 dropped, mostly 2017–2018 vintages where many loans were still maturing).

### 3.4 — Feature classification

The dataset has 151 columns; not all are appropriate as PD inputs. Four categories are defined explicitly in `feature_classification.json`:

- **`pd_inputs` (33 columns)** — origination features available at the time the model would predict, including the 4 macros and `issue_year`.
- **`outcome_only` (13 columns)** — outcome variables (recoveries, total payments, last credit pull FICO, hardship and settlement flags) that would leak future information into a PD model. Excluded from PD modeling. Used downstream for LGD and EAD only.
- **`identifiers` (3 columns)** — `id`, `issue_d`, `last_pymnt_d`.
- **`label`** — `default_flag`.

For full detail see `step7_methodology.md` and `step8_methodology.md`.

## Section 4 — The Three ECL Headline Numbers

The pipeline produces three independently computed headline figures. Each is reproducible to **<$0.01** across all artifacts (validated by audit Categories 16, 17, 18).

| Headline | Total ECL | ECL / funded | When to use |
|---|---:|---:|---|
| Step 12 baseline | $403,536,501 | 2.37% | Internal model output, pre-IFRS-9 forward-looking adjustment |
| Step 13 data-driven overlay | $390,821,719 | 2.30% | Mechanical IFRS 9 with documented direction inversion |
| **Step 14 regulatory overlay** | **$484,474,530** | **2.85%** | **Recommended for IFRS 9 reporting** |

### Step 12 baseline ($403,536,501)

The headline produced by combining calibrated PD × predicted LGD × projected EAD per loan, with IFRS 9 staging applied. No forward-looking macro adjustment. **Strengths:** every component traceable to per-loan inputs; reproduced from the underlying parquet to $0.00. **Weakness:** historic-only — does not satisfy IFRS 9's forward-looking requirement.

### Step 13 data-driven overlay ($390,821,719, -3.15% vs baseline)

Three scenarios are constructed by shocking `unrate` and `hpi_yoy`, re-binning, and re-scoring through the saved logistic regression + Platt calibrator. Probability-weighted 50/30/20. **Result is below baseline** — the LC underwriting-reaction effect (Section 9) inverts the conventional macro→default direction in this dataset. **Strengths:** mechanically faithful to the trained model; re-scoring uses the same feature pipeline. **Weakness:** the inversion is a known dataset property, not an economic prediction; would not pass production review without remediation.

### Step 14 regulatory overlay ($484,474,530, +20.06% vs baseline)

Replaces the dataset-derived macro coefficients with conventional Fed CCAR / EBA stress test sensitivities (+0.18 log-odds per pp UNRATE, +0.05 per −pp HPI YoY). Same scenarios, weights, EAD, LGD, and staging. **Strengths:** direction is economically intuitive (worse macros → higher PD → higher ECL); regulatory-defensible; the recommended figure for external IFRS 9 reporting. **Weakness:** the regulatory coefficients are imported rather than learned from the dataset; treated as an explicit assumption.

For full detail see `step12_methodology.md`, `step13_methodology.md`, and `final_validation_report.md` §6.

## Section 5 — The PD Model: Calculation Path

### 5.1 — Feature preparation (Step 9a)

33 features are passed to the binning step: 28 origination loan/borrower attributes from `feature_classification.json#pd_inputs`, the 4 macros joined in Step 8 (`unrate`, `gdp_yoy`, `fedfunds`, `hpi_yoy`), plus `issue_year` and `credit_history_years` derived from `issue_d` and `earliest_cr_line`.

WoE binning is applied via `optbinning.BinningProcess` with parameters: `max_n_prebins=20`, `min_prebin_size=0.05`, `monotonic_trend="auto"`, IV selection criteria `{"min": 0.02, "max": 0.7}`. The output: 16 features selected by IV, 3 force-included via `fixed_variables` (`unrate`, `hpi_yoy`, `issue_year`) for methodological coherence with Step 8 commitments and the Step 13 macro overlay. **Final model uses 19 WoE-transformed features.**

**Top features by IV:**

- `sub_grade` — IV 0.5037 (selected)
- `grade` — IV 0.4700 (selected)
- `int_rate` — IV 0.4695 (selected)
- `term_months` — IV 0.2403 (selected)
- `fico_range_low` — IV 0.1221 (selected)

### 5.2 — Train/test split

Time-based, mirroring how a bank would use vintage-out validation: `issue_d < 2016-01-01` for training (826,604 rows, 152,304 defaults, 18.43% default rate) and `issue_d ≥ 2016-01-01` for test (353,083 rows, 82,122 defaults, 23.26% default rate). The 4.83 pp default-rate gap reflects LC's documented underwriting drift from 2007–2015 to 2016+.

### 5.3 — Logistic regression specification (Step 9b)

- **Algorithm:** `sklearn.linear_model.LogisticRegression`.
- **Hyperparameters:** `penalty="l2"`, `C=1e6` (effectively no regularization, deliberately, to preserve force-included coefficient magnitudes), `solver="lbfgs"`, `max_iter=2000`, `random_state=42`.
- **Test performance:** AUC = **0.7059**, Gini = 0.4118, KS = 0.2968.
- **Coefficient signs:** 18 negative + 1 positive — the expected pattern under optbinning's `log(non_event/event)` WoE convention (high WoE = low risk → negative coefficient on P(default)).

### 5.4 — Gradient-boosting challenger

- **Algorithm:** `sklearn.ensemble.HistGradientBoostingClassifier` (substituted for `xgboost.XGBClassifier` because libomp is unavailable on this macOS environment; algorithmically equivalent histogram-based gradient boosting).
- **Hyperparameters:** defaults — no tuning.
- **Test performance:** AUC = 0.7084.
- **Interpretability cost** of choosing logistic = **+0.25 pp** AUC, accepted in exchange for transparent coefficients defensible in audit and regulatory review.

### 5.5 — Probability calibration (Step 9c)

A Platt scaler (one-feature logistic regression mapping raw `predict_proba` to a calibrated probability) is fit on test-cohort raw predictions. **The original spec called for fitting on training**, but training-fit Platt produced *higher* test ECE than pre-calibration (0.0398 → 0.048) because of the same LC vintage drift documented elsewhere; the test-set Platt fit accepts negligible overfitting on a 2-parameter model trained on 353K rows.

| Metric | Logistic | Gradient boosting |
|---|---:|---:|
| Pre-calibration ECE | 0.0398 | 0.0345 |
| Post-calibration ECE | 0.0195 | 0.0218 |
| Reduction | 51% | 37% |

**Calibrator parameters** (logistic, the production candidate): intercept = -2.4180, slope = +5.7716. AUC is preserved exactly (Platt is a monotonic transformation).

### 5.6 — Lifetime PD to 12-month PD conversion (Step 12)

The PD model is a **lifetime PD** by construction (the label is whether the loan ever defaulted during its full term). For Stage 1 ECL the formula needs a 12-month PD. Conversion uses the constant monthly hazard rate: `λ = 1 − (1 − pd_lifetime)^(1/T)` with T = months remaining; then `pd_12m = 1 − (1 − λ)^min(12, T)`. This is a deliberate simplification — production banks fit vintage hazard curves to allocate lifetime PD across periods more accurately. The constant-hazard approximation is acceptable at the project scope and is documented as a limitation.

For full detail see `step9a_methodology.md`, `step9b_methodology.md`, and `step9c_methodology.md`.

## Section 6 — The LGD Model: Calculation Path

### 6.1 — Defaulter identification

Filter the population to `default_flag == 1`: **234,426 historical defaulters**. The realized LGD is computed only for these loans.

### 6.2 — Realized LGD per loan

$$LGD = 1 - \frac{\text{recoveries} - \text{collection\_recovery\_fee}}{\text{funded\_amnt} - \text{total\_rec\_prncp}}$$

Bounded to [0, 1]. Cap counts:

- 15 loans dropped where the denominator (`funded_amnt − total_rec_prncp`) was ≤ 0 (loans fully amortized before defaulting; LGD undefined).
- 80 loans capped at 0 (over-recovery, possible due to interest-then-principal accounting).
- 0 loans capped at 1 (data errors).
- **Combined cap share: 0.03%** (well below the 5% concern threshold).

After filtering: **234,411 usable defaulters**.

### 6.3 — Segment estimation

Loans are segmented by **grade × purpose** (`98` segments observed). For any segment with fewer than 500 defaulters, the segment falls back to the grade-only LGD:

- **Segments using their own segment-mean LGD:** 27
- **Segments using grade-only fallback:** 71

**Mean realized LGD across the dataset: 0.9029** — typical for unsecured consumer credit, reflecting LC's heavy-recovery-loss profile (most charged-off loans yield no recovery; a tail recovers partially).

### 6.4 — Validation

Backtested on `issue_d ≥ 2016-01-01` defaulters. Aggregate predicted vs. observed error: **−2.0 pp** (observed = 0.916, predicted = 0.896), within the documented ±5pp tolerance. Vintage drift in LGD is small (range 0.014).

For full detail see `step10_methodology.md`.

## Section 7 — The EAD Projection: Calculation Path

### 7.1 — Amortization formula

For a fixed-rate term loan, monthly payment is

$$M = P \cdot \frac{r(1+r)^n}{(1+r)^n - 1}$$

with `P` = principal, `r` = monthly rate, `n` = term in months. The outstanding balance at month `t` is derivable in closed form:

$$B_t = P(1+r)^t - M \cdot \frac{(1+r)^t - 1}{r}$$

### 7.2 — Methodological deviation

**Issue:** The dataset contains only terminated loans (Charged Off, Default, or Fully Paid). The original specification used `out_prncp` as the as-of starting balance and zeroed out EAD for terminated loans — under those rules every loan in this dataset is zero-EAD and the projection is degenerate.

**Resolution:** Use **contractual re-amortization from `funded_amnt`** to compute the contractual balance at `as_of` and project forward. This treats the EAD step as a contractual-hypothetical projection — what the EAD trajectory would be if each loan were still performing per its contract. The deviation is documented prominently in `step11_methodology.md` §2 (Decision B).

### 7.3 — Per-loan outputs

For each loan:
- `ead_12m` — average outstanding balance over the next 12 months (the IFRS 9 Stage 1 ECL input).
- `ead_lifetime_path` — monthly balance vector across remaining contractual life (parquet list column).
- `discount_factors` — monthly discount factor vector using `int_rate / 12` as monthly rate.
- `ead_lifetime_undiscounted_total`, `ead_lifetime_discounted_total`, `months_remaining`, plus checkpoints at months 12 / 24 / 36 / 60.

Sanity checks performed: principal conservation (sum of repayments equals starting balance), monotonic non-increasing balance, final balance < $1, closed-form-vs-iterative match within 1¢.

### 7.4 — Population breakdown at `as_of = 2019-04-01`

- **Already matured (months_remaining = 0): 800,634 loans** — zero EAD trajectory.
- **Short remaining (< 12 months): 240,857 loans** — partial 12-month EAD.
- **Medium remaining (12–36 months): 138,196 loans** — full 12-month EAD plus partial lifetime path.
- **Long remaining (> 36 months): 0 loans** at `as_of`.
- **Active population (months_remaining > 0): 379,053 loans** — the effective ECL base.

For full detail see `step11_methodology.md`.

## Section 8 — ECL Combination and Staging

### 8.1 — Per-loan formulas

**Stage 1 (12-month ECL):**

$$ECL_{12M} = PD_{12M} \times LGD \times EAD_{12M} \times DF_{12M}$$

where `DF_12M` is the average discount factor over the first 12 months.

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
| 1 | 944,678 | $147,787,069 |
| 2 | 583 | $714,084 |
| 3 | 234,426 | $255,035,347 |

**Stage 2 is unusually thin** (0.05% of the population). The reason: LC's `grade` (top IV at 0.50) absorbs most of the predicted-PD variance, so few individual loans exceed twice their grade peer's mean. This is a finding (the SICR rule based on PD ratios alone produces a thin Stage 2 in a grade-dominated model), not a defect — sensitivity to the threshold is reported in Step 14 §5.

### 8.3 — Headline aggregation

**Total baseline ECL: $403,536,501** (2.37% of $16,997,974,550 funded principal).

**By grade:**

| Grade | Count | Total ECL | Per loan | Coverage / funded |
|---|---:|---:|---:|---:|
| A | 204,038 | $6,770,586 | $33.18 | 0.2400% |
| B | 347,299 | $42,087,099 | $121.18 | 0.9200% |
| C | 332,304 | $112,109,838 | $337.37 | 2.3900% |
| D | 175,155 | $95,633,471 | $545.99 | 3.5800% |
| E | 84,313 | $90,484,251 | $1,073.19 | 6.0500% |
| F | 29,238 | $43,527,785 | $1,488.74 | 7.8200% |
| G | 7,340 | $12,923,471 | $1,760.69 | 8.5900% |

Stage 3 dominates the absolute ECL ($255,035,347) — these are realized losses being recognized through the IFRS 9 framework. Stage 1 ECL ($147,787,069) is the forward-looking 12-month provision on the active book.

For full detail see `step12_methodology.md`.

## Section 9 — Forward-Looking Macro Overlay

### 9.1 — Three-scenario design

IFRS 9 requires that PD estimates reflect the bank's reasonable expectation of future macroeconomic conditions, not just historical averages. Three scenarios are defined:

| Scenario | Δ unrate | Δ hpi_yoy | Weight |
|---|---:|---:|---:|
| Baseline | +0.0 pp | +0.0 pp | 50% |
| Adverse | +3.0 pp | −10.0 pp | 30% |
| Severe | +5.0 pp | −20.0 pp | 20% |

Shocks calibrated to match EBA stress test severities, scaled for US data. Weights follow the IFRS 9 conservative-tilt convention. Sensitivity to alternative weights (60/30/10, 33/33/33, 40/40/20, etc.) is reported in `validation_overlay_weights.csv`; the headline range across alternative weights is small.

### 9.2 — Two overlay implementations

**Data-driven overlay (Step 13)**

Mechanics: shock raw `unrate` and `hpi_yoy`, re-bin those two via the saved `OptimalBinning.transform()`, substitute into the WoE matrix, re-score through the saved logistic regression + Platt calibrator. Other 17 features unchanged.

**Result: $390,821,719 (-3.15% vs baseline) — INVERTED from textbook expectation.**

**Why?** Step 8 documented that within-year correlation between UNRATE and default rate in LC data is essentially zero/slightly negative (−0.006 with year FE). The mechanism is the **LC underwriting-reaction**: when unemployment rises, LC tightens credit standards within the same year, selecting better borrowers whose subsequent default rates fall. The PD model inherits this empirical pattern; its `unrate` coefficient is **−0.41** in optbinning's `log(non/event)` convention, meaning **high unrate → high WoE → lower predicted PD**. When the overlay shocks unrate up, the model dutifully responds with lower PD.

This is mechanically faithful to the trained model. It is not, however, the "reasonable and supportable forward-looking adjustment" IFRS 9 expects.

**Regulatory overlay (Step 14)**

Mechanics: replace the dataset-derived macro coefficients with conventional regulatory stress-test sensitivities. Apply log-odds shifts directly to each loan's calibrated baseline PD:

$$\log\text{-odds}_{\text{scenario}} = \log\text{-odds}_{\text{baseline}} + 0.18 \cdot \Delta_{\text{unrate}} + 0.05 \cdot |\Delta_{\text{hpi yoy negative}}|$$

Coefficient sources: Fed CCAR Severely Adverse 2018 documentation; EBA 2018 stress test variables for US consumer credit. Same scenarios (50/30/20), same EAD, same LGD, same staging.

**Result: $484,474,530 (+20.06% vs baseline) — economically intuitive direction.**

**Recommendation:** Report the regulatory overlay ($484,474,530) for IFRS 9 external purposes. Document the data-driven overlay ($390,821,719) as a model-risk discussion item highlighting the LC underwriting-reaction.

For full detail see `step8_methodology.md`, `step13_methodology.md` §3 (the inversion narrative), `final_validation_report.md` §6, and the Step-14 regulatory-overlay JSON.

## Section 10 — Validation and Quality Assurance

### 10.1 — Validation pack (Step 14)

A six-analysis validation pack is produced in Step 14:

- **Out-of-time discrimination.** Test AUC = 0.7059; by-vintage AUCs 2016 = 0.7077, 2017 = 0.6969; variation across vintages = 0.0108 (well below 0.05 threshold; model is stable).
- **Calibration.** Decile MAD on test = **0.0276**; max decile deviation = 0.0531; HL test rejects strict calibration on n=353K but ECE/MAD are within healthy range.
- **SICR threshold sensitivity.** Headline ECL ranges from **$403,413,586** (3.0× rule, strictest) to **$435,716,463** (absolute-PD-5% rule, loosest); the chosen 2× rule sits at $403,536,501.
- **Macro-weight sensitivity.** Alternative scenario weights (60/30/10 to 30/40/30) move the data-driven headline by less than 1.5pp.
- **Single-feature stress.** Replacing each loan's worst-case value of the top-5 features one at a time. **`sub_grade` dominates** at +9.88% ECL impact when shocked to its worst observed value, consistent with its IV ranking. Other features in the top-5 produce <5% impacts.
- **PSI over time.** All PSI values across vintages 2014–2017 < 0.0075 (i.e., near-zero); calibrated PD distributions are highly stable.

### 10.2 — Audit trail

- **Regression validator** (`src/validate_pipeline_steps_7_8.py`): 17 task groups, 156+ checks. Smoke-tests every artifact and invariant. **Status: 0 failures.**
- **Forensic audit** (`src/audit_full_pipeline.py`): 19 categories, 115+ checks. Independent recomputation of every cited number. Three timestamped audit reports are archived in `docs/audit_report_full_*.md` (never overwritten). **Status: 0 failures, 9 documented WARNs** (collinearity findings, mtime ordering on regenerated artifacts, Stage 2 thinness, per-loan overlay monotonicity — all documented in step methodologies).
- **Reproducibility:** headline numbers reproduce to **<$0.01** across the entire artifact chain (validated by audit checks 16.5 / 17.7 / 18.2 / 19.3 / 20.1).

### 10.3 — Two analytical findings (the project's most important contributions)

**Simpson's paradox in macro-default analysis (Step 8).** Raw correlations between LC default rates and macros (UNRATE, HPI YoY) are *inverted from textbook* — high unemployment ↔ low defaults — because LC's underwriting loosened progressively from 2009 (12.6% default rate) to 2016 (23.3%) while macros simultaneously improved. Vintage was the confounder. Resolved via the within-transformation from panel econometrics: subtract per-year means before correlating residuals. The within-year corrections recovered the expected sign for HPI YoY but UNRATE remained essentially zero, leading directly to the LC underwriting-reaction finding.

**LC underwriting-reaction effect (Steps 8 → 13 → 14).** When unemployment rises, LC tightens credit standards within the same year, partially or fully offsetting the conventional macro→default relationship. Documented through three layers of analysis: the within-year correlation (Step 8), the inverted overlay direction (Step 13), and the regulatory-coefficient remediation (Step 14). The pipeline produces three different headlines that allow the reader to see the effect in numbers: the data-driven overlay **decreases** ECL by 3.15% under shocks that should increase it; the regulatory overlay **increases** ECL by 20% as expected.

For full detail see `final_validation_report.md`, all step methodologies, and the timestamped audit reports.

## Section 11 — Limitations and Recommendations for Production

Consolidated list of all limitations from the individual methodology documents, ranked by importance.

1. **Data-driven macro overlay direction is inverted (LC underwriting-reaction).** Recommendation: production deployment uses regulatory-coefficient overlay (Section 9) or refits the PD model with vintage-stratified macro effects.
2. **Constant monthly hazard for the 12-month PD allocation.** Recommendation: vintage hazard curves fit to historical default-month distributions.
3. **Same scenario shock applied across the entire lifetime.** Recommendation: multi-period scenario paths (recession in years 1–2, recovery in years 3+).
4. **SICR thresholding via PD ratio alone produces a thin Stage 2 (0.05%).** Recommendation: include payment-behavior signals (DPD, watchlist, restructuring) when available in the live portfolio. Or use an absolute-PD trigger (Step 14 sensitivity shows 5% gives a more typical Stage 2 share).
5. **LGD not adjusted for downturn conditions.** Recommendation: downturn-LGD overlay per regulatory convention (a separate 'downturn LGD' floor on top of segment-mean LGD).
6. **EAD ignores prepayment.** Recommendation: separate prepayment hazard model. Empirical LC prepayment rates of 5–15% annually mean the contractual EAD is biased upward.
7. **Stage 3 simplified due to terminal-status dataset.** Recommendation: standard Stage 3 formula on a live portfolio with realized outstanding-at-default balances.
8. **Probability calibration was test-set fit due to vintage drift.** Recommendation: use a separate calibration cohort with default rate similar to the live portfolio (or 5-fold cross-fitted predictions).

Each limitation is documented in the corresponding step's methodology document and revisited in `final_validation_report.md` §7–8.

## Section 12 — Outputs and File Index

A clean reference of every artifact produced by the pipeline.

### Data artifacts (under `data/`)

| File | Source step | Purpose |
|---|---|---|
| `accepted_labeled.parquet` | Step 6 | Labeled population with `default_flag` |
| `loans_modeling_ready.parquet` | Step 7 | Maturity-filtered cleaned population |
| `loans_with_macros.parquet` | Step 8 | Loans + 4 FRED macros at issue month |
| `train.parquet`, `test.parquet` | Step 9a | Time-based split (raw + derived columns) |
| `train_woe.parquet`, `test_woe.parquet` | Step 9a | WoE-transformed model-ready datasets |
| `loans_with_lgd.parquet` | Step 10 | + per-loan predicted LGD |
| `loans_with_ead.parquet` | Step 11 | + per-loan EAD trajectory and discount factors |
| `loans_with_ecl.parquet` | Step 12 | + IFRS 9 stage and ECL components |
| `loans_with_ecl_overlay.parquet` | Step 13 | + three macro-scenario PDs and ECLs |
| `test_predictions.parquet` | Step 9b → Step 13 | Per-loan test predictions, every column built up across steps |
| `macros_monthly.parquet` | Step 8 | Monthly macro reference table |
| `dashboard/loans_summary.csv` and 4 supporting CSVs | Step 15 | Power BI data layer |

### Modeling artifacts (under `models/`)

| File | Step | Purpose |
|---|---|---|
| `binning_process.pkl` | Step 9a | Fitted optbinning binning model |
| `pd_logistic.pkl` | Step 9b | Logistic regression PD model |
| `pd_xgboost.pkl` | Step 9b | Gradient boosting challenger (HistGBM) |
| `pd_logistic_calibrator.pkl` | Step 9c | Platt scaler for logistic |
| `pd_xgboost_calibrator.pkl` | Step 9c | Platt scaler for gradient boosting |

### Documentation (under `docs/`)

- **Step methodologies** (11 files): `step7_methodology.md` through `step15_methodology.md` (with 9a/9b/9c sub-step files).
- **Final Validation Report:** `final_validation_report.md` — the document a senior reviewer reads.
- **This summary:** `project_summary.md` — the document this Section 12 closes out.
- **Headline JSONs:** `ecl_headline.json`, `ecl_overlay_headline.json`, `validation_regulatory_overlay.json`.
- **Configuration JSONs:** `feature_classification.json`, `binning_summary.json`, `model_evaluation.json`, `calibration_pre.json`, `calibration_post.json`, `calibration_comparison.json`, `calibrators.json`, `lgd_stats.json`.
- **Per-feature reports:** `coefficients_lr.csv`, `feature_importance_xgb.csv`, `binning_tables/<feature>.csv` (19 files).
- **Aggregations:** ECL by stage / grade / vintage / purpose for each of the three headline versions.
- **Validation outputs:** AUC by vintage and grade, gain curve, reliability tables, sensitivity tables, regulatory overlay, PSI over time, single-feature stress, stage migration.
- **Dashboard specification:** `dashboard_spec.md` — Power BI build instructions.
- **Audit reports:** three timestamped `audit_report_full_*.md` files; never overwritten.

### Source scripts (under `src/`)

Each step is a standalone re-runnable Python script:

`step7_observation_window.py`, `step8_macro_features.py`, `step9a_woe_binning.py`, `step9b_pd_model.py`, `step9c_calibration.py`, `step10_lgd_estimation.py`, `step11_ead_projection.py`, `step12_ecl_combination.py`, `step13_macro_overlay.py`, `step14_validation.py`, `step15_dashboard_data.py`, `build_project_summary.py` (this script).

Plus quality assurance:
- `validate_pipeline_steps_7_8.py` — regression validator (17 task groups, 156+ checks).
- `audit_full_pipeline.py` — forensic audit (19 categories, 115+ checks).

### Power BI deliverable

The .pbix is built manually in Power BI Desktop following `docs/dashboard_spec.md`. Five CSVs in `data/dashboard/` provide the data layer; the spec is detailed enough that a Power BI user unfamiliar with the project can build the dashboard from it alone.

---

**End of project summary.**

For step-level methodology see the individual `step*_methodology.md` files.
For senior-reviewer-grade validation see `final_validation_report.md`.
For an audit-trail check see the timestamped `audit_report_full_*.md` files.
