# Step 9a — Train/Test Split + WoE/IV Binning

## 1. Purpose

Weight of Evidence (WoE) binning is the standard preparation step for credit-risk logistic regression. It (a) linearizes non-linear effects in features like FICO, DTI, and income — turning them into per-unit log-odds shifts that logistic regression can consume natively; (b) handles missing values cleanly by giving them their own bin and WoE value, with no need for imputation; (c) provides Information Value (IV) as a single predictive-power score per feature for selection; and (d) is the format banks, regulators, and audit reviewers expect to see in any credit-scoring artifact.

## 2. Train/test split

**Cutoff:** `issue_d < 2016-01-01` for train; `issue_d ≥ 2016-01-01` for test.

- **Train:** 826,604 rows, 152,304 defaults, default rate 18.43%.
- **Test:** 353,083 rows, 82,122 defaults, default rate 23.26%.
- **Default-rate diff (test − train):** +4.83pp.

The test default rate exceeds the train default rate. This reflects LendingClub's vintage drift — defaults were ~12–17% in 2009–2013 and rose to ~20–23% by 2015–2017. The train period (2007–2015) and test period (2016+) therefore have systematically different baseline rates. The PD model in Step 9b will use `issue_year` as a covariate to absorb this drift directly; the WoE step here treats the train period as-is.

## 3. Feature derivation

Two derived features are added to the loan-level data before binning:

- **`issue_year`** = `issue_d.dt.year.astype('int')`. Listed in `pd_inputs` since the Step 8 patch but not previously materialized.
- **`credit_history_years`** = `(issue_d − earliest_cr_line).dt.days / 365.25`. Replaces `earliest_cr_line` in the working feature list (the `feature_classification.json` file is unchanged; the substitution applies in this script's working copy only). The methodology committed to using credit-history length as the model input rather than the raw earliest-credit-line date.

## 4. Binning configuration

Implemented with `optbinning.BinningProcess` (deterministic, no random seed needed). Parameters:

| Parameter | Value | Rationale |
|---|---|---|
| `max_n_prebins` | 20 | At most 20 candidate bins per feature; prevents overfitting to small bins. |
| `min_prebin_size` | 0.05 | Each bin must hold at least 5% of population; ensures statistical reliability. |
| `monotonic_trend` | `"auto"` | optbinning chooses monotonic vs. non-monotonic per feature based on best fit. Most credit features bin monotonically; some (e.g., `purpose`) won't. |
| `selection_criteria` | IV ∈ [0.02, 0.7] | Drop features with IV < 0.02 (no signal). Hard ceiling at 0.7 flags possible leakage; conventional retail-credit ceiling is 0.5, raised slightly here to accommodate `grade`/`sub_grade`, which are LC's pre-computed PD score and legitimately have IV > 0.5. |
| `n_jobs` | 1 | Single-threaded for stability on macOS. |

**Categorical features (7):** `grade`, `sub_grade`, `purpose`, `home_ownership`, `verification_status`, `addr_state`, `application_type`.

**Numerical features (26):** `loan_amnt`, `funded_amnt`, `term_months`, `int_rate`, `installment`, `annual_inc`, `dti`, `fico_range_low`, `fico_range_high`, `emp_length_years`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `pub_rec_bankruptcies`, `mort_acc`, `tax_liens`, `credit_history_years`, `unrate`, `gdp_yoy`, `fedfunds`, `hpi_yoy`, `issue_year`.

`issue_year` is treated as numerical so that low-volume early years can naturally be merged into the lowest bin.

**Force-included features (`fixed_variables`):**

Three features were force-included via `fixed_variables` despite IV below the 0.02 selection threshold:

- `unrate` (IV 0.019) — required for the forward-looking macro overlay in Step 14, which applies macro-stress scenarios to the PD model and requires a macro coefficient to operate on.
- `hpi_yoy` (IV 0.017) — second macro variable for the overlay; showed clean negative sign in the Step 8 within-year correlation check.
- `issue_year` (IV 0.017) — vintage covariate, committed to in Step 8's methodology to disentangle macro effects from LC's underwriting drift.

`gdp_yoy` (IV 0.009) and `fedfunds` (IV 0.007) were not force-included; their IV is materially below the threshold and their economic signs were ambiguous in within-year analysis.

The three forced features have low marginal IV because LC's `grade` (IV 0.470) already absorbs most of the vintage and macro variation — vintage and grade are co-evolved in this dataset. Forcing inclusion gives the PD model an explicit vintage and macro signal even though `grade` carries most of it implicitly.

## 5. Results

**Features input:** 33  
**Selected:** 19 (16 IV-selected + 3 force-included)  
**Force-included:**

- `unrate` (IV 0.019)
- `issue_year` (IV 0.017)
- `hpi_yoy` (IV 0.017)

**Dropped:** 14  
**Fit time:** 10.7s

**Top 15 features by IV:**

| Feature | Type | IV | n_bins | Status |
|---|---|---:|---:|:---:|
| sub_grade | categorical | 0.5037 | 17 | selected |
| grade | categorical | 0.4700 | 5 | selected |
| int_rate | numerical | 0.4695 | 14 | selected |
| term_months | numerical | 0.2403 | 2 | selected |
| fico_range_low | numerical | 0.1221 | 13 | selected |
| fico_range_high | numerical | 0.1221 | 13 | selected |
| dti | numerical | 0.0761 | 14 | selected |
| verification_status | categorical | 0.0513 | 3 | selected |
| funded_amnt | numerical | 0.0373 | 8 | selected |
| loan_amnt | numerical | 0.0373 | 8 | selected |
| annual_inc | numerical | 0.0321 | 12 | selected |
| installment | numerical | 0.0298 | 8 | selected |
| mort_acc | numerical | 0.0296 | 6 | selected |
| inq_last_6mths | numerical | 0.0276 | 4 | selected |
| revol_util | numerical | 0.0229 | 13 | selected |

Full IV table is in `docs/binning_summary.json`. Per-feature bin tables (cut points, counts, default rates, WoE, IV contribution) are in `docs/binning_tables/<feature>.csv`.

## 6. Sanity-check results

- **9.1 Zero-null coverage:** `train_woe` and `test_woe` contain zero nulls. WoE binning includes a "missing" bin per feature, so input nulls become valid WoE values. ✓
- **9.2 Low-coverage bins on test:** _No features had >5% test rows in low-coverage bins._

- **9.3 IV consistency:** **Recomputed-IV vs reported (`sub_grade`):** reported = 0.5037, recomputed = 0.5037, relative difference < 1% ✓
- **9.4 Monotonicity:** every feature declared monotonic by optbinning has WoE values that are actually monotonic across bins. ✓

## 7. Outputs

- **Fitted binning model:** `models/binning_process.pkl` (use `joblib.load` to deserialize).
- **Train/test parquets (raw + derivations):** `data/train.parquet`, `data/test.parquet`.
- **WoE-transformed parquets (model-ready):** `data/train_woe.parquet`, `data/test_woe.parquet`.
- **Machine-readable summary:** `docs/binning_summary.json`.
- **Per-feature bin tables:** `docs/binning_tables/<feature>.csv` + `_summary.csv`.

## 8. Limitations

- WoE binning is fit on training data only; bins for the test set inherit training cuts. New categorical values seen only in test (e.g., a state never in training) get the "unseen" WoE value, with potentially weaker calibration.
- The binning is not refit per fold or vintage; this is acceptable for a first model but a production version would refit binning periodically as the portfolio evolves.
- IV thresholds (0.02 floor, 0.7 ceiling) are conventional. Sensitivity analysis on the lower bound is left to the validation step.
- WoE replacement removes individual-loan variation within a bin: every loan in the same bin gets the same WoE value. This is the intended behavior for logistic regression, but downstream ML models (e.g., XGBoost in Step 9b) may be limited compared to using raw features directly.
- The three force-included features (`unrate`, `hpi_yoy`, `issue_year`) have marginal IV below the conventional 0.02 cutoff. In the logistic regression of Step 9b, their coefficients may be small and have wide confidence intervals. This is acceptable because their role in the project is not primarily predictive — `grade` carries most of the predictive load — but methodological: enabling the macro overlay and providing explicit vintage control. Sensitivity analysis on whether these features change the test-set AUC will be reported in Step 9b.
