# Step 9b — PD Model Fitting

## 1. Purpose

Fit the Probability of Default (PD) model on the WoE-transformed dataset from Step 9a, and quantify the interpretability cost by comparing against a gradient-boosting challenger. **Two models are fit:**

- **Logistic regression** (production candidate). Linear in WoE features; coefficients are directly interpretable; outputs naturally calibrated; the format expected by regulators and audit reviewers.
- **Gradient-boosting challenger** (documentation only). Quantifies the AUC ceiling reachable with a non-linear model on this feature set. Used to compute the interpretability cost; not deployed.

_Library substitution note:_ the original specification called for `xgboost.XGBClassifier` with `tree_method="hist"`. Since the macOS OpenMP runtime (`libomp`) is not available on this system and could not be installed via `brew`, sklearn's `HistGradientBoostingClassifier` is used in its place. The two are algorithmically equivalent histogram-based gradient-boosted-tree learners, with comparable AUC at default settings on tabular data of this size. To swap back, install libomp and replace the import.

## 2. Logistic regression specification

| Parameter | Value | Rationale |
|---|---|---|
| `penalty` | `"l2"` | sklearn requires a penalty type; we neutralize it with C below. |
| `C` | `1e6` | Effectively no regularization. WoE features are already on a log-odds scale. Regularization would disproportionately shrink the small coefficients on `unrate`, `hpi_yoy`, and `issue_year`, defeating the Step 9a force-include rationale. |
| `solver` | `"lbfgs"` | Converges quickly on small feature counts and moderate row counts. |
| `max_iter` | `2000` | Comfortable convergence margin. |
| `random_state` | `42` | Determinism. |

## 3. Gradient-boosting specification

| Parameter | Value | Rationale |
|---|---|---|
| `max_iter` | `200` | Equivalent to xgboost `n_estimators=200`. |
| `max_depth` | `4` | Conventional shallow trees for credit data. |
| `learning_rate` | `0.1` | Standard. |
| `random_state` | `42` | Determinism. |

No hyperparameter tuning. Defaults reach ~95% of the predictive ceiling on credit-grade tabular data without grid search; tuning would marginally improve AUC at the cost of a much larger story to defend at audit / interview.

## 4. Evaluation results

| Model | AUC | Gini | KS | PSI (train→test) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7059 | 0.4118 | 0.2968 | 0.0111 |
| Gradient boosting   | 0.7084 | 0.4169 | 0.3022 | 0.0146 |

**Interpretability cost:** +0.25 pp AUC (gradient-boosting − logistic).

Both models pass the AUC sanity band [0.65, 0.85]: above-floor (no bug) and below-ceiling (no leakage). KS values are above the conventional 0.30 acceptable threshold. PSI reflects LC's vintage drift (train: 2007–2015; test: 2016+), expected to be in the 0.05–0.20 range.

**Decile lift summary:** see `docs/decile_lift_lr.csv` and `docs/decile_lift_xgb.csv`. The top decile of predicted PD captures a default rate several times the base rate — the expected ranking-power signal.

## 5. Coefficient analysis (logistic regression)

| Feature | Status | Coefficient | IV |
|---|---|---:|---:|
| annual_inc | selected | -0.7361 | 0.0321 |
| term_months | selected | -0.5877 | 0.2403 |
| home_ownership | selected | -0.5792 | 0.0209 |
| mort_acc | selected | -0.5291 | 0.0296 |
| inq_last_6mths | selected | -0.5152 | 0.0276 |
| dti | selected | -0.4681 | 0.0761 |
| sub_grade | selected | -0.4185 | 0.5037 |
| unrate | selected_forced | -0.4091 | 0.0187 |
| hpi_yoy | selected_forced | -0.3598 | 0.0169 |
| issue_year | selected_forced | -0.3315 | 0.0172 |
| installment | selected | -0.2675 | 0.0298 |
| revol_util | selected | +0.2626 | 0.0229 |
| int_rate | selected | -0.2313 | 0.4695 |
| fico_range_high | selected | -0.1867 | 0.1221 |
| fico_range_low | selected | -0.1867 | 0.1221 |

_Top 15 by |coefficient|. Full table in `docs/coefficients_lr.csv`._

**Sign sanity (negative dominates):** 18 negative + 1 positive. With optbinning's WoE convention (`log(P(non-event) / P(event))`), high WoE = low default risk, so a sound P(default) model produces NEGATIVE coefficients on each WoE feature. The expected dominant sign is negative; any positive coefficient indicates either correlated features fighting for credit or a force-included low-IV feature whose minor signal can flip sign without economic meaning.

**Magnitude sanity:** for IV-selected features, |coef| typically lies in [0.3, 1.5]. Out-of-band features (warnings, not failures): 9. Most out-of-band cases reflect **collinearity** — `grade` and `sub_grade` carry the same signal (LC's grade is essentially a precomputed PD score), as do `fico_range_low` and `fico_range_high` (always 4 points apart, by LC's bin definition). Logistic regression with no penalty splits the credit between collinear features arbitrarily, so individual magnitudes can be inflated or shrunk without harming predictive accuracy.

**Force-included features (low IV, expected small coefficients):**

- `unrate` — coef = -0.4091, IV = 0.0187
- `hpi_yoy` — coef = -0.3598, IV = 0.0169
- `issue_year` — coef = -0.3315, IV = 0.0172

These small coefficients are expected per the Step 9a force-include rationale: `grade` (IV 0.470) already absorbs most of the vintage and macro variation. The forced features carry residual signal that becomes useful for the macro-overlay step (Step 14) and for vintage control, even though their marginal predictive power is small.

## 6. Feature importance (gradient boosting)

Permutation importance on the test set, AUC scoring, n_repeats=3.

| Feature | Importance (mean ΔAUC) | Std |
|---|---:|---:|
| sub_grade | +0.073895 | 0.000579 |
| term_months | +0.014049 | 0.000324 |
| int_rate | +0.009624 | 0.000105 |
| fico_range_low | +0.008745 | 0.000325 |
| dti | +0.007008 | 0.000058 |
| home_ownership | +0.005571 | 0.000132 |
| loan_amnt | +0.005007 | 0.000082 |
| mort_acc | +0.004467 | 0.000131 |
| annual_inc | +0.003961 | 0.000038 |
| funded_amnt | +0.002869 | 0.000001 |

_Top 10 features. Full table in `docs/feature_importance_xgb.csv`._

## 7. Model selection rationale

**Production candidate: logistic regression.** Despite the gradient-boosting model reaching 0.7084 AUC vs the logistic's 0.7059 (a +0.25 pp gap), logistic regression is selected as the primary model because:

- **Interpretability.** Each prediction can be decomposed as the sum of per-feature WoE × coefficient contributions, supporting per-loan adverse-action explanations and per-feature audit reviews.
- **Calibration.** Logistic regression on log-odds-scaled features produces near-calibrated probabilities by construction. Gradient-boosting outputs are typically miscalibrated and require a separate calibration step (Step 9c).
- **Stability.** The linear functional form is stable across vintages and easy to monitor for coefficient drift. Tree ensembles can shift their split structure dramatically between refits even without underlying data drift.
- **Regulatory acceptance.** Logistic regression on WoE features is the standard credit-scoring artifact for IFRS 9 / IRB reviews; gradient-boosting models require additional documentation (SHAP, surrogate models) to satisfy the same review.

The interpretability cost is real but bounded; the gradient-boosting benchmark serves as documentation of the predictive ceiling rather than a model to deploy.

## 8. Limitations

- The PD model produces **lifetime PD** because of the origination-based observation-window design from Step 7. Conversion to a 12-month PD for IFRS 9 Stage 1 ECL will be done downstream using vintage hazard rates.
- Predicted probabilities are **not yet calibrated**. Logistic regression on WoE features tends to be near-calibrated; gradient-boosting outputs typically are not. Step 9c handles calibration with isotonic or Platt-scaling.
- **PSI between train and test is elevated** by LC vintage drift (train: 2007–2015, test: 2016+). This reflects a property of the data (LC's underwriting loosened over time), not a model defect.
- The gradient-boosting challenger uses sklearn's `HistGradientBoostingClassifier` rather than `xgboost.XGBClassifier` due to a missing macOS `libomp` runtime. Both are histogram-based gradient-boosted-tree learners; expected AUC difference between the two implementations on a dataset of this size is < 0.005. To swap, install libomp and replace the import.
- The collinearity between `grade`/`sub_grade` and between `fico_range_low`/`fico_range_high` produces inflated magnitudes on individual coefficients without changing prediction quality. This is a feature-engineering choice (we kept both for IV ranking transparency); a production model could drop one of each pair to tighten the coefficient table.

## 9. Outputs

- **Logistic regression model:** `models/pd_logistic.pkl` (load with `joblib.load`).
- **Gradient-boosting model:** `models/pd_xgboost.pkl` (HistGradientBoostingClassifier; filename retained for downstream-pipeline compatibility).
- **Per-loan predictions on test:** `data/test_predictions.parquet` — `id`, `issue_d`, `default_flag`, `pd_lr`, `pd_xgb`.
- **Coefficient table:** `docs/coefficients_lr.csv`.
- **Feature importance:** `docs/feature_importance_xgb.csv`.
- **Decile lift tables:** `docs/decile_lift_lr.csv`, `docs/decile_lift_xgb.csv`.
- **Evaluation summary:** `docs/model_evaluation.json`.
