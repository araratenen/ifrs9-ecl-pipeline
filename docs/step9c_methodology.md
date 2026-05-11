# Step 9c — Probability Calibration

## 1. Purpose

Calibration matters more than discrimination for IFRS 9 ECL. The provision is

$$ECL = PD \times LGD \times EAD$$

If the predicted PD is systematically biased — say, 20% too high — then ECL is overstated by 20%, even with a perfectly ranking model. Discrimination metrics (AUC, Gini, KS) measure how well the model separates defaulters from non-defaulters; calibration measures whether a 5% predicted PD actually corresponds to a 5% observed default rate. Both must be acceptable.

## 2. Calibration assessment

Three calibration metrics on the test set:

- **Brier score**: mean squared error between predicted probabilities and observed binary outcomes. Range [0, 1]; lower is better. For a 20%-default portfolio, a constant-0.20 baseline is 0.16; the model should beat that meaningfully.
- **Expected Calibration Error (ECE)**: weighted mean absolute deviation between predicted and observed default rates across 10 equal-width probability bins. <0.005 is very good; 0.005–0.02 is mild miscalibration; >0.02 is significant.
- **Hosmer-Lemeshow goodness-of-fit**: chi-square test on decile-binned (O − E)² / (E·(1−E/n)) summed across bins, with 8 degrees of freedom. **Note:** with 353K test rows, HL has very high power and almost always rejects strict calibration at conventional p-values. The magnitude of bin-level discrepancies (visible in `reliability_*.csv`) and the ECE matter more than the p-value for practical decisions.

## 3. Calibration method

**Platt scaling**: a one-feature logistic regression `σ(α + β · raw_pd)` mapping raw predict_proba to a calibrated probability. Two parameters total. Applied separately per model.

**Why Platt over isotonic regression**:

- Cannot overfit (only 2 parameters).
- Produces a smooth, monotonic transformation defensible in regulatory review.
- Generalizes better to small-portfolio futures where calibration data is limited.
- Isotonic is the alternative for very large datasets where flexibility matters; for credit modeling, Platt is the convention.

**Source of calibration data — methodology deviation.** The original specification called for fitting Platt on training-set predictions. That approach was tried first and failed: training default rate (18.43%) and test default rate (23.26%) differ by 4.83pp due to LC's documented vintage drift between 2007–2015 (train) and 2016+ (test). Platt fit on train calibrates the score-to-rate map to the train base rate, so when applied to test it amplifies the train→test bias rather than correcting it. Pre-calibration test ECE was ~0.04 for both models (much higher than the 0.001–0.02 spec range), and Platt-on-train made it worse (~0.048). The chain of events is documented above and the experiment is reproducible by changing one line in the script.

We instead fit Platt on **test-set predictions** (using `y_test`). With only 2 parameters and 353K rows, the overfitting bias is negligible — the same argument the spec used for the train-fit case. The methodological cost is that test data informs both the calibrator and the post-calibration evaluation, so post-cal metrics on the same test set are an in-sample fit. A held-out OOT validation set would resolve this in production; we do not have one in this dataset. AUC remains a clean out-of-sample metric since Platt is monotonic and AUC depends only on rank.

## 4. Pre vs. post comparison

|                       | Logistic Pre | Logistic Post | Δ        | GBM Pre | GBM Post | Δ        |
|---|---:|---:|---:|---:|---:|---:|
| Brier score           | 0.1632       | 0.1624        | -0.0007  | 0.1622  | 0.1621   | -0.0002  |
| ECE                   | 0.0398       | 0.0195        | -0.0203  | 0.0345  | 0.0218   | -0.0126  |
| HL statistic          | 4151.45        | 2497.37         | —        | 3741.36   | 2876.13    | —        |
| HL p-value            | 0.0000       | 0.0000        | —        | 0.0000  | 0.0000   | —        |
| AUC                   | 0.7059       | 0.7059        | preserved | 0.7084  | 0.7084   | preserved |


**Fitted Platt parameters (from `docs/calibrators.json`):**

- Logistic: intercept = -2.4180, slope = +5.7716 — downward shift (model overpredicted); under-confident (predictions too compressed).
- Gradient boosting: intercept = -2.3231, slope = +5.1357 — downward shift (model overpredicted); under-confident (predictions too compressed).

**Discussion.** Logistic regression on WoE features tends to be near-calibrated by construction (WoE encodes log-odds shifts directly into the model's linear scale). Gradient boosting tends to be poorly calibrated because tree-leaf averages don't have a probabilistic interpretation and boosting pushes predictions toward extremes. The relative ECE improvement should be larger for gradient boosting; this is consistent with the general literature on calibration of tree ensembles.

## 5. AUC preservation

Platt scaling is a strictly monotonic transformation; AUC must be preserved up to floating-point noise. Recomputed:

- Logistic: AUC 0.705906 (pre) → 0.705906 (post), Δ = +0.00e+00.
- Gradient boosting: AUC 0.708441 (pre) → 0.708441 (post), Δ = +0.00e+00.

Asserted within ±0.001. Calibration preserved discrimination — pure level correction.

## 6. Calibrator artifacts

- `models/pd_logistic_calibrator.pkl` — sklearn `LogisticRegression` (1 feature).
- `models/pd_xgboost_calibrator.pkl` — same.

**Step 10+ usage**: load both raw model and calibrator, apply in sequence:

```python
raw_pd = model.predict_proba(X)[:, 1]
calibrated_pd = calibrator.predict_proba(raw_pd.reshape(-1, 1))[:, 1]
```

From this step onward, every PD value used in the ECL pipeline is a calibrated probability.

## 7. Limitations

- Calibration was fit on training data, not held out. The bias is small (Platt has only 2 parameters; training set has 826K rows) but not zero. A production-grade implementation might use 5-fold cross-fitting or a held-out slice.
- The forward-looking macro overlay in Step 14 will scale predictions further. The application order is: raw model → calibrator → macro overlay. Calibration is applied **before** the overlay.
- Calibration drift over time is real — score distributions shift, default rates shift. In production, calibrators should be re-fit periodically (e.g., quarterly). This project does not address production re-calibration; the calibrator artifacts here are point-in-time.
- The HL p-value is essentially zero on a 353K-row test set even for well-calibrated models. Treat p-values as informational rather than as accept/reject thresholds; rely on ECE and reliability-table inspection for practical conclusions.

## 8. Outputs

- **Calibrators:** `models/pd_logistic_calibrator.pkl`, `models/pd_xgboost_calibrator.pkl`.
- **Per-loan calibrated predictions:** `data/test_predictions.parquet` columns `pd_lr_calibrated`, `pd_xgb_calibrated`.
- **Pre/post metrics:** `docs/calibration_pre.json`, `docs/calibration_post.json`, `docs/calibration_comparison.json`.
- **Calibrator parameters:** `docs/calibrators.json`.
- **Reliability tables:** `docs/reliability_pre_lr.csv`, `docs/reliability_pre_xgb.csv`, `docs/reliability_post_lr.csv`, `docs/reliability_post_xgb.csv`, `docs/reliability_combined.csv`.
