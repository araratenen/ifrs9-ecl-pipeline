# Pipeline Validation Report — Steps 7 and 8

Run timestamp: `2026-05-09T19:02:49`

## Task 1 — File existence and integrity

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 1.1 | loans_modeling_ready.parquet | 58.8 MB | OK |
| ✓ PASS | 1.2 | loans_with_macros.parquet | 59.1 MB | OK |
| ✓ PASS | 1.3 | macros_monthly.parquet | 0.0 MB | OK |
| ✓ PASS | 1.4 | feature_classification.json | 0.0 MB | OK |
| ✓ PASS | 1.5 | step7_methodology.md | 0.0 MB | OK |
| ✓ PASS | 1.6 | step8_methodology.md | 0.0 MB | OK |
| ✓ PASS | 1.7 | step7_observation_window.py | 0.0 MB | OK |
| ✓ PASS | 1.8 | step8_macro_features.py | 0.0 MB | OK |

## Task 2 — Schema validation

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 2.1 | Row counts match between Step 7 and Step 8 outputs | ready=1,179,687, macros=1,179,687 | equal |
| ✓ PASS | 2.2 | Loan IDs identical between the two parquets | ready=1,179,687, macros=1,179,687, sym_diff=0 | identical sets |
| ✓ PASS | 2.3 | Column delta = exactly 4 macros, no Step 7 cols dropped | added=['fedfunds', 'gdp_yoy', 'hpi_yoy', 'unrate'], removed=[] | added=[fedfunds,gdp_yoy,hpi_yoy,unrate], removed=[] |
| ✓ PASS | 2.4 | All 22 required columns present | all present | all present |
| ✓ PASS | 2.5 | Column types are sensible | all OK | datetime/int/float/cat as expected |

## Task 3 — Data quality re-verification

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 3.1 | No DNMCP rows | 0 rows | 0 |
| ✓ PASS | 3.2 | No dti == 999 sentinel | 0 rows; 72 NaN | 0 rows |
| ✓ PASS | 3.3 | revol_util ≤ 100 | max=100.0 | ≤ 100 |
| ✓ PASS | 3.4 | annual_inc winsorized ≤ $250,000 | max=$250,000 | ≤ $250,000 |
| ✓ PASS | 3.5 | fico_range_low ≥ 660 (LC issuance floor) | min=660 | ≥ 660 |
| ✓ PASS | 3.6 | months_observable ≥ 24 (maturity filter) | min=24 | ≥ 24 |
| ✓ PASS | 3.7 | No nulls in critical fields | 0 nulls in all fields | 0 nulls |

## Task 4 — Feature classification JSON validity

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 4.1 | JSON has 4 expected top-level keys | keys=['identifiers', 'label', 'outcome_only', 'pd_inputs'] | ['identifiers', 'label', 'outcome_only', 'pd_inputs'] |
| ✓ PASS | 4.2 | Label == 'default_flag' | label='default_flag' | 'default_flag' |
| ✓ PASS | 4.3 | pd_inputs ≈ 33 features incl. 4 macros + issue_year | count=33, macros=yes, issue_year=yes | 33±2 incl. macros + issue_year |
| ✓ PASS | 4.4 | No overlap between pd_inputs and outcome_only | no overlap | empty |
| ✓ PASS | 4.5 | id and issue_d in identifiers, not pd_inputs | id_OK=True, issue_d_OK=True | both in identifiers |
| ✓ PASS | 4.6 | pd_inputs and outcome_only resolvable to columns | all present | all present (issue_year derivable) |
| ✓ PASS | 4.7 | No leakage columns in pd_inputs | no leaks | empty |

## Task 5 — Distributional sanity

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 5.1 | Overall default rate in 19–21% | 19.87% | 19–21% |
| ✓ PASS | 5.2 | Default rate monotonic A→G | {'A': 6.0, 'B': 13.24, 'C': 22.22, 'D': 30.21, 'E': 38.59, 'F': 45.26, 'G': 49.85} | non-decreasing |
| ✓ PASS | 5.3 | Default rate monotonic ↓ across FICO bands | {'660-700': 23.39, '700-740': 15.84, '740-780': 10.07, '780+': 6.48} | non-increasing |
| ✓ PASS | 5.4 | All years have 5–30% default rate | {2007: 18.1, 2008: 15.8, 2009: 12.6, 2010: 12.9, 2011: 15.2, 2012: 16.2, 2013: 15.6, 2014: 18.4, 2015: 20.2, 2016: 23.3, 2017: 23.1} | 5–30% |
| ✓ PASS | 5.5 | Macros within plausible US ranges | all in range | see bounds |
| ✓ PASS | 5.6 | All macro columns zero nulls | 0 nulls in all macros | 0 nulls |

## Task 6 — Cross-deliverable consistency

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 6.1 | Step 7 cited row count matches actual | actual=1,179,687, cited=1,179,687 | within ±10 |
| ✓ PASS | 6.2 | Step 8 cited row count matches actual | actual=1,179,687, cited=1,179,687 | within ±10 |
| ✓ PASS | 6.3 | Step 7 cited default rate matches actual | actual=19.87%, cited=19.87% | within 0.1pp |
| ✓ PASS | 6.4 | Within-year correlations match cited values | actual={'unrate': -0.0062, 'gdp_yoy': 0.0047, 'fedfunds': 0.0071, 'hpi_yoy': -0.0096}, cited={'unrate': -0.0062, 'gdp_yoy': 0.0047, 'fedfunds': 0.0071, 'hpi_yoy': -0.0096}, max_diff=0.0000 | within ±0.001 |

## Task 7 — Modeling-readiness check

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 7.1 | Sample size > 500K | 1,179,687 | > 500,000 |
| ✓ PASS | 7.2 | Defaults ≥ 50K | 234,426 | ≥ 50,000 |
| ✓ PASS | 7.3 | Time-split feasible (issue_d < 2016 vs ≥ 2016) | train=826,604 (152,304 def), test=353,083 (82,122 def) | both > 100K rows + 10K defaults |
| ✓ PASS | 7.4 | All pd_inputs present or derivable | all OK | all OK |
| ✓ PASS | 7.5 | pd_inputs non-degenerate | all OK | ≥2 numeric/datetime; 2–100 cat |
| ✓ PASS | 7.6 | No pd_input column > 50% missing | all < 30% | ≤ 50% |

## Task 8 — Step 9a artifacts (WoE binning)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 8.1 | models/binning_process.pkl | 0.1 MB | OK |
| ✓ PASS | 8.2 | data/train.parquet | 41.7 MB | OK |
| ✓ PASS | 8.3 | data/test.parquet | 18.6 MB | OK |
| ✓ PASS | 8.4 | data/train_woe.parquet | 9.7 MB | OK |
| ✓ PASS | 8.5 | data/test_woe.parquet | 4.4 MB | OK |
| ✓ PASS | 8.6 | docs/binning_summary.json | 0.0 MB | OK |
| ✓ PASS | 8.7 | docs/binning_tables/_summary.csv | 0.0 MB | OK |
| ✓ PASS | 8.8 | train_woe row count matches train | train=826,604, train_woe=826,604 | equal |
| ✓ PASS | 8.9 | test_woe row count matches test | test=353,083, test_woe=353,083 | equal |
| ✓ PASS | 8.10 | Zero nulls in WoE parquets | train=0, test=0 | 0 |
| ✓ PASS | 8.11 | IV values reasonable (≥0 and ≤0.8) | min=0.0000, max=0.5037 | 0 ≤ iv ≤ 0.8 |
| ✓ PASS | 8.12 | Per-feature binning CSV exists for every selected feature | 19 CSVs OK (16 IV + 3 forced) | all present |

## Task 9 — Step 9b artifacts (PD models)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 9.1 | models/pd_logistic.pkl | 0.0 MB | OK |
| ✓ PASS | 9.2 | models/pd_xgboost.pkl | 0.4 MB | OK |
| ✓ PASS | 9.3 | data/test_predictions.parquet | 36.1 MB | OK |
| ✓ PASS | 9.4 | docs/model_evaluation.json | 0.0 MB | OK |
| ✓ PASS | 9.5 | docs/coefficients_lr.csv | 0.0 MB | OK |
| ✓ PASS | 9.6 | docs/feature_importance_xgb.csv | 0.0 MB | OK |
| ✓ PASS | 9.7 | docs/step9b_methodology.md | 0.0 MB | OK |
| ✓ PASS | 9.8 | pd_logistic.pkl is a LogisticRegression | type=LogisticRegression | LogisticRegression |
| ✓ PASS | 9.9 | pd_xgboost.pkl is a sklearn classifier with predict_proba | type=HistGradientBoostingClassifier | has predict_proba |
| ✓ PASS | 9.10 | test_predictions.parquet has required columns + no PD nulls | cols=['default_flag', 'ead_12m', 'ead_lifetime_discounted_total', 'ecl_12m', 'ecl_final', 'ecl_lifetime', 'ecl_total', 'ecl_total_adverse', 'ecl_total_baseline', 'ecl_total_severe', 'id', 'ifrs9_stage', 'issue_d', 'lgd_predicted', 'months_remaining', 'pd_12m', 'pd_lifetime', 'pd_lr', 'pd_lr_calibrated', 'pd_xgb', 'pd_xgb_calibrated'], null_pd=0 | ['default_flag', 'id', 'issue_d', 'pd_lr', 'pd_xgb'], 0 nulls |
| ✓ PASS | 9.11 | Both models AUC in [0.65, 0.85] | LR=0.7059, GBM=0.7084 | [0.65, 0.85] |
| ✓ PASS | 9.12 | coefficients_lr.csv has 19 rows | rows=19 | 19 |
| ✓ PASS | 9.13 | feature_importance_xgb.csv has 19 rows | rows=19 | 19 |

## Task 10 — Step 9c artifacts (Platt calibration)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 10.1 | models/pd_logistic_calibrator.pkl | 0.0 MB | OK |
| ✓ PASS | 10.2 | models/pd_xgboost_calibrator.pkl | 0.0 MB | OK |
| ✓ PASS | 10.3 | docs/calibration_comparison.json | 0.0 MB | OK |
| ✓ PASS | 10.4 | docs/calibrators.json | 0.0 MB | OK |
| ✓ PASS | 10.5 | docs/reliability_combined.csv | 0.0 MB | OK |
| ✓ PASS | 10.6 | docs/step9c_methodology.md | 0.0 MB | OK |
| ✓ PASS | 10.7 | Calibrators are LogisticRegression instances | LR=LogisticRegression, GBM=LogisticRegression | LogisticRegression |
| ✓ PASS | 10.8 | test_predictions has calibrated columns with no nulls | cols=True, nulls=0 | both cols present, 0 nulls |
| ✓ PASS | 10.9 | Calibrated PDs in [0, 1] | LR=[0.0887, 0.8200], GBM=[0.0921, 0.8290] | [0, 1] |
| ✓ PASS | 10.10 | Post-ECE ≤ Pre-ECE (calibration didn't worsen) | LR: 0.0398→0.0195; GBM: 0.0345→0.0218 | post ≤ pre |
| ✓ PASS | 10.11 | reliability_combined.csv has 10 deciles | rows=10 | 10 |
| ✓ PASS | 10.12 | AUC preserved within ±0.001 by calibration | LR Δ=+0.00e+00, GBM Δ=+0.00e+00 | \|Δ\| < 0.001 |

## Task 11 — Step 10 artifacts (LGD)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 11.1 | data/loans_with_lgd.parquet | 60.0 MB | OK |
| ✓ PASS | 11.2 | docs/lgd_lookup.csv | 0.0 MB | OK |
| ✓ PASS | 11.3 | docs/lgd_histogram.csv | 0.0 MB | OK |
| ✓ PASS | 11.4 | docs/lgd_backtest.csv | 0.0 MB | OK |
| ✓ PASS | 11.5 | docs/lgd_sensitivity.csv | 0.0 MB | OK |
| ✓ PASS | 11.6 | docs/step10_methodology.md | 0.0 MB | OK |
| ✓ PASS | 11.7 | loans_with_lgd row count == loans_with_macros | macros=1,179,687, lgd=1,179,687 | equal |
| ✓ PASS | 11.8 | lgd_predicted has no nulls and is in [0, 1] | nulls=0, range=[0.8891, 0.9032] | 0 nulls, [0, 1] |
| ✓ PASS | 11.9 | test_predictions has lgd_predicted, no nulls | has_col=True, nulls=0 | present + 0 nulls |
| ✓ PASS | 11.10 | lgd_lookup.csv has non-null estimates | rows=98, nulls=0 | non-empty, no nulls |
| ✓ PASS | 11.11 | Validation predicted ≈ observed within ±0.05 | observed=0.9157, predicted=0.8959, \|diff\|=0.0199 | ≤ 0.05 |

## Task 12 — Step 11 artifacts (EAD)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 12.1 | data/loans_with_ead.parquet | 83.0 MB | OK |
| ✓ PASS | 12.2 | docs/ead_histogram.csv | 0.0 MB | OK |
| ✓ PASS | 12.3 | docs/ead_months_remaining_distribution.csv | 0.0 MB | OK |
| ✓ PASS | 12.4 | docs/ead_status_breakdown.csv | 0.0 MB | OK |
| ✓ PASS | 12.5 | docs/step11_methodology.md | 0.0 MB | OK |
| ✓ PASS | 12.6 | loans_with_ead row count == loans_with_lgd | lgd=1,179,687, ead=1,179,687 | equal |
| ✓ PASS | 12.7 | EAD columns present and non-null | missing=[], nulls=0 | all present, 0 nulls |
| ✓ PASS | 12.8 | ead_12m ≥ 0 for all loans | negative=0 | 0 |
| ✓ PASS | 12.9 | ead_lifetime_path length == months_remaining (2K sample) | bad_path_len=0 | 0 |
| ✓ PASS | 12.10 | discount_factors length == months_remaining (2K sample) | bad_disc_len=0 | 0 |
| ✓ PASS | 12.11 | Inactive loans have empty path | non_empty=0 | 0 |
| ✓ PASS | 12.12 | Aggregate ratio EAD_12m / funded in [0.05, 0.95] | ratio=0.0731 | [0.05, 0.95] |
| ✓ PASS | 12.13 | test_predictions has new EAD columns, no nulls | has_cols=True, nulls=0 | present + 0 nulls |

## Task 13 — Step 12 artifacts (ECL)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 13.1 | data/loans_with_ecl.parquet | 107.7 MB | OK |
| ✓ PASS | 13.2 | docs/ecl_headline.json | 0.0 MB | OK |
| ✓ PASS | 13.3 | docs/ecl_by_stage.csv | 0.0 MB | OK |
| ✓ PASS | 13.4 | docs/ecl_by_grade.csv | 0.0 MB | OK |
| ✓ PASS | 13.5 | docs/ecl_by_vintage.csv | 0.0 MB | OK |
| ✓ PASS | 13.6 | docs/ecl_by_purpose.csv | 0.0 MB | OK |
| ✓ PASS | 13.7 | docs/step12_methodology.md | 0.0 MB | OK |
| ✓ PASS | 13.8 | loans_with_ecl row count == loans_with_ead | ead=1,179,687, ecl=1,179,687 | equal |
| ✓ PASS | 13.9 | ECL columns present, non-null, non-negative | nulls=0, negatives=0 | 0 / 0 |
| ✓ PASS | 13.10 | Headline JSON total_ecl matches parquet sum | json=$403,536,500.57, parquet=$403,536,500.57, \|diff\|=$0.0000 | \|diff\| < $1 |
| ✓ PASS | 13.11 | Σ stage ECL == total ECL | \|diff\|=$0.0000 | \|diff\| < $1 |
| ✓ PASS | 13.12 | ecl_12m ≤ ecl_lifetime (long loans) | violations=0 | 0 |
| ✓ PASS | 13.13 | ECL ≤ EAD × LGD invariant | violations=0 | 0 |
| ✓ PASS | 13.14 | test_predictions has Step 12 columns, no nulls | has=True, nulls=0 | present + 0 nulls |

## Task 14 — Step 13 artifacts (Overlay)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 14.1 | data/loans_with_ecl_overlay.parquet | 164.6 MB | OK |
| ✓ PASS | 14.2 | docs/ecl_overlay_headline.json | 0.0 MB | OK |
| ✓ PASS | 14.3 | docs/ecl_overlay_by_stage.csv | 0.0 MB | OK |
| ✓ PASS | 14.4 | docs/ecl_overlay_by_grade.csv | 0.0 MB | OK |
| ✓ PASS | 14.5 | docs/ecl_overlay_by_vintage.csv | 0.0 MB | OK |
| ✓ PASS | 14.6 | docs/step13_methodology.md | 0.0 MB | OK |
| ✓ PASS | 14.7 | loans_with_ecl_overlay row count == loans_with_ecl | overlay=1,179,687, ecl=1,179,687 | equal |
| ✓ PASS | 14.8 | Overlay columns non-null and non-negative | nulls=0, negatives=0 | 0 / 0 |
| ✓ PASS | 14.9 | Scenario weights sum to 1.0 | sum=1.0 | 1.0 |
| ✓ PASS | 14.10 | Baseline ECL matches Step 12 within $1 | \|diff\|=$0.0000 | < $1 |
| ✓ PASS | 14.11 | Aggregate base/final/severe ordered (monotonic, either direction) | base=$403,536,501, final=$390,821,719, sev=$376,233,658 — inverted (LC underwriting-reaction) | ordered (direction documented) |
| ✓ PASS | 14.12 | test_predictions has overlay columns, no nulls | has=True, nulls=0 | present + 0 nulls |

## Task 15 — Step 14 artifacts (Validation pack)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 15.1 | validation_discrimination.json | 0.0 MB | OK |
| ✓ PASS | 15.2 | validation_calibration.json | 0.0 MB | OK |
| ✓ PASS | 15.3 | validation_auc_by_vintage.csv | 0.0 MB | OK |
| ✓ PASS | 15.4 | validation_auc_by_grade.csv | 0.0 MB | OK |
| ✓ PASS | 15.5 | validation_gain_curve.csv | 0.0 MB | OK |
| ✓ PASS | 15.6 | validation_reliability_test.csv | 0.0 MB | OK |
| ✓ PASS | 15.7 | validation_sicr_sensitivity.csv | 0.0 MB | OK |
| ✓ PASS | 15.8 | validation_overlay_weights.csv | 0.0 MB | OK |
| ✓ PASS | 15.9 | validation_regulatory_overlay.json | 0.0 MB | OK |
| ✓ PASS | 15.10 | validation_psi_over_time.csv | 0.0 MB | OK |
| ✓ PASS | 15.11 | validation_single_feature_stress.csv | 0.0 MB | OK |
| ✓ PASS | 15.12 | validation_stage_migration.csv | 0.0 MB | OK |
| ✓ PASS | 15.13 | final_validation_report.md | 0.0 MB | OK |
| ✓ PASS | 15.14 | Final report has all 10 sections | all 10 sections present | all 10 headers |
| ✓ PASS | 15.15 | Three headline numbers present in final report | baseline=True, overlay=True, regulatory=True | all three present |
| ✓ PASS | 15.16 | Regulatory overlay ECL > baseline (validates plausibility) | reg=$484,474,530 > base=$403,536,501 = True | reg > base |
| ✓ PASS | 15.17 | validation_sicr_sensitivity.csv has expected columns | cols=['change_from_2x_pct', 'stage2_share_pct', 'threshold_rule', 'total_ecl_baseline'] | expected columns |

## Task 16 — Step 15 artifacts (Dashboard data)

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 16.1 | data/dashboard/loans_summary.csv | 341.4 MB | OK |
| ✓ PASS | 16.2 | data/dashboard/headline_metrics.csv | 0.0 MB | OK |
| ✓ PASS | 16.3 | data/dashboard/discrimination_metrics.csv | 0.0 MB | OK |
| ✓ PASS | 16.4 | data/dashboard/calibration_table.csv | 0.0 MB | OK |
| ✓ PASS | 16.5 | data/dashboard/sensitivity_table.csv | 0.0 MB | OK |
| ✓ PASS | 16.6 | docs/dashboard_spec.md | 0.0 MB | OK |
| ✓ PASS | 16.7 | loans_summary.csv row count matches loans_with_ecl_overlay | dashboard=1,179,687, parquet=1,179,687 | equal |
| ✓ PASS | 16.8 | headline_metrics has rows for all three versions | versions=['baseline', 'data_overlay', 'regulatory'] | {'baseline', 'data_overlay', 'regulatory'} ⊆ versions |
| ✓ PASS | 16.9 | dashboard_spec.md has all 7 sections | all sections present | all 7 sections |

## Task 17 — Unified project summary

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 17.1 | docs/project_summary.md exists, non-empty | 0.0 MB | OK |
| ✓ PASS | 17.2 | All 12 H2 section headers present | all 12 sections | 12 H2 headers |
| ✓ PASS | 17.3 | Three headline numbers present in summary | all 3 present | 3 headlines |

## Task 18 — Final project dossier

| Status | ID | Check | Value | Expected |
|---|---|---|---|---|
| ✓ PASS | 18.1 | docs/final_project_dossier.md exists, non-empty | 0.1 MB | OK |
| ✓ PASS | 18.2 | All 15 H2 section headers present (Section 0–14) | all 15 sections | 15 H2 headers |
| ✓ PASS | 18.3 | Word count within 5,000–10,000 range | 8,856 words | 5,000 ≤ wc ≤ 10,000 |
| ✓ PASS | 18.4 | Three headline numbers present in dossier | all 3 present | 3 headlines |
| ✓ PASS | 18.5 | ≥50 inline verification tags present | 73 tags | ≥50 |
| ✓ PASS | 18.6 | Zero VERIFY FAILED tags in body | 0 failed | 0 |
| ✓ PASS | 18.7 | Verification appendix table present (Section 13) | appendix table found | appendix table in Section 13 |
| ✓ PASS | 18.8 | File index (Section 12) present with category groupings | all 4 categories present | Cleaned data / Models / Validation / Source code |

## Validation Summary

- Total checks: **167**
- Passed: **167**
- Failed: **0**
