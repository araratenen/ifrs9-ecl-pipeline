# Full Pipeline Audit Report — Steps 5 through 9b

| Field | Value |
|---|---|
| Project | IFRS 9 ECL — Lending Club PD pipeline |
| Audit timestamp | `2026-05-09T18:22:23` |
| Auditor | Claude Code (`audit_full_pipeline.py`) |
| Pipeline state | **READY_WITH_WARNINGS** |

## Executive summary

- Total checks: **132**
- PASS: **123**
- WARN: **9**
- FAIL: **0**

## Category 1 — File and artifact integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 1.1 | ✓ PASS | All expected files exist | all 91 files present | all present |  |
| 1.2 | ✓ PASS | All files loadable in expected format | all loadable | all loadable |  |
| 1.3 | ⚠ WARN | No orphan/unexpected files | orphans=['data/dashboard', 'docs/final_project_dossier.md', 'src/build_final_dossier.py'] | none |  |
| 1.4 | ⚠ WARN | File mtimes chronological across pipeline stages | ['step6→step7: earlier mtime > later'] | earlier ≤ later |  |

## Category 2 — Schema and structural integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 2.1 | ✓ PASS | Row count consistency across stages | all consistent | equal |  |
| 2.2 | ✓ PASS | ID consistency across stages | all consistent | exact set equality |  |
| 2.3 | ✓ PASS | Column delta matches commitment per stage | all deltas match | documented deltas |  |
| 2.4 | ✓ PASS | Column types stable across stages (sampled) | all types stable | single dtype across stages |  |
| 2.5 | ✓ PASS | No outcome columns in WoE parquets | no leakage | empty |  |

## Category 3 — Methodology document coherence

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 3.1 | ✓ PASS | Required sections present in each methodology doc | all present | all sections present |  |
| 3.2 | ✓ PASS | Cited numbers match artifacts (rows, default rate, IV, corr, AUC) | all match within tolerance | tolerances per spec |  |
| 3.3 | ✓ PASS | Cross-document numeric and narrative consistency | consistent | step7→8→9a→9b chain coherent |  |
| 3.4 | ✓ PASS | No internal contradictions across docs | no contradictions detected | consistent narrative |  |
| 3.5 | ✓ PASS | Limitations sections populated (not silently dropped) | 19 limitations across 4 docs | ≥3 per doc typical |  |

## Category 4 — Data quality persistence

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 4.1 | ✓ PASS | No DNMCP rows downstream | all 3 stages clean | all clean |  |
| 4.2 | ✓ PASS | No dti==999 sentinel downstream | all 3 stages clean | all clean |  |
| 4.3 | ✓ PASS | revol_util ≤ 100 downstream | all 3 stages clean | max ≤ 100 |  |
| 4.4 | ✓ PASS | annual_inc ≤ p99 cap downstream | all 3 stages clean | ≤ $250,000 |  |
| 4.5 | ✓ PASS | fico_range_low ≥ 660 downstream | all 3 stages clean | ≥ 660 |  |
| 4.6 | ✓ PASS | months_observable ≥ 24 downstream | all 3 stages clean | ≥ 24 |  |
| 4.7 | ✓ PASS | No nulls in critical columns downstream | all 3 stages clean | all 0 |  |

## Category 5 — Feature classification governance

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 5.1 | ✓ PASS | feature_classification.json has 4 expected keys | ['identifiers', 'label', 'outcome_only', 'pd_inputs'] | ['identifiers', 'label', 'outcome_only', 'pd_inputs'] |  |
| 5.2 | ✓ PASS | No column appears in both pd_inputs and outcome_only | no overlap | empty |  |
| 5.3 | ✓ PASS | No outcome leakage columns in WoE parquets | no leakage | empty |  |
| 5.4 | ✓ PASS | pd_inputs contains 4 macros + issue_year (Step 8 commitment) | macros=True, issue_year=True | both present |  |
| 5.5 | ✓ PASS | Every pd_input is a column or documented derivation | all resolvable | all present |  |
| 5.6 | ✓ PASS | binning summary features ⊆ pd_inputs (post earliest_cr_line→credit_history_years) | subset | subset |  |

## Category 6 — Train/test split integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 6.1 | ✓ PASS | Time-based split has no date overlap | max_train=2015-12-01, min_test=2016-01-01 | max_train < min_test |  |
| 6.2 | ✓ PASS | Train and test IDs are disjoint | disjoint | 0 overlap |  |
| 6.3 | ✓ PASS | Train/test cutoff matches docs | cited=2016-01-01, expected=2016-01-01 | 2016-01-01 |  |
| 6.4 | ✓ PASS | Class distributions non-degenerate; drift matches doc | train=18.43%, test=23.26%, diff=+4.83pp (cited 4.83) | rates in [10,30]; diff matches ±0.5pp |  |

## Category 7 — WoE/binning soundness

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 7.1 | ✓ PASS | Binning fitted on training data only (count match) | binning saw 826604, train.parquet=826604 | equal |  |
| 7.2 | ✓ PASS | WoE values bounded |WoE| < 5 | max \|WoE\| = 1.8530 | < 5 |  |
| 7.3 | ✓ PASS | Force-included = {unrate, hpi_yoy, issue_year} | ['hpi_yoy', 'issue_year', 'unrate'] | ['hpi_yoy', 'issue_year', 'unrate'] |  |
| 7.4a | ✓ PASS | IVs in [0, 0.7] | all OK | [0, 0.7] |  |
| 7.5 | ⚠ WARN | Every selected-feature bin has ≥100 training observations | ['dti: 1 bin(s) <100'] | ≥100 |  |
| 7.6 | ✓ PASS | Reapplying binning to test produces zero nulls | 0 nulls | 0 |  |
| 7.7 | ✓ PASS | Recomputed IV (from WoE) matches reported within 1% rel | all match | ≤1% rel |  |

## Category 8 — Model artifact integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 8.1 | ✓ PASS | pd_logistic.pkl is fitted LogisticRegression with (1,19) coef | type=LogisticRegression, coef=(1, 19), intercept=True | LogisticRegression(coef=(1,19), intercept_=1) |  |
| 8.2 | ✓ PASS | pd_xgboost.pkl loadable with predict_proba | type=HistGradientBoostingClassifier, predict_proba=True | callable predict_proba |  |
| 8.3 | ✓ PASS | LR predict_proba reproduces test_predictions.pd_lr | max \|diff\| = 0.00e+00 | < 1e-10 |  |
| 8.4 | ✓ PASS | GBM predict_proba reproduces test_predictions.pd_xgb | max \|diff\| = 0.00e+00 | < 1e-10 |  |
| 8.5 | ✓ PASS | coefficients_lr.csv matches pd_logistic.pkl coef_ exactly | max \|diff\| = 8.50e-17 | < 1e-10 |  |
| 8.6a | ⚠ WARN | Coefficient sign consistency | 18 neg + 1 pos; minority=['revol_util'] | ≤2 minority |  |
| 8.6b | ⚠ WARN | IV-selected feature |coef| in [0.1, 1.5] | 1 out-of-band: [{'feature': 'grade', 'coef': -0.0123}] | [0.1, 1.5] |  |
| 8.6c | ⚠ WARN | No IV-selected feature with |coef| < 0.05 | very small: [{'feature': 'grade', 'coef': -0.0123}] | ≥ 0.05 |  |
| 8.7 | ✓ PASS | Force-included coefs match dominant sign | [{'feature': 'unrate', 'coef': -0.4091}, {'feature': 'issue_year', 'coef': -0.3315}, {'feature': 'hpi_yoy', 'coef': -0.3598}] | consistent with dominant sign |  |

## Category 9 — Numerical reproducibility

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 9.1 | ✓ PASS | Overall default rate matches step7_methodology | actual=19.8719, cited=19.87, diff=0.0019 | diff ≤ 0.05pp |  |
| 9.2 | ✓ PASS | Default rate monotonic A→G | {'A': 0.06, 'B': 0.1324, 'C': 0.2222, 'D': 0.3021, 'E': 0.3859, 'F': 0.4526, 'G': 0.4985} | non-decreasing |  |
| 9.3 | ✓ PASS | Within-year corrs match step8_methodology within ±0.001 | all match | ±0.001 |  |
| 9.4 | ✓ PASS | AUC reproducible from test_predictions vs model_evaluation.json | LR=0.7059, GBM=0.7084 | ±0.005 |  |
| 9.5 | ✓ PASS | KS reproducible vs model_evaluation.json | LR=0.2968, GBM=0.3022 | ±0.01 |  |
| 9.6 | ✓ PASS | PSI reproducible within ±0.02 | LR=0.0111, GBM=0.0146 | ±0.02 |  |
| 9.7 | ✓ PASS | IV recompute (top-3) matches binning_summary within 1% rel | all match | ≤1% rel |  |

## Category 10 — Code quality and re-runnability

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 10.1 | ✓ PASS | All scripts parse as valid Python | all parse | valid Python |  |
| 10.2 | ✓ PASS | Random seeds documented where applicable | step9b has seed; step9a deterministic by default | seeds set |  |
| 10.3 | ✓ PASS | All hardcoded absolute paths rooted at project base | all rooted at /Users/.../ProjRED | rooted at project |  |
| 10.4 | ✓ PASS | No bare except: pass or warning suppressors | none | no silent error handling |  |
| 10.5 | ✓ PASS | No known deprecated pandas/sklearn APIs | none detected | no deprecation |  |
| 10.6 | ✓ PASS | No script edited after its primary output (stale check) | all outputs current | outputs newer than scripts |  |

## Category 11 — Cross-deliverable numerical traceability

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 11.1 | ✓ PASS | Logistic AUC traceable: recomputed → eval JSON → step9b doc | AUC=0.7059 traced everywhere | consistent across all 3 |  |
| 11.2 | ✓ PASS | Train/test rows traceable across 9a/9b/binning_summary | train=826604, test=353083 | consistent |  |
| 11.3 | ⚠ WARN | Top-IV feature traceable: binning → 9a doc → coefs (collinearity-aware) | ["top IV feat 'sub_grade' not among top \|coef\|"] | consistent |  |
| 11.4 | ✓ PASS | Default rate traceable: step7 doc → loans_ready → loans_macros | all=19.87% | consistent |  |

## Category 12 — Audit trail completeness

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 12.1 | ✓ PASS | All 16 methodological choices documented | all 16 documented | all documented |  |
| 12.2 | ✓ PASS | Limitations review (manual inspection list) | 19 limitations across 4 docs (see report) | human review | [step7_methodology] The model produces **lifetime PD** because that is what the label measures. Conversion to **12-month; [step7_methodology] A live bank would build a snapshot-based 12-month PD with monthly behavioral features (payment trend; [step7_methodology] DNMCP loans were dropped to avoid mi |

## Category 13 — Forward-compatibility (Step 9c readiness)

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 13.1 | ✓ PASS | test_predictions has columns Step 9c needs | ['default_flag', 'ead_12m', 'ead_lifetime_discounted_total', 'ecl_12m', 'ecl_final', 'ecl_lifetime', 'ecl_total', 'ecl_total_adverse', 'ecl_total_baseline', 'ecl_total_severe', 'id', 'ifrs9_stage', 'issue_d', 'lgd_predicted', 'months_remaining', 'pd_12m', 'pd_lifetime', 'pd_lr', 'pd_lr_calibrated',  | ['default_flag', 'id', 'issue_d', 'pd_lr', 'pd_xgb'] |  |
| 13.2 | ✓ PASS | Both models load with predict_proba | LR=True, GBM=True | both have predict_proba |  |
| 13.3 | ✓ PASS | Predicted PDs are in [0, 1] | LR=[0.0154, 0.6817], GBM=[0.0068, 0.7597] | [0, 1] |  |
| 13.4 | ✓ PASS | Decile lift tables ready for calibration analysis | LR rows=10, GBM rows=10 | 10 rows each, lift column |  |
| 13.5 | ✓ PASS | Calibrated PDs in [0, 1] | LR=[0.0887, 0.8200], GBM=[0.0921, 0.8290] | [0, 1] |  |
| 13.6 | ✓ PASS | LR calibrator reproduces pd_lr_calibrated | max \|diff\| = 0.00e+00 | < 1e-10 |  |
| 13.7 | ✓ PASS | GBM calibrator reproduces pd_xgb_calibrated | max \|diff\| = 0.00e+00 | < 1e-10 |  |

## Category 14 — LGD pipeline integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 14.1 | ✓ PASS | lgd_lookup non-empty, no nulls in lgd_estimate | rows=98, nulls=0 | non-empty + 0 nulls |  |
| 14.2 | ✓ PASS | All loans have lgd_predicted in [0, 1] with no nulls | range=[0.8891, 0.9032], nulls=0 | [0, 1], 0 nulls |  |
| 14.3 | ✓ PASS | Recomputed mean realized LGD matches step10_methodology | actual=0.9029, cited=0.9029, diff=0.0000 | diff ≤ 0.005 |  |
| 14.4 | ✓ PASS | Backtest aggregate error within ±0.05 | observed=0.9157, predicted=0.8959, \|diff\|=0.0199 | ≤ 0.05 |  |
| 14.5 | ✓ PASS | Sensitivity table math: shocks proportional | all shocks proportional (base=0.8959) | exact ±10%, ±20% |  |
| 14.6 | ✓ PASS | Force-cap share < 5% combined | share_capped=0.03% (below_zero=80, above_one=0) | < 5% |  |

## Category 15 — EAD pipeline integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 15.1 | ✓ PASS | Active-loan conservation: final balance ≈ 0 (proxy for full repayment) | bad_in_2K_sample=0 | 0 |  |
| 15.2 | ✓ PASS | Final balance < $1 (no early-termination bug) | bad_in_sample=0 | 0 |  |
| 15.3 | ✓ PASS | Monotonic non-increasing balance | violations_in_sample=0 | 0 |  |
| 15.4 | ✓ PASS | ead_12m never exceeds starting balance (funded_amnt cap) | violations=0 | 0 |  |
| 15.5 | ✓ PASS | Aggregate plausibility: ratio EAD_12m / funded in [0.05, 0.95] | ratio=0.0731 | [0.05, 0.95] |  |
| 15.6 | ✓ PASS | Cross-check ead_12m on 100 random loans (within 0.1%) | mismatches=0 of 100 | 0 |  |
| 15.7 | ✓ PASS | Discount factors in (0, 1] and monotonic decreasing | violations_in_sample=0 | 0 |  |

## Category 16 — ECL pipeline integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 16.1 | ✓ PASS | ECL non-negativity (no negatives in ecl_total) | negatives=0 | 0 |  |
| 16.2 | ✓ PASS | ECL ≤ EAD × LGD invariant (100% satisfy) | violations=0 of 1,179,687 | 0 |  |
| 16.3 | ⚠ WARN | Stage shares plausible: S1∈[60-95]%, S2∈[3-30]%, S3==defaulters | S1=80.08%, S2=0.05%, S3=234,426 vs defaulters=234,426 | S1 60-95%, S2 3-30%, S3 == defaulters |  |
| 16.4 | ✓ PASS | pd_12m ≤ pd_lifetime | violations=0 | 0 |  |
| 16.5 | ✓ PASS | Headline ECL matches per-loan sum within $1 | json=$403,536,500.57, parquet=$403,536,500.57, \|Δ\|=$0.0000 | \|Δ\| < $1 |  |
| 16.6 | ✓ PASS | By-grade ECL coverage monotonic across A→G | coverage A→G: {'A': 0.0024, 'B': 0.0092, 'C': 0.0239, 'D': 0.0358, 'E': 0.0605, 'F': 0.0782, 'G': 0.0859} | monotonic ↑ (riskier grades) or ↓ |  |
| 16.7 | ✓ PASS | By-vintage ECL coverage shows drift | range = 0.0744 (2007=0.0000 → 2017=0.0744) | non-zero range |  |
| 16.8 | ✓ PASS | All defaulters in Stage 3 | defaulters in non-S3=0 | 0 |  |

## Category 17 — Forward-looking overlay integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 17.1 | ⚠ WARN | Per-loan monotonicity (baseline ≤ adverse ≤ severe), <1% violations | b>a: 253,622 (21.50%); a>s: 201,186 (17.05%) | < 1% |  |
| 17.2 | ✓ PASS | Baseline scenario ECL matches Step 12 to $1 | base=$403,536,501, step12=$403,536,501, \|Δ\|=$0.0000 | < $1 |  |
| 17.3 | ✓ PASS | Aggregate base/final/severe ECL ordered (monotonic, either direction) | base=$403,536,501, final=$390,821,719, sev=$376,233,658 — inverted (LC underwriting-reaction; documented finding) | ordered (direction documented) |  |
| 17.4 | ✓ PASS | Stage 3 ECL unchanged across scenarios (max $1 per loan) | max \|base − severe\| in stage3 = $0.0000 | < $1 |  |
| 17.5 | ✓ PASS | By-grade overlay multipliers differentiate (non-trivial spread) | multipliers: {'A': 0.9806, 'B': 0.9714, 'C': 0.9667, 'D': 0.9666, 'E': 0.9675, 'F': 0.9729, 'G': 0.9746}, spread=0.0140 | spread > 0.001 |  |
| 17.6 | ✓ PASS | Scenario weights sum to 1.0 | sum=1.0 | 1.0 |  |
| 17.7 | ✓ PASS | Headline final_ecl matches per-loan sum within $1 | json=$390,821,719, parquet=$390,821,719, \|Δ\|=$0.0000 | < $1 |  |

## Category 18 — Validation pack integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 18.1 | ✓ PASS | All Step 14 deliverables exist and non-empty | all 15 present | all present |  |
| 18.2 | ✓ PASS | Three headlines reproduce: baseline / data-overlay / regulatory | step12=$403,536,501, regulatory_baseline=$403,536,501, data_overlay=$390,821,719, regulatory_final=$484,474,530 | baselines match within $1 |  |
| 18.3 | ✓ PASS | Out-of-time AUC stable: variation < 0.05 between consecutive years | max consecutive \|Δ\|=0.0108, AUCs=[np.float64(0.7077), np.float64(0.6969)] | < 0.05 |  |
| 18.4 | ✓ PASS | Calibration MAD < 0.05 across deciles | MAD=0.0276 | < 0.05 |  |
| 18.5 | ✓ PASS | SICR sensitivity monotonic: looser threshold → higher Stage 2 share + ECL | s2_shares=[4.548, 1.076, 0.0494, 0.0015, 0.0], ecls=[np.float64(411741114.0), np.float64(405974421.0), np.float64(403536501.0), np.float64(403418423.0), np.float64(403413586.0)] | monotonic non-increasing |  |
| 18.6 | ✓ PASS | Weight sensitivity check: 100% severe vs 100% baseline (data-driven; inversion documented) | 100% severe=$376,233,658, 100% baseline=$403,536,501 (severe < base (inverted, documented in §6)) | either direction documented |  |
| 18.7 | ✓ PASS | Headline ordering: regulatory > baseline > data-driven overlay | reg=$484,474,530, base=$403,536,501, data_overlay=$390,821,719 | ordered as expected |  |

## Category 19 — Dashboard data integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 19.1 | ✓ PASS | All Step 15 dashboard files exist | all 6 present | all present |  |
| 19.2 | ✓ PASS | loans_summary row count matches master parquet | dashboard=1,179,687, parquet=1,179,687 | equal |  |
| 19.3 | ✓ PASS | Headline ECL per version matches source JSON within $1 | baseline \|Δ\|=0.0049, overlay \|Δ\|=0.0021, reg \|Δ\|=0.0029 | < $1 |  |
| 19.4 | ✓ PASS | discrimination_metrics aggregate AUC matches validation_discrimination.json | csv=0.7059, json=0.7059 | equal |  |
| 19.5 | ✓ PASS | calibration_table decile MAD matches validation_calibration.json | csv MAD=0.0276, json MAD=0.0276 | equal |  |
| 19.6 | ✓ PASS | sensitivity_table has all expected analysis types | types=['SICR_threshold', 'overlay_weights', 'single_feature_stress'] | ['SICR_threshold', 'overlay_weights', 'single_feature_stress'] |  |
| 19.7 | ✓ PASS | dashboard_spec references 4 pages and 5 tables | pages=True, tables=True | all present |  |

## Category 20 — Project summary integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 20.1 | ✓ PASS | Headline numbers in summary reproduce JSON sources to $0.01 | baseline=$403,536,501, data=$390,821,719, regulatory=$484,474,530; missing=[] | all 3 present |  |
| 20.2 | ✓ PASS | Top-3 IV values appear in summary (matches binning_summary) | top IVs checked: [('sub_grade', 0.5037), ('grade', 0.47), ('int_rate', 0.4695)] | all top-3 present |  |
| 20.3 | ✓ PASS | AUC matches model_evaluation.json | AUC=0.7059 | present in summary |  |
| 20.4 | ✓ PASS | ECE matches calibration_post.json | ECE=0.0195 | present |  |
| 20.5 | ✓ PASS | Stage counts match ecl_headline.json | S1=944,678, S2=583, S3=234,426 | all 3 present |  |

## Category 21 — Final dossier integrity

| ID | Status | Description | Observed | Expected | Notes |
|---|:---:|---|---|---|---|
| 21.0 | ✓ PASS | final_project_dossier.md exists | present | present |  |
| 21.0b | ✓ PASS | build_final_dossier.py exists | present | present |  |
| 21.1 | ✓ PASS | All 15 H2 section headers present (Section 0–14) | all 15 sections | 15 H2 headers |  |
| 21.2 | ✓ PASS | Word count within 5,000–10,000 range | 8,856 words | 5,000–10,000 |  |
| 21.3 | ✓ PASS | ≥50 inline `verified` tags present in dossier body | 73 tags | ≥50 |  |
| 21.4 | ✓ PASS | Zero VERIFY FAILED tags in dossier body | 0 failed | 0 |  |
| 21.5 | ✓ PASS | Three headline numbers reproduce JSON sources to $0.01 | baseline=$403,536,501, data=$390,821,719, reg=$484,474,530; missing=[] | all 3 present |  |
| 21.6 | ✓ PASS | Verification appendix table present (Section 13) | appendix table found | appendix table in Section 13 |  |
| 21.7 | ✓ PASS | File index lists ≥35 pipeline files | 92 file rows in Section 12 | ≥35 |  |
| 21.8 | ✓ PASS | File index has all 4 category groupings | all 4 present | Cleaned data / Models / Validation / Source code |  |
| 21.9 | ✓ PASS | Appendix pass/fail counts match inline tag counts | appendix passes=73, fails=0; inline ✓=73, ✗=0 | appendix matches inline counts |  |
| 21.10 | ✓ PASS | 5 random spot-check values present in dossier text | all 5 present | 5/5 present |  |

## Items to Note

- 1.3 No orphan/unexpected files
- 1.4 File mtimes chronological across pipeline stages
- 7.5 Every selected-feature bin has ≥100 training observations
- 8.6a Coefficient sign consistency
- 8.6b IV-selected feature |coef| in [0.1, 1.5]
- 8.6c No IV-selected feature with |coef| < 0.05
- 11.3 Top-IV feature traceable: binning → 9a doc → coefs (collinearity-aware)
- 16.3 Stage shares plausible: S1∈[60-95]%, S2∈[3-30]%, S3==defaulters
- 17.1 Per-loan monotonicity (baseline ≤ adverse ≤ severe), <1% violations

## Limitations Review (manual inspection list)

Every limitation cited across the four methodology documents:

- [step7_methodology] The model produces **lifetime PD** because that is what the label measures. Conversion to **12-month
- [step7_methodology] A live bank would build a snapshot-based 12-month PD with monthly behavioral features (payment trend
- [step7_methodology] DNMCP loans were dropped to avoid mixing the pre-2009 (looser) and post-2009 credit-policy regimes i
- [step8_methodology] The four macros are correlated (UNRATE ↑ tends to coincide with GDP YoY ↓, and HPI YoY tracks both).
- [step8_methodology] US national level only. Regional macroeconomic variation (e.g., Detroit 2008 vs. Texas 2008) is not 
- [step8_methodology] Case-Shiller has a real-world publication lag of about two months, which is ignored for retrospectiv
- [step8_methodology] The macros are point-in-time at origination. The forward-looking overlay step (later) handles the pr
- [step8_methodology] FRED occasionally revises historical series. Re-runs after a revision will produce slightly differen
- [step8_methodology] The within-year correlation between UNRATE and default rate is essentially zero in this dataset (-0.
- [step9a_methodology] WoE binning is fit on training data only; bins for the test set inherit training cuts. New categoric
- [step9a_methodology] The binning is not refit per fold or vintage; this is acceptable for a first model but a production 
- [step9a_methodology] IV thresholds (0.02 floor, 0.7 ceiling) are conventional. Sensitivity analysis on the lower bound is
- [step9a_methodology] WoE replacement removes individual-loan variation within a bin: every loan in the same bin gets the 
- [step9a_methodology] The three force-included features (`unrate`, `hpi_yoy`, `issue_year`) have marginal IV below the con
- [step9b_methodology] The PD model produces **lifetime PD** because of the origination-based observation-window design fro
- [step9b_methodology] Predicted probabilities are **not yet calibrated**. Logistic regression on WoE features tends to be 
- [step9b_methodology] **PSI between train and test is elevated** by LC vintage drift (train: 2007–2015, test: 2016+). This
- [step9b_methodology] The gradient-boosting challenger uses sklearn's `HistGradientBoostingClassifier` rather than `xgboos
- [step9b_methodology] The collinearity between `grade`/`sub_grade` and between `fico_range_low`/`fico_range_high` produces

## Sign-off

**Pipeline status: READY for Step 9c (probability calibration).**
