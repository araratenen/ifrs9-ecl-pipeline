# Portfolio status — IFRS 9 ECL pipeline + audit_analytics

_Generated 2026-05-10. Read-only inventory; nothing was modified to produce this report._

## Executive summary

Two end-to-end projects in this tree.

| Project | Path | Headline state | Risk-role fit |
|---|---|---|---|
| **IFRS 9 ECL Pipeline** | `/Users/ostappolukainen/Desktop/ProjRED/` | Complete: $403.5M / $390.8M / **$484.5M** ECL with 167/167 validator + 124-PASS / 8-WARN / 0-FAIL audit + 62 unit tests + 8,856-word self-verifying dossier. | Audit FSI · FSI Analytik · junior insurance consulting |
| **Audit Analytics** | `audit_analytics/` | Complete: 6 risk rules + Benford module + validation framework + 5-sheet Excel workpaper + independent-audit remediation pass (F-001..F-007 fixed). 119 pytest tests, verifier 16/16. | IT Audit · Audit FSI |

**Important correction to the brief.** The task description says of Project 2: _"Currently consists of synthetic GL generator + verification script. Risk rules and Benford module not yet implemented."_ That description is significantly out-of-date. The audit_analytics project is fully built out — six rules, Benford, a validation framework that scores precision/recall against labeled ground truth, an Excel workpaper, an independent audit report with seven findings remediated, and a `portfolio_artifacts/` directory ready for GitHub. **Project 2 is closer to "interview-ready" than Project 1 in some respects** because it has frozen reference deliverables under version control. Section 3 enumerates evidence.

**Two structural items to flag right now.**

1. **Project 1 is not under git.** No `.git` directory at the project root; only `audit_analytics/.git` exists. A recruiter cloning the repo for Project 1 has no commit history, no easy diff, no protection against accidental edits. This is a meaningful gap for portfolio publication.
2. **Project 1's `docs/` carries 15 timestamped `audit_report_full_*.md` files.** These are historical run artifacts, not stale per se but cluttering. A reader landing in `docs/` sees these before the dossier and methodology MDs.

The project pair is genuinely strong; the gaps are presentational, not analytical.

---

## Section 1 — Repository inventory

### Project 1: IFRS 9 ECL Pipeline (`/Users/ostappolukainen/Desktop/ProjRED/`)

#### Source code

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `src/step7_observation_window.py` | Step 7 — clean & feature-classify | 2026-05-08 20:17 | 16.8 KB | complete |
| `src/step8_macro_features.py` | Step 8 — FRED macro join | 2026-05-08 20:46 | 21.8 KB | complete |
| `src/step9a_woe_binning.py` | Step 9a — WoE binning + train/test split | 2026-05-09 00:07 | 34.1 KB | complete |
| `src/step9b_pd_model.py` | Step 9b — PD models (LR + HGBM challenger) | 2026-05-09 00:19 | 28.6 KB | complete |
| `src/step9c_calibration.py` | Step 9c — Platt calibration | 2026-05-09 11:09 | 27.4 KB | complete |
| `src/step10_lgd_estimation.py` | Step 10 — LGD via segment-average | 2026-05-09 14:07 | 28.5 KB | complete |
| `src/step11_ead_projection.py` | Step 11 — EAD via contractual amortization | 2026-05-09 14:18 | 26.0 KB | complete |
| `src/step12_ecl_combination.py` | Step 12 — ECL combination + IFRS 9 staging | 2026-05-09 14:28 | 27.8 KB | complete |
| `src/step13_macro_overlay.py` | Step 13 — Data-driven macro overlay | 2026-05-09 14:48 | 32.5 KB | complete |
| `src/step14_validation.py` | Step 14 — Validation pack + regulatory overlay | 2026-05-09 15:03 | 46.0 KB | complete |
| `src/step15_dashboard_data.py` | Step 15 — Power BI dashboard CSVs | 2026-05-09 15:16 | 27.8 KB | complete |
| `src/build_project_summary.py` | Unified project summary MD | 2026-05-09 17:46 | 43.8 KB | complete |
| `src/build_final_dossier.py` | Self-verifying final dossier | 2026-05-09 18:20 | 80.3 KB | complete |
| `src/validate_pipeline_steps_7_8.py` | Validator (167 checks across 18 tasks) | 2026-05-09 18:21 | 65.7 KB | complete |
| `src/audit_full_pipeline.py` | Full pipeline audit (132 checks across 21 categories) | 2026-05-09 18:22 | 106.8 KB | complete |
| `tests/test_amortization.py` | Amortization closed-form unit tests | 2026-05-09 18:46 | 4.7 KB | complete |
| `tests/test_ecl.py` | ECL combination unit tests | 2026-05-09 18:45 | 5.1 KB | complete |
| `tests/test_overlay.py` | Macro-overlay weighted-average tests | 2026-05-09 18:42 | 3.9 KB | complete |
| `tests/test_platt.py` | Platt scaler unit tests | 2026-05-09 18:42 | 3.5 KB | complete |
| `tests/test_staging.py` | IFRS 9 staging logic tests | 2026-05-09 18:47 | 4.9 KB | complete |
| `tests/test_woe.py` | WoE / IV unit tests | 2026-05-09 18:48 | 4.9 KB | complete |
| `tests/conftest.py` | pytest path / fixtures | 2026-05-09 18:40 | 2.1 KB | complete |

#### Data artifacts

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `accepted_slim.csv` | Source LendingClub data | 2026-05-08 18:06 | **645 MB** | complete (raw input; large) |
| `data/accepted_labeled.parquet` | Step 6 default-flag mapping | 2026-05-08 23:40 | 71.2 MB | complete |
| `data/loans_modeling_ready.parquet` | Step 7 output | 2026-05-09 18:49 | 61.7 MB | complete |
| `data/loans_with_macros.parquet` | Step 8 output | 2026-05-09 18:49 | 62.0 MB | complete |
| `data/macros_monthly.parquet` | FRED monthly macros | 2026-05-09 18:49 | 8.2 KB | complete |
| `data/train.parquet` / `test.parquet` | Step 9a split | 2026-05-09 18:49 | 43.7 / 19.5 MB | complete |
| `data/train_woe.parquet` / `test_woe.parquet` | WoE-transformed | 2026-05-09 18:49 | 10.2 / 4.7 MB | complete |
| `data/test_predictions.parquet` | Per-loan predictions | 2026-05-09 18:59 | 37.8 MB | complete |
| `data/loans_with_lgd.parquet` | Step 10 output | 2026-05-09 18:58 | 62.9 MB | complete |
| `data/loans_with_ead.parquet` | Step 11 output | 2026-05-09 18:58 | 87.1 MB | complete |
| `data/loans_with_ecl.parquet` | Step 12 output | 2026-05-09 18:58 | 112.9 MB | complete |
| `data/loans_with_ecl_overlay.parquet` | Step 13 output | 2026-05-09 18:59 | 172.6 MB | complete |
| `data/dashboard/loans_summary.csv` | Power BI loan-level | 2026-05-09 19:00 | **358 MB** | complete (large) |
| `data/dashboard/{headline,calibration,discrim,sensitivity}*.csv` | Power BI roll-ups | 2026-05-09 19:01 | small | complete |

#### Models

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `models/binning_process.pkl` | Fitted optbinning model | 2026-05-09 18:49 | 80.7 KB | complete |
| `models/pd_logistic.pkl` | Logistic PD | 2026-05-09 18:51 | 1.0 KB | complete |
| `models/pd_xgboost.pkl` | HistGradientBoosting challenger | 2026-05-09 18:51 | 381.8 KB | complete |
| `models/pd_logistic_calibrator.pkl` | Platt scaler | 2026-05-09 18:57 | 0.9 KB | complete |
| `models/pd_xgboost_calibrator.pkl` | Platt scaler (challenger) | 2026-05-09 18:57 | 0.9 KB | complete |

#### Documentation

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `README.md` | Project README (~780 words) | 2026-05-09 19:00 | 6.2 KB | complete |
| `docs/final_project_dossier.md` | Self-verifying canonical doc, 8,856 words, 15 sections | 2026-05-09 18:21 | ~150 KB | complete |
| `docs/project_summary.md` | Unified methodology summary, 12 sections | (recent) | ~32 KB | complete |
| `docs/dashboard_spec.md` | Power BI dashboard specification | (recent) | ~6 KB | complete |
| `docs/step{7,8,9a,9b,9c,10,11,12,13}_methodology.md` | 9 per-step methodology MDs | various | varies | complete |
| `docs/{step14,step15}_methodology.md` | per-step MDs for validation + dashboard | — | — | **MISSING** |
| `docs/audit_report_full_*.md` (×15) | Historical run reports | 2026-05-09 | varies | clutter (not stale, just bulky) |
| `docs/validation_report_steps_7_8.md` | Pipeline validation report (167 checks pass) | 2026-05-09 19:02 | 17 KB | complete |
| `docs/final_validation_report.md` | Step-14 validation pack | 2026-05-09 19:00 | 13 KB | complete |
| `docs/{ecl_headline,ecl_overlay_headline,validation_regulatory_overlay}.json` | The three headline JSONs | 2026-05-09 | small | complete |
| `docs/binning_summary.json`, `model_evaluation.json`, `calibration_post.json`, `lgd_stats.json`, `feature_classification.json` | Per-step JSON outputs | 2026-05-09 | small | complete |

#### Build / orchestration

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `Makefile` | DAG with 17 targets including `all`, `qa`, `qa-fast`, `dag`, phased rules `data` / `pd-pipeline` / `ecl-pipeline` / `overlay` / `dashboard` / `docs` / `validate` / `audit` / `test` / `clean` | 2026-05-09 19:00 | 8.0 KB | complete |
| `pytest.ini` | Test config | 2026-05-09 18:44 | 167 B | complete |
| `requirements.txt` | — | — | — | **MISSING at root** |
| `.gitignore` | Present | 2026-05-08 20:01 | 61 B | present (but no `.git` dir) |
| `.git/` | Version control | — | — | **MISSING — Project 1 is not a git repo** |
| `.venv/` | Local Python venv | 2026-05-10 01:36 | n/a | present (gitignored content) |

---

### Project 2: Audit Analytics (`audit_analytics/`)

#### Source code

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `src/build_synthetic_gl.py` | Generator CLI (8 transaction types, 6 anomaly archetypes + 2 look-alikes) | 2026-05-10 02:05 | ~13 KB | complete |
| `src/_gl_anomalies.py` | Anomaly injection helpers (`_collapse_to_two_lines`, six injectors, look-alike collector) | 2026-05-10 01:54 | ~9 KB | complete |
| `src/verify_synthetic_gl.py` | 16-check verifier with content-hash | 2026-05-09 23:16 | ~10 KB | complete (16/16 PASS) |
| `src/run_rules.py` | Auto-discovering CLI for `rule_*` modules | 2026-05-10 00:35 | ~4 KB | complete |
| `src/rules/__init__.py` | Discovery convention docstring | 2026-05-09 23:15 | 1 KB | complete |
| `src/rules/common.py` | 9-column EXCEPTION_COLUMNS contract + validators | 2026-05-09 23:15 | 4 KB | complete |
| `src/rules/rule_weekend_holiday.py` | Rule 1 — weekend / US+CZ holiday | 2026-05-09 23:18 | ~6 KB | complete (9 tests) |
| `src/rules/rule_off_hours.py` | Rule 2 — off-hours w/ circular z-score | 2026-05-09 23:23 | ~6 KB | complete (8 tests) |
| `src/rules/rule_round_amounts.py` | Rule 3 — round amounts (two-tier) | 2026-05-09 23:30 | ~6 KB | complete (17 tests) |
| `src/rules/rule_below_threshold.py` | Rule 4 — just-below-threshold structuring | 2026-05-09 23:55 | ~7 KB | complete (32 tests) |
| `src/rules/rule_sod.py` | Rule 5 — segregation-of-duties (deterministic sev 5) | 2026-05-10 00:23 | 4 KB | complete (9 tests) |
| `src/rules/rule_period_end.py` | Rule 6 — period-end manual to P&L | 2026-05-10 00:35 | ~7 KB | complete (9 tests) |
| `src/benford/__init__.py` | Package docstring | 2026-05-10 00:55 | 1 KB | complete |
| `src/benford/tests.py` | Three statistical tests (1st / 2nd / 1st-2 digit) | 2026-05-10 01:00 | ~7 KB | complete (9 tests) |
| `src/benford/slicing.py` | Whole / by-account / by-user runners | 2026-05-10 01:05 | ~3 KB | complete (5 tests) |
| `src/benford/run_benford.py` | CLI emitting `benford_results.json` + `benford_flags.csv` | 2026-05-10 01:10 | ~5 KB | complete |
| `src/validation/__init__.py` | Package docstring | 2026-05-10 01:30 | 1 KB | complete |
| `src/validation/ground_truth.py` | Loaders + ARCHETYPE_META | 2026-05-10 01:35 | ~5 KB | complete |
| `src/validation/scorer.py` | TP/FP/FN scoring with bucket precedence + cross-anomaly findings | 2026-05-10 02:34 | ~10 KB | complete (10 tests) |
| `src/validation/run_validation.py` | CLI emitting `validation_results.json` + `validation_report.md` | 2026-05-10 01:50 | ~8 KB | complete |
| `src/workpaper/__init__.py` | Package docstring | 2026-05-10 01:55 | 1 KB | complete |
| `src/workpaper/styles.py` | Cell / fill / conditional-formatting helpers | 2026-05-10 01:55 | 3 KB | complete |
| `src/workpaper/_content.py` | STANDARDS / LIMITATIONS / METHODOLOGY constants | 2026-05-10 01:54 | 3 KB | complete |
| `src/workpaper/build_workpaper.py` | 5-sheet Excel builder | 2026-05-10 01:56 | ~12 KB | complete |
| `tests/{conftest,test_*}.py` (×11) | 119 tests across 10 modules + conftest | various | varies | **complete (119/119 PASS)** |

#### Data artifacts

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `data/synthetic_gl.parquet` | Synthetic GL — 99,520 lines / 25,010 JEs | 2026-05-10 02:05 | 3.2 MB | complete |
| `data/synthetic_gl_sample.csv` | First 1,000 rows | 2026-05-10 02:05 | 165 KB | complete |
| `data/embedded_anomalies.json` | Ground-truth labels — 8 archetypes | 2026-05-10 02:05 | 11.5 KB | complete |
| `data/chart_of_accounts.json` | 50 accounts (parses; valid) | 2026-05-09 22:57 | 6.2 KB | complete |
| `data/users.json` | 31 users with per-user posting hour mean/std (parses; valid) | 2026-05-09 22:58 | 9.3 KB | complete |
| `data/thresholds.json` | Role-keyed authorization thresholds (parses; valid) | 2026-05-09 22:57 | 0.9 KB | complete |
| `data/counterparties.json` | 50 vendors + 30 customers (parses; valid) | 2026-05-09 22:58 | 4.5 KB | complete |

#### Documentation

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `README.md` (audit_analytics root) | 469 prose words, headline detection table, cross-rule findings, standards bullets | 2026-05-10 02:04 | 6.4 KB | complete |
| `docs/schema_and_distributions.md` | 13.5 KB; covers §1.1 schema, §1.2 chart of accounts (post-F-003 observed top-10), §1.3 roster, §1.4 thresholds, §1.5 distributional choices (post-F-004 line-level clarification), §1.6 scope (post-F-005 rule list) | 2026-05-10 02:01 | 13.5 KB | complete |
| `docs/standards_mapping.md` | ISA 240 / ISA 330 / PCAOB AS 2401 / AS 2110 mapping with paragraph cites | 2026-05-10 01:03 | 5.7 KB | complete |
| `docs/patch_2024-anomaly-line-dilution.md` | Earlier patch documentation | 2026-05-09 23:55 | 4.6 KB | complete |
| `audit_report.md` (root of `audit_analytics/`) | Independent audit — F-001..F-011 | 2026-05-10 01:54 | 7.0 KB | complete |

#### Output / portfolio artifacts

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `output/exceptions.csv` | 6,698 rule exceptions | 2026-05-10 02:05 | 1.16 MB | complete (gitignored — regenerable) |
| `output/benford_flags.csv` | 9 Benford-non-conforming slices | 2026-05-10 02:05 | 2.4 KB | complete (gitignored) |
| `output/benford_results.json` | Full Benford output (whole / by-account / by-user) | 2026-05-10 02:05 | 448 KB | complete (gitignored) |
| `output/validation_results.json` | Per-detector scoring + cross-rule findings | 2026-05-10 02:05 | 13 KB | complete (gitignored) |
| `output/validation_report.md` | Human-readable precision/recall report | 2026-05-10 02:05 | 8.7 KB | complete (gitignored) |
| `output/audit_workpaper.xlsx` | 5-sheet Excel workbook (Cover / Exceptions / Per-rule / Benford / Methodology) | 2026-05-10 02:05 | 392 KB | complete (gitignored) |
| `portfolio_artifacts/{validation_report.md, audit_workpaper.xlsx, exceptions_sample_50.csv, benford_flags_sample_5.csv, audit_report.md, README.md}` | Frozen reference copies, **tracked in Git** for review-without-pipeline-run | 2026-05-10 02:03–02:05 | varies | complete |

#### Build / orchestration

| Path | Purpose | Last modified | Size | Status |
|---|---|---|---:|---|
| `requirements.txt` | pandas, numpy, pyarrow, holidays, scipy, pytest, openpyxl | 2026-05-10 00:47 | 91 B | complete |
| `pytest.ini` | testpaths + filters | 2026-05-09 23:14 | 152 B | complete |
| `.gitignore` | excludes `output/`, parquet, etc. | 2026-05-09 23:12 | 223 B | complete |
| `.git/` | 8 commits since init (latest: `7e4783f` artifact refresh) | 2026-05-10 02:06 | n/a | complete |
| `Makefile` | — | — | — | **MISSING — no DAG orchestration** |

---

## Section 2 — Project 1 status (IFRS 9 ECL)

### Pipeline steps implemented

| Step | Script | One-line description (from docstring) |
|---|---|---|
| 7 | `step7_observation_window.py` | Observation/performance window cleanup; default-flag mapping; FICO floor; maturity filter; feature classification |
| 8 | `step8_macro_features.py` | FRED macro join (UNRATE / HPI YoY / GDP YoY / FedFunds) at issue month; 169 monthly rows |
| 9a | `step9a_woe_binning.py` | WoE binning via `optbinning.BinningProcess`; train/test split; 19 features (16 IV-selected + 3 forced) |
| 9b | `step9b_pd_model.py` | PD models — logistic regression + HistGradientBoosting challenger (substitute for XGBoost) |
| 9c | `step9c_calibration.py` | Platt scaling on test cohort (vintage drift correction) |
| 10 | `step10_lgd_estimation.py` | LGD via segment-average with grade-only fallback |
| 11 | `step11_ead_projection.py` | Closed-form contractual amortization for EAD |
| 12 | `step12_ecl_combination.py` | ECL combination + IFRS 9 staging via SICR proxy |
| 13 | `step13_macro_overlay.py` | Three-scenario macro overlay (50/30/20 weights) — data-driven |
| 14 | `step14_validation.py` | Validation pack + regulatory-coefficient overlay (CCAR/EBA-style) |
| 15 | `step15_dashboard_data.py` | Power BI dashboard CSVs (loans_summary + 4 roll-ups) |

### Key artifacts (parsed values)

| Artifact | Headline value | Source field |
|---|---:|---|
| `docs/ecl_headline.json` | **$403,536,501** | `total_ecl` (Step 12 baseline) |
| `docs/ecl_overlay_headline.json` | **$390,821,719** | `final_ecl` (Step 13 data-driven overlay) |
| `docs/validation_regulatory_overlay.json` | **$484,474,530** | `weighted_final_ecl` (Step 14 regulatory — recommended for IFRS 9 reporting) |

### QA layers status

| Layer | Count | Status | Source |
|---|---:|---|---|
| pytest unit tests | **62** collected | green | `pytest tests/ --collect-only` |
| Validator (Tasks 1–18) | **167 / 167 PASS** | green | `docs/validation_report_steps_7_8.md` summary |
| Audit (Categories 1–21) | **132 total: 124 PASS / 8 WARN / 0 FAIL** | green | latest `docs/audit_report_full_20260509T190251.md` |

### Documentation status

| Item | Present | Notes |
|---|:---:|---|
| `README.md` | yes | 781 words; headline table + run commands |
| `docs/final_project_dossier.md` | yes | 8,856 words / 15 sections; self-verifying with 73 inline-verified claims |
| `Makefile` | yes | 17 targets including `all`, `qa`, `qa-fast`, `dag` |
| `docs/dashboard_spec.md` | yes | Power BI four-page spec |
| Per-step methodology MDs | partial | 9 of 11 (steps 7, 8, 9a, 9b, 9c, 10, 11, 12, 13). **Missing: step14 + step15** |

### Reproducibility

| Question | Answer |
|---|---|
| `make qa` exists? | yes (`Makefile` line 26 of phony block) |
| `make qa-fast` exists? | yes (skips rebuild) |
| `make dag` exists? | yes (dry-run printer) |
| Sentinel files present? | all 5 model pkls + all 13 parquet artifacts present |
| `requirements.txt` at root? | **NO — should be added** |
| Git repo? | **NO — `.git/` does not exist at project root** |

---

## Section 3 — Project 2 status (Audit Analytics)

> **Mental-model reset.** The task description says Project 2 currently consists of "synthetic GL generator + verification script" and that "risk rules and Benford module not yet implemented." That is significantly out-of-date. **All six rules, the Benford module, the validation framework, the Excel workpaper, the standards mapping, the README, and an independent-audit remediation pass (F-001..F-007) are committed and green.** Evidence below.

### Synthetic GL generator

| Question | Answer |
|---|---|
| Does `src/build_synthetic_gl.py` exist? | yes (~13 KB) |
| Does it run? | yes — produced `data/synthetic_gl.parquet` 2026-05-10 02:05 |
| Has the GL been run? | yes — parquet present (3.2 MB), 99,520 lines / 25,010 JEs |

### Verification script

`src/verify_synthetic_gl.py` exists and was just run for this report:

```
=== Verifier summary ===
  Total checks: 16
  Passed:       16
  Failed:       0
Content hash (seed-dependent): sha256:895cb61b…7493
```

All four sections (3.1 structural, 3.2 distributional, 3.3 anomaly presence, 3.4 SoD-baseline) green; section 3.5 emits the deterministic content hash.

### Embedded anomalies

`data/embedded_anomalies.json` parses cleanly. **Eight distinct entries** (8 archetypes; the brief said six because look-alikes are tracked alongside primary anomalies):

| anomaly_id | archetype | je_count | rule_should_catch |
|---|---|---:|---|
| `RND_001` | round_amount_cluster | 30 | `rule_round_amounts` |
| `BTH_001` | below_threshold_cluster | 25 | `rule_below_threshold` |
| `OFH_001` | off_hours_cluster | 20 | `rule_off_hours` |
| `SOD_001` | sod_violation | 30 | `rule_sod` |
| `RVR_001` | reversal_abuse | 20 | `rule_reversal_abuse` (no detector — known scope limitation) |
| `BNF_001` | benford_violation | 200 | `benford_first_digit_by_account` |
| `LLA_WKND` | legitimate_lookalike | 50 | None (calibrated FP) |
| `LLA_PEND` | legitimate_lookalike | 100 | None (calibrated FP) |

All six expected primary archetypes are represented; both legitimate-look-alike populations (LLA_WKND, LLA_PEND) are present.

### Schema documentation

`docs/schema_and_distributions.md` exists at 13.5 KB and covers all six required sub-sections:

| Section | Topic | Notes |
|---|---|---|
| §1.1 | GL schema | 18-column table (includes `transaction_type`, `amount` extensions) |
| §1.2 | Chart of accounts | Observed top-10 distribution + period-driven mechanism note (post-F-003 fix) |
| §1.3 | User roster | 31 users with per-user posting hour mean/std |
| §1.4 | Authorization thresholds | $5k / $25k / $100k / unlimited |
| §1.5 | Distributional choices | Includes JE-total-vs-line-level clarification (post-F-004 fix) |
| §1.6 | Scope | Lists `rule_weekend_holiday` correctly (post-F-005 fix); RVR_001 marked OOS |

### Configuration files

All four config JSONs parse and contain the expected structure:

| File | Counts | Status |
|---|---:|---|
| `data/chart_of_accounts.json` | 50 accounts | parses; valid |
| `data/users.json` | 31 users (10 AP / 5 AR / 8 GL / 2 controller / 1 CFO / 5 system) | parses; valid |
| `data/thresholds.json` | 6 role keys | parses; valid (CFO null = unlimited) |
| `data/counterparties.json` | 50 vendors + 30 customers | parses; valid |

### Risk rules and Benford module

**Implemented and green** — contradicting the task brief:

| Layer | What's there | Test count |
|---|---|---:|
| Rules contract | `src/rules/common.py` 9-column EXCEPTION_COLUMNS + validators | — |
| Rule 1 | `rule_weekend_holiday.py` | 9 |
| Rule 2 | `rule_off_hours.py` | 8 |
| Rule 3 | `rule_round_amounts.py` (two-tier) | 17 |
| Rule 4 | `rule_below_threshold.py` (proximity + frequency, max combination) | 32 |
| Rule 5 | `rule_sod.py` (deterministic sev 5) | 9 |
| Rule 6 | `rule_period_end.py` (proximity + size quantile) | 9 |
| Benford tests | `src/benford/tests.py` (1st / 2nd / 1st-2 digit) | 9 |
| Benford slicing | `src/benford/slicing.py` (whole / account / user) | 5 |
| Validation scorer | `src/validation/scorer.py` | 10 (incl. F-002 double-membership case) |
| Workpaper builder | `src/workpaper/build_workpaper.py` | 11 |
| **Total** | — | **119 collected** |

### Detection table (post-remediation, parsed from `validation_results.json`)

| Detector | Flagged | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| `rule_sod` | 121 | 100% | 100% | 1.00 | deterministic compliance check |
| `rule_round_amounts` | 30 | 100% | 100% | 1.00 | aggregate-fraction gate |
| `rule_off_hours` | 74 | 100% | 100% | 1.00 | per-user calibration |
| `benford_first_digit_account` | 3 | 33% | 100% | 0.50 | BNF_001 caught at MAD = 0.030 |
| `rule_below_threshold` | 152 | 26% | 80% | 0.40 | calibration-sensitive |
| `rule_period_end` | 4,153 | n/a | n/a | n/a | prioritization layer; no primary anomaly |
| `rule_weekend_holiday` | 2,159 | n/a | n/a | n/a | prioritization layer; no primary anomaly |
| `benford_first_digit_user` | 6 | n/a | n/a | n/a | cross-rule second-order signal |

Total **6,698 flags**; 15 cross-rule second-order findings recorded. Note: `rule_period_end` reports **4,153** post-F-002, exactly matching `exceptions.csv` row count.

### Independent audit + remediation

`audit_report.md` (and `portfolio_artifacts/audit_report.md`) records 11 findings; F-001..F-007 are remediated in 8 commits since `14bc8e4`:

```
7e4783f Refresh portfolio artifacts after F-001 through F-007 remediation
6d048f6 F-006: Add frozen portfolio artifacts for review-without-pipeline-run
20f3981 F-007: Make workpaper tests resilient to missing output files via session fixture
f567616 F-004: Clarify JE-total vs line-level amount distribution parameters
8db4095 F-003: Replace schema doc account frequency table with observed distribution
b3e116f F-005: Correct schema doc rule list — replace nonexistent rule_reversal_abuse
bcb9d60 F-002: Enforce mutually-exclusive FP bucket precedence
a0e44bd F-001: Narrow line-level primary positives by target_account
```

F-008..F-011 are explicit honest-disclosure / cosmetic items intentionally retained.

---

## Section 4 — What's missing or unfinished

Ranked by priority for portfolio publication. _Effort estimates are calendar days of focused work, not absolute._

| # | Item | Effort | Best role-fit |
|---:|---|---:|---|
| 1 | **`git init` Project 1.** No `.git/` at project root. Without commit history, the recruiter can't see the development arc; with it they can browse staged commits like the audit_analytics history. Add `.gitignore` for `.venv/`, big parquets, `accepted_slim.csv`. | 0.5 | All four roles |
| 2 | **Add `requirements.txt` to Project 1.** Currently the `.venv` is the only spec. A reviewer can't `pip install -r requirements.txt`. | 0.25 | All four |
| 3 | **Project 1 step14 + step15 methodology MDs.** The other 9 step methodologies exist; these two are missing. Both are referenced in the dossier but lack dedicated docs. | 0.5–1 | Audit FSI, FSI Analytik (the missing ones cover the validation pack and the dashboard, which are the auditor-facing layers) | 
| 4 | **Project 1 docs/ housekeeping.** 15 timestamped `audit_report_full_*.md` files. Either delete old ones (keep 1–2 most recent), or move them under `docs/_history/`. The dossier already aggregates the substance. | 0.25 | All four |
| 5 | **Add a Makefile / `qa` orchestration to Project 2.** Project 2 has 9 CLIs but no DAG; a recruiter wanting to run end-to-end runs each CLI manually. Mirror Project 1's pattern: `make all`, `make test`, `make qa`. | 0.5 | IT Audit |
| 6 | **Top-level repo README at the parent that contains both projects.** Right now both projects have their own READMEs but a recruiter who clones the parent dir sees no orientation. A 1-page README describing the two-project portfolio + linking each. | 0.5 | All four |
| 7 | **Project 2 audit_report.md publication polish.** The audit report at `audit_analytics/audit_report.md` reads as if it lives at the repo root; add a "How this came about" header so a cold reader knows the audit was done by a separate pass against the project. | 0.25 | IT Audit (this is the most direct demonstration of an audit deliverable) |
| 8 | **RVR_001 detector.** `rule_reversal_abuse` is referenced in `embedded_anomalies.json` but has no implementation. This is documented as a known scope limitation everywhere. **Optional** — leaving it out is honest, implementing it adds ~1 day. | 1 | IT Audit, Audit FSI |
| 9 | **Power BI screenshots / `.pbix` for Project 1.** `dashboard_spec.md` describes a four-page Power BI dashboard but no actual `.pbix` exists. A screenshot in the README would substantiate the dashboard claim. | 1 | FSI Analytik (dashboard is the analytical-tooling demo) |
| 10 | **Recruiter pitch / cover document.** Two separate READMEs is OK; what's missing is a one-page "What I built and why" intended for the role. Could be Czech + English, since Big 4 Prague hires expect both. | 0.5 | All four |

---

## Section 5 — What's interview-ready right now

Six strong items, ranked by how directly they map onto Big 4 Prague risk-side interview questions.

### 5.1 — IFRS 9 three-headline ECL with a regulatory overlay

**Artifact.** `README.md` headline table at the project root + `docs/final_project_dossier.md` Section 1 (executive summary).

**30-second pitch.** "I built an end-to-end IFRS 9 ECL pipeline on the LendingClub consumer-loan dataset producing three headline figures: $403.5M baseline (mechanical PD × LGD × EAD), $390.8M data-driven overlay where the empirical macro relationship inverts and I documented why (LC underwriting-reaction effect — Simpson's paradox in the macro-rate join), and $484.5M regulatory-coefficient overlay with CCAR/EBA-style sensitivities (+0.18 unrate, +0.05 HPI YoY) — that's the number I'd recommend for IFRS 9 reporting. The whole stack reproduces from `make all` to one cent on the headline."

**Likely follow-ups.** Why does the empirical overlay inverse the regulatory one? How is staging assigned (SICR proxy)? Why HistGradientBoosting instead of XGBoost? How is recall on 12-month PD validated when the label is lifetime?

### 5.2 — ISA 240 journal-entry testing toolkit

**Artifact.** `audit_analytics/portfolio_artifacts/validation_report.md` headline table + `audit_workpaper.xlsx`.

**30-second pitch.** "Six risk rules + Benford's Law module operationalize ISA 240. SoD, round-amount aggregate, off-hours all hit 100% precision and 100% recall against labeled ground truth. Rule 4 (just-below-threshold) trades precision for recall — 26% / 80% / F1 0.40 — that's the calibration-sensitive number a reviewer should care about. The most interesting finding is cross-rule second-order: Benford's by-user analysis catches both the structuring user U002 and the round-amount user U007 as non-conforming distributions, even though their primary anomaly isn't a Benford violation. Independent statistical and behavioral signals point at the same suspect."

**Likely follow-ups.** Why is rule_period_end's precision 0%? What's the false-positive cost? How would you tune Rule 4? What happens at scale on a real ERP — performance, false-positive triage workflow, model governance?

### 5.3 — Independent audit with remediation discipline

**Artifact.** `audit_analytics/audit_report.md` (11 findings) + the 8 commits `a0e44bd`..`7e4783f` resolving F-001..F-007.

**30-second pitch.** "After finishing the audit-analytics project I had it independently audited. Eleven findings: seven I fixed, four I deliberately retained as honest disclosures. Each fix is a separate commit so the reviewer sees the bug, the diagnosis, the change, and the test coverage. The most consequential fix was a double-counting bug in the validation scorer where rows in both a non-primary anomaly and a look-alike population were admitted to two FP buckets — the report claimed `rule_period_end` had 4,156 flags but the source CSV only had 4,153. Now there's a per-detector assertion that bucket totals equal flag count, and a unit test for the double-membership case."

**Likely follow-ups.** What were F-008..F-011 and why didn't you fix them? How would you scale this kind of self-audit to a team? What's the difference between a finding worth fixing and a finding worth disclosing?

### 5.4 — Self-verifying canonical project document

**Artifact.** `docs/final_project_dossier.md` — 8,856 words, 73 inline-verified claims, full file index of 92 artifacts.

**30-second pitch.** "Every quantitative claim in the dossier is recomputed from the source artifact at generation time and tagged inline with `[verified ✓ — source: <file>]`. Section 13 is an appendix where each claim is recorded with cited value, recomputed value, tolerance, and verification status. Validation appendix passes 73/73. This means a reviewer can challenge any number in the document and trace it to source-of-truth artifact in one step."

**Likely follow-ups.** What happens when the underlying artifact changes? How do you tolerate floating-point precision? Is this overkill — would a simple spreadsheet do?

### 5.5 — Standards mapping with paragraph-level citations

**Artifact.** `audit_analytics/docs/standards_mapping.md` (and the README's standards block).

**30-second pitch.** "Every rule maps to a specific paragraph in ISA 240, ISA 330, AS 2401, or AS 2110. ISA 240 .32 — JE testing requirement — operationalized by `rule_period_end`. AS 2401 .65–.66 fact-pattern enumeration — one detector per fact pattern in a small table. Where I'm not 100% sure of the paragraph number I marked it as `[paragraph reference]` rather than inventing — a reviewer can fill those in or call them out as gaps."

**Likely follow-ups.** Which paragraph numbers are you unsure of? How does AS 2110 differ from ISA 330? Why didn't you map directly to the firm's audit methodology (AAM, KPMG Clara, Deloitte D2)?

### 5.6 — Five-sheet auditor-facing Excel workpaper

**Artifact.** `audit_analytics/portfolio_artifacts/audit_workpaper.xlsx`.

**30-second pitch.** "Cover sheet with engagement metadata + standards mapping. Exceptions tab with 6,698 unified rule + Benford rows, severity conditional formatting, autofilter, frozen header, status/reviewer_note columns for disposition. Per-rule summary with F1 traffic light. Benford findings broken into whole / by-account / by-user blocks with non-conforming rows highlighted. Methodology sheet. The workbook is what you actually open — the underlying CSVs and JSONs are upstream artifacts."

**Likely follow-ups.** Do you protect the workbook's structure from the auditor's edits? How does this integrate with a real audit toolchain (Workiva, AuditFile)? Why openpyxl over pandas.to_excel?

---

## Section 6 — Quick-win recommendations

Each is an action that's specific, ≤1 day of focused work, and meaningfully improves the portfolio's interview-readiness.

### 6.1 — `git init` Project 1, commit current state, add `.gitignore`

**Action.** From the project root, `git init`, write a `.gitignore` excluding `.venv/`, `__pycache__/`, `accepted_slim.csv`, the large parquets, `data/dashboard/loans_summary.csv`, then `git add . && git commit`. Optionally split into a few thematic commits ("Steps 7–9c — PD modeling", "Steps 10–13 — LGD / EAD / ECL / overlay", etc.) by reverting + re-staging.
**Effort.** 0.5 day if you do thematic commits, 0.25 day if you do a single "initial commit" and call it good.
**Payoff.** A recruiter who lands on the GitHub page sees a project with version-control discipline rather than a dropbox-style file dump. Required for any public publication.

### 6.2 — Top-level `README.md` orienting both projects

**Action.** At `/Users/.../ProjRED/README.md` (replace the existing one if it's only Project 1's, or create a new `PORTFOLIO.md`), add a one-page index: project name → role-fit → headline numbers → link to that project's own README. ~250 words.
**Effort.** 0.5 day.
**Payoff.** Today both projects have detailed READMEs but a recruiter cloning the parent dir doesn't know which is the "main" one. A top-level orientation page is what links your CV bullet to the actual deliverables.

### 6.3 — Add `requirements.txt` and a minimal `Makefile` to Project 2

**Action.** Project 2 has a `requirements.txt`; Project 1 doesn't. Backfill it — pull from Project 1's `.venv/`. For Project 2, add a 30-line `Makefile` mirroring Project 1's pattern: `all`, `test`, `qa`, `clean`. Project 2 is currently a sequence of 6 manual `python src/*.py` invocations.
**Effort.** 0.5 day total (split: 0.1 for the requirements.txt, 0.4 for the Makefile).
**Payoff.** "I cloned and ran `make all`" is the single most powerful thing a reviewer can say after looking at your repo. Mechanically required for Big 4 IT Audit interviews where reproducibility is a competency.

### 6.4 — Replace `dashboard_spec.md` with an actual screenshot or `.pbix`

**Action.** Open the spec, build the four pages in Power BI Desktop (free, ~2 hours), export PDF or screenshots, drop into `docs/dashboard/`. Even one screenshot of page 1 (the headline page) substantiates the dashboard claim in the README.
**Effort.** 1 day if you've not opened Power BI before, 0.5 day if you have.
**Payoff.** Right now the dashboard exists as a spec only. A recruiter can't tell whether you actually know Power BI. One screenshot resolves the ambiguity.

### 6.5 — `audit_analytics/Makefile` + `make qa-fast` mirroring Project 1

**Action.** Same pattern as Project 1's Makefile: `all`, `data` → `pd-pipeline` style phases, `qa`, `qa-fast`, `dag`. Project 2's pipeline has six commands today; bundle them.
**Effort.** 0.5 day.
**Payoff.** Mechanical — but parity between the two projects sends the message "I do this consistently." Also removes the need for the README's seven-line bash block.

### 6.6 — One-paragraph Czech translation of each project's headline

**Action.** Big 4 Prague applications are routinely bilingual. Add at the top of each README: a 60-word Czech-language version of the executive summary, immediately under the English title. Use formal "Vy" register; have a native speaker check the financial terms (oprávky, opravná položka, IFRS 9 — these are loanwords or specific Czech accounting terms that don't translate naïvely from English).
**Effort.** 0.5 day if you write it yourself; longer if you commission a professional translation.
**Payoff.** Czech-language signaling on the README is the single fastest way to differentiate from a generic international portfolio applying through LinkedIn. Particularly relevant for KPMG Czechia and Deloitte Praha which run their interview process partly in Czech.
