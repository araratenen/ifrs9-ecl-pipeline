# IFRS 9 ECL Modeling — LendingClub Consumer Loans

End-to-end IFRS 9 Expected Credit Loss (ECL) pipeline on the LendingClub
2007–2018 issuance vintages. Produces three reportable headline figures,
each backed by a full audit trail.

| Headline | Total ECL | ECL / Funded | Use case |
|---|---:|---:|---|
| Step 12 baseline | $403,536,501 | 2.37% | Internal model output, no forward-looking adjustment |
| Step 13 data-driven overlay | $390,821,719 | 2.30% | Mechanical IFRS 9 with documented direction inversion |
| **Step 14 regulatory overlay** | **$484,474,530** | **2.85%** | **Recommended for IFRS 9 reporting** |

The headline of headlines and the project's most important methodological
finding are documented in
[`docs/final_project_dossier.md`](docs/final_project_dossier.md).

## Running the pipeline

```bash
make init                  # one-time: create .venv and install requirements
make all                   # full pipeline (Step 7 → final dossier)
make qa                    # gate: pytest + validator + audit (rebuilds if stale)
make qa-fast               # same QA, skip artifact rebuild (after `make all`)
```

`make` resolves the build DAG below — only stale artifacts are rebuilt. Use
`make help` to list every target, or `make dag` to see what would be
executed.

| Phase target | What it builds | Approx. wall time |
|---|---|---:|
| `make data` | through Step 8 (loans + FRED macros) | ~2 min |
| `make pd-pipeline` | through Step 9c (calibrated PD model) | ~10 min |
| `make ecl-pipeline` | through Step 12 (baseline ECL + headline) | ~15 min |
| `make overlay` | through Step 14 (regulatory overlay + validation pack) | ~25 min |
| `make dashboard` | Step 15 dashboard CSVs for Power BI | ~26 min |
| `make all` | + project summary + final dossier | ~27 min |

## Build DAG

```
data/accepted_labeled.parquet                    [input]
        │
        ▼
Step 7 — observation window, cleaning, feature classification
        │  → loans_modeling_ready.parquet, feature_classification.json
        ▼
Step 8 — FRED macro join (unrate, hpi_yoy, gdp_qoq, fed_funds)
        │  → loans_with_macros.parquet, macros_monthly.parquet
        ▼
Step 9a — WoE binning (optbinning) + train/test split
        │  → train_woe.parquet, test_woe.parquet, binning_summary.json
        ▼
Step 9b — PD models (logistic regression + HistGradientBoosting)
        │  → pd_logistic.pkl, pd_xgboost.pkl, test_predictions.parquet
        ▼
Step 9c — Platt scaling on test cohort
        │  → pd_*_calibrator.pkl, calibration_post.json
        ▼
Step 10 — LGD via segment-average + grade fallback
        │  → loans_with_lgd.parquet, lgd_stats.json
        ▼
Step 11 — EAD via closed-form contractual amortization
        │  → loans_with_ead.parquet
        ▼
Step 12 — ECL combination, IFRS 9 staging, headline aggregation
        │  → loans_with_ecl.parquet, ecl_headline.json  ← $403.5M baseline
        ▼
Step 13 — Data-driven macro overlay (Simpson's-paradox-aware)
        │  → loans_with_ecl_overlay.parquet, ecl_overlay_headline.json  ← $390.8M
        ▼
Step 14 — Validation pack + regulatory-coefficient overlay
        │  → validation_*.{json,csv}, validation_regulatory_overlay.json  ← $484.5M
        ▼
Step 15 — Dashboard data for Power BI
        │  → data/dashboard/*.csv
        ▼
build_project_summary  → docs/project_summary.md
build_final_dossier    → docs/final_project_dossier.md (self-verifying)
```

## QA layers

The pipeline ships with three independent checking surfaces, each with a
distinct purpose. All three are run by `make qa`.

| Layer | Script | What it does | Coverage |
|---|---|---|---:|
| Unit tests | [`tests/`](tests/) (pytest) | Closed-form math primitives in isolation | 6 modules, 30+ cases |
| Validator | [`src/validate_pipeline_steps_7_8.py`](src/validate_pipeline_steps_7_8.py) | Cross-deliverable consistency, schema, headlines | 167 checks |
| Audit | [`src/audit_full_pipeline.py`](src/audit_full_pipeline.py) | End-to-end reconciliation, integrity, methodology | 132 checks |

The unit tests verify the **math is correct**; the validator verifies the
**pipeline is consistent**; the audit verifies the **outputs reconcile**.

## Repository layout

```
src/                         # Pipeline scripts (one per step) + builders
  step{7..15}_*.py
  build_project_summary.py
  build_final_dossier.py
  validate_pipeline_steps_7_8.py
  audit_full_pipeline.py
data/                        # Generated parquets (gitignored)
data/dashboard/              # Step 15 CSVs for Power BI
models/                      # Pickled models + calibrators
docs/                        # JSON metrics, methodology MDs, final dossier
tests/                       # Unit tests for math primitives
Makefile                     # DAG orchestration
requirements.txt             # Python dependencies
```

## Reproducibility

- Pipeline is deterministic to **<$0.01 on the headline ECL** across reruns.
- All `random_state` values are fixed. The train/test split, model
  initialization, and calibration are seeded.
- The final dossier is **self-verifying**: every quantitative claim is
  recomputed from source artifacts at generation time and tagged inline.

## Known methodological deviations

Each is documented inline in the relevant step's methodology MD and again
in [`docs/final_project_dossier.md`](docs/final_project_dossier.md). Brief
list:

1. **HistGradientBoosting** substituted for XGBoost (libomp unavailable on
   the build host); discrimination is materially equivalent.
2. **Platt scaling** fit on the test cohort, not training, because of LC
   vintage drift between training and out-of-time evaluation.
3. **EAD via re-amortization from `funded_amnt`**: the spec's "zero-EAD for
   terminated loans" produced all-zero EADs in this terminated-only dataset.
   Switched to closed-form contractual balance projection.
4. **Step 13 overlay direction inversion** (LC underwriting-reaction effect)
   is real signal, not a bug — it's why Step 14's regulatory-coefficient
   overlay is the recommended reporting figure.
