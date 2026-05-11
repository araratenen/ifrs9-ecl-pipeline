"""
Pipeline Validation — Steps 7 and 8.

Loads all deliverables, runs ~30 checks across seven categories, prints a
structured report, and writes the report to docs/validation_report_steps_7_8.md.

Read-only: does not modify any artifact. Exits with code 1 if any critical
check fails.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("/Users/ostappolukainen/Desktop/ProjRED")
LOANS_READY = ROOT / "data" / "loans_modeling_ready.parquet"
LOANS_MACROS = ROOT / "data" / "loans_with_macros.parquet"
MACROS = ROOT / "data" / "macros_monthly.parquet"
FC_JSON = ROOT / "docs" / "feature_classification.json"
STEP7_MD = ROOT / "docs" / "step7_methodology.md"
STEP8_MD = ROOT / "docs" / "step8_methodology.md"
STEP7_PY = ROOT / "src" / "step7_observation_window.py"
STEP8_PY = ROOT / "src" / "step8_macro_features.py"
REPORT = ROOT / "docs" / "validation_report_steps_7_8.md"

# Step 9a artifacts
TRAIN_PARQUET = ROOT / "data" / "train.parquet"
TEST_PARQUET = ROOT / "data" / "test.parquet"
TRAIN_WOE = ROOT / "data" / "train_woe.parquet"
TEST_WOE = ROOT / "data" / "test_woe.parquet"
BINNING_PKL = ROOT / "models" / "binning_process.pkl"
BINNING_SUMMARY = ROOT / "docs" / "binning_summary.json"
BINNING_TABLES_DIR = ROOT / "docs" / "binning_tables"
STEP9A_MD = ROOT / "docs" / "step9a_methodology.md"
STEP9A_PY = ROOT / "src" / "step9a_woe_binning.py"

# Step 9b artifacts
PD_LR = ROOT / "models" / "pd_logistic.pkl"
PD_XGB = ROOT / "models" / "pd_xgboost.pkl"
TEST_PREDICTIONS = ROOT / "data" / "test_predictions.parquet"
EVAL_JSON = ROOT / "docs" / "model_evaluation.json"
COEFS_CSV = ROOT / "docs" / "coefficients_lr.csv"
IMPORTANCE_CSV = ROOT / "docs" / "feature_importance_xgb.csv"
LIFT_LR_CSV = ROOT / "docs" / "decile_lift_lr.csv"
LIFT_XGB_CSV = ROOT / "docs" / "decile_lift_xgb.csv"
STEP9B_MD = ROOT / "docs" / "step9b_methodology.md"

# Step 9c artifacts
CAL_LR = ROOT / "models" / "pd_logistic_calibrator.pkl"
CAL_XGB = ROOT / "models" / "pd_xgboost_calibrator.pkl"
CAL_PRE_JSON = ROOT / "docs" / "calibration_pre.json"
CAL_POST_JSON = ROOT / "docs" / "calibration_post.json"
CAL_COMP_JSON = ROOT / "docs" / "calibration_comparison.json"
CALIBRATORS_JSON = ROOT / "docs" / "calibrators.json"
REL_COMBINED = ROOT / "docs" / "reliability_combined.csv"
STEP9C_MD = ROOT / "docs" / "step9c_methodology.md"

# Step 10 artifacts
LOANS_LGD = ROOT / "data" / "loans_with_lgd.parquet"
LGD_LOOKUP = ROOT / "docs" / "lgd_lookup.csv"
LGD_HIST = ROOT / "docs" / "lgd_histogram.csv"
LGD_BACKTEST = ROOT / "docs" / "lgd_backtest.csv"
LGD_SENSITIVITY = ROOT / "docs" / "lgd_sensitivity.csv"
LGD_STATS = ROOT / "docs" / "lgd_stats.json"
STEP10_MD = ROOT / "docs" / "step10_methodology.md"

# Step 11 artifacts
LOANS_EAD = ROOT / "data" / "loans_with_ead.parquet"
EAD_HIST = ROOT / "docs" / "ead_histogram.csv"
EAD_MONTHS_DIST = ROOT / "docs" / "ead_months_remaining_distribution.csv"
EAD_STATUS_BREAK = ROOT / "docs" / "ead_status_breakdown.csv"
STEP11_MD = ROOT / "docs" / "step11_methodology.md"

# Step 12 artifacts
LOANS_ECL = ROOT / "data" / "loans_with_ecl.parquet"
ECL_HEADLINE = ROOT / "docs" / "ecl_headline.json"
ECL_BY_STAGE = ROOT / "docs" / "ecl_by_stage.csv"
ECL_BY_GRADE = ROOT / "docs" / "ecl_by_grade.csv"
ECL_BY_VINTAGE = ROOT / "docs" / "ecl_by_vintage.csv"
ECL_BY_PURPOSE = ROOT / "docs" / "ecl_by_purpose.csv"
STEP12_MD = ROOT / "docs" / "step12_methodology.md"

# Step 13 artifacts
LOANS_OVERLAY = ROOT / "data" / "loans_with_ecl_overlay.parquet"
OVERLAY_HEADLINE = ROOT / "docs" / "ecl_overlay_headline.json"
OVERLAY_BY_STAGE = ROOT / "docs" / "ecl_overlay_by_stage.csv"
OVERLAY_BY_GRADE = ROOT / "docs" / "ecl_overlay_by_grade.csv"
OVERLAY_BY_VINTAGE = ROOT / "docs" / "ecl_overlay_by_vintage.csv"
STEP13_MD = ROOT / "docs" / "step13_methodology.md"

PROJECT_SUMMARY = ROOT / "docs" / "project_summary.md"
FINAL_DOSSIER = ROOT / "docs" / "final_project_dossier.md"

# Step 15 artifacts (dashboard)
DASHBOARD_DIR = ROOT / "data" / "dashboard"
DASH_LOANS = DASHBOARD_DIR / "loans_summary.csv"
DASH_HEADLINE = DASHBOARD_DIR / "headline_metrics.csv"
DASH_DISCRIM = DASHBOARD_DIR / "discrimination_metrics.csv"
DASH_CALIB = DASHBOARD_DIR / "calibration_table.csv"
DASH_SENS = DASHBOARD_DIR / "sensitivity_table.csv"
DASHBOARD_SPEC = ROOT / "docs" / "dashboard_spec.md"

# Step 14 artifacts
VAL_DISCRIM = ROOT / "docs" / "validation_discrimination.json"
VAL_CALIB = ROOT / "docs" / "validation_calibration.json"
VAL_AUC_VINTAGE = ROOT / "docs" / "validation_auc_by_vintage.csv"
VAL_AUC_GRADE = ROOT / "docs" / "validation_auc_by_grade.csv"
VAL_GAIN = ROOT / "docs" / "validation_gain_curve.csv"
VAL_RELIAB = ROOT / "docs" / "validation_reliability_test.csv"
VAL_CALIB_GRADE = ROOT / "docs" / "validation_calibration_by_grade.csv"
VAL_CALIB_VINTAGE = ROOT / "docs" / "validation_calibration_by_vintage.csv"
VAL_SICR = ROOT / "docs" / "validation_sicr_sensitivity.csv"
VAL_WEIGHTS = ROOT / "docs" / "validation_overlay_weights.csv"
VAL_REG_OVERLAY = ROOT / "docs" / "validation_regulatory_overlay.json"
VAL_PSI = ROOT / "docs" / "validation_psi_over_time.csv"
VAL_FEATURE_STRESS = ROOT / "docs" / "validation_single_feature_stress.csv"
VAL_STAGE_MIGRATION = ROOT / "docs" / "validation_stage_migration.csv"
FINAL_REPORT = ROOT / "docs" / "final_validation_report.md"

AS_OF = pd.Timestamp("2019-04-01")
ANNUAL_INC_CAP = 250_000.0
LEAKAGE_COLS = [
    "last_fico_range_low", "last_fico_range_high", "loan_status",
    "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
    "out_prncp", "chargeoff_within_12_mths", "hardship_flag", "debt_settlement_flag",
]


def make(check_id: str, name: str, ok: bool, value, expected: str = "") -> tuple:
    return (check_id, name, bool(ok), str(value), str(expected))


# ---------- Task 1 ----------

def task_1_files() -> tuple[list, bool]:
    files = [
        ("1.1", "loans_modeling_ready.parquet", LOANS_READY, "parquet"),
        ("1.2", "loans_with_macros.parquet", LOANS_MACROS, "parquet"),
        ("1.3", "macros_monthly.parquet", MACROS, "parquet"),
        ("1.4", "feature_classification.json", FC_JSON, "json"),
        ("1.5", "step7_methodology.md", STEP7_MD, "md"),
        ("1.6", "step8_methodology.md", STEP8_MD, "md"),
        ("1.7", "step7_observation_window.py", STEP7_PY, "py"),
        ("1.8", "step8_macro_features.py", STEP8_PY, "py"),
    ]
    results, all_ok = [], True
    for cid, name, path, kind in files:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_ok = False
    return results, all_ok


def _check_file(path: Path, kind: str) -> tuple[bool, str, str]:
    if not path.exists():
        return False, "missing", "exists"
    size = path.stat().st_size
    if size == 0:
        return False, "empty", "non-empty"
    size_mb = size / 1024 ** 2
    try:
        if kind == "parquet":
            pq.ParquetFile(path).schema  # cheap metadata read
        elif kind == "json":
            json.loads(path.read_text())
        elif kind == "pkl":
            import joblib
            joblib.load(path)
        elif kind == "csv":
            pd.read_csv(path, nrows=5)
        elif kind in ("md", "py"):
            path.read_text()
    except Exception as e:
        return False, f"unreadable: {type(e).__name__}", "readable"
    return True, f"{size_mb:.1f} MB", "OK"


# ---------- Task 2 ----------

def task_2_schema(loans_ready: pd.DataFrame, loans_macros: pd.DataFrame) -> list:
    results = []

    n_ready, n_macros = len(loans_ready), len(loans_macros)
    results.append(make(
        "2.1", "Row counts match between Step 7 and Step 8 outputs",
        n_ready == n_macros,
        f"ready={n_ready:,}, macros={n_macros:,}", "equal",
    ))

    ids_ready = set(loans_ready["id"].astype(str))
    ids_macros = set(loans_macros["id"].astype(str))
    sym_diff = len(ids_ready ^ ids_macros)
    results.append(make(
        "2.2", "Loan IDs identical between the two parquets",
        ids_ready == ids_macros,
        f"ready={len(ids_ready):,}, macros={len(ids_macros):,}, sym_diff={sym_diff}",
        "identical sets",
    ))

    cols_ready, cols_macros = set(loans_ready.columns), set(loans_macros.columns)
    added = cols_macros - cols_ready
    removed = cols_ready - cols_macros
    expected_added = {"unrate", "gdp_yoy", "fedfunds", "hpi_yoy"}
    results.append(make(
        "2.3", "Column delta = exactly 4 macros, no Step 7 cols dropped",
        added == expected_added and removed == set(),
        f"added={sorted(added)}, removed={sorted(removed)}",
        "added=[fedfunds,gdp_yoy,hpi_yoy,unrate], removed=[]",
    ))

    required = [
        "id", "issue_d", "default_flag",
        "loan_amnt", "term_months", "int_rate", "grade", "purpose", "dti",
        "fico_range_low", "annual_inc", "home_ownership", "emp_length_years",
        "loan_status", "recoveries", "collection_recovery_fee", "total_pymnt", "out_prncp",
        "unrate", "gdp_yoy", "fedfunds", "hpi_yoy",
    ]
    missing = [c for c in required if c not in loans_macros.columns]
    results.append(make(
        "2.4", f"All {len(required)} required columns present",
        not missing,
        f"missing={missing}" if missing else "all present",
        "all present",
    ))

    type_failures: list[str] = []
    if not pd.api.types.is_datetime64_any_dtype(loans_macros["issue_d"]):
        type_failures.append(f"issue_d:{loans_macros['issue_d'].dtype}")
    if not pd.api.types.is_integer_dtype(loans_macros["default_flag"]):
        type_failures.append(f"default_flag:{loans_macros['default_flag'].dtype}")
    df_unique = set(loans_macros["default_flag"].unique().tolist())
    if df_unique != {0, 1}:
        type_failures.append(f"default_flag values:{df_unique}")
    for c in ["loan_amnt", "int_rate", "dti", "fico_range_low", "annual_inc",
              "unrate", "gdp_yoy", "fedfunds", "hpi_yoy"]:
        if not pd.api.types.is_numeric_dtype(loans_macros[c]):
            type_failures.append(f"{c}:{loans_macros[c].dtype}")
    for c in ["grade", "purpose", "home_ownership"]:
        d = loans_macros[c].dtype
        is_cat_or_str = (
            isinstance(d, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(d)
            or d == object
        )
        if not is_cat_or_str:
            type_failures.append(f"{c}:{d}")
    results.append(make(
        "2.5", "Column types are sensible",
        not type_failures,
        "; ".join(type_failures) if type_failures else "all OK",
        "datetime/int/float/cat as expected",
    ))

    return results


# ---------- Task 3 ----------

def task_3_quality(loans: pd.DataFrame) -> list:
    results = []

    dnmcp = loans["loan_status"].astype(str).str.startswith("Does not meet the credit policy")
    n = int(dnmcp.sum())
    results.append(make("3.1", "No DNMCP rows", n == 0, f"{n} rows", "0"))

    n999 = int((loans["dti"] == 999).sum())
    n_nan = int(loans["dti"].isna().sum())
    results.append(make(
        "3.2", "No dti == 999 sentinel",
        n999 == 0, f"{n999} rows; {n_nan} NaN", "0 rows",
    ))

    max_ru = float(loans["revol_util"].max())
    results.append(make(
        "3.3", "revol_util ≤ 100", max_ru <= 100, f"max={max_ru}", "≤ 100",
    ))

    max_ai = float(loans["annual_inc"].max())
    results.append(make(
        "3.4", f"annual_inc winsorized ≤ ${ANNUAL_INC_CAP:,.0f}",
        max_ai <= ANNUAL_INC_CAP,
        f"max=${max_ai:,.0f}", f"≤ ${ANNUAL_INC_CAP:,.0f}",
    ))

    min_fico = int(loans["fico_range_low"].min())
    results.append(make(
        "3.5", "fico_range_low ≥ 660 (LC issuance floor)",
        min_fico >= 660, f"min={min_fico}", "≥ 660",
    ))

    months = (
        (AS_OF.year - loans["issue_d"].dt.year) * 12
        + (AS_OF.month - loans["issue_d"].dt.month)
    )
    min_mo = int(months.min())
    results.append(make(
        "3.6", "months_observable ≥ 24 (maturity filter)",
        min_mo >= 24, f"min={min_mo}", "≥ 24",
    ))

    critical = [
        "default_flag", "issue_d", "fico_range_low", "int_rate",
        "loan_amnt", "term_months",
        "unrate", "gdp_yoy", "fedfunds", "hpi_yoy",
    ]
    nulls = {c: int(loans[c].isna().sum()) for c in critical}
    bad = {c: n for c, n in nulls.items() if n > 0}
    results.append(make(
        "3.7", "No nulls in critical fields",
        not bad,
        f"bad={bad}" if bad else "0 nulls in all fields",
        "0 nulls",
    ))

    return results


# ---------- Task 4 ----------

def task_4_classification(fc: dict, loans: pd.DataFrame) -> list:
    results = []

    expected_keys = {"pd_inputs", "identifiers", "outcome_only", "label"}
    keys = set(fc.keys())
    results.append(make(
        "4.1", "JSON has 4 expected top-level keys",
        keys == expected_keys,
        f"keys={sorted(keys)}",
        f"{sorted(expected_keys)}",
    ))

    label_ok = fc.get("label") == "default_flag"
    results.append(make(
        "4.2", "Label == 'default_flag'",
        label_ok, f"label={fc.get('label')!r}", "'default_flag'",
    ))

    pd_inputs = fc.get("pd_inputs", [])
    has_macros = all(m in pd_inputs for m in ["unrate", "gdp_yoy", "fedfunds", "hpi_yoy"])
    has_year = "issue_year" in pd_inputs
    n = len(pd_inputs)
    count_ok = 31 <= n <= 35
    results.append(make(
        "4.3", "pd_inputs ≈ 33 features incl. 4 macros + issue_year",
        has_macros and has_year and count_ok,
        f"count={n}, macros={'yes' if has_macros else 'no'}, "
        f"issue_year={'yes' if has_year else 'no'}",
        "33±2 incl. macros + issue_year",
    ))

    overlap = set(pd_inputs) & set(fc.get("outcome_only", []))
    results.append(make(
        "4.4", "No overlap between pd_inputs and outcome_only",
        not overlap,
        f"overlap={sorted(overlap)}" if overlap else "no overlap",
        "empty",
    ))

    identifiers = fc.get("identifiers", [])
    pd_set = set(pd_inputs)
    id_ok = ("id" in identifiers) and ("id" not in pd_set)
    issue_d_ok = ("issue_d" in identifiers) and ("issue_d" not in pd_set)
    results.append(make(
        "4.5", "id and issue_d in identifiers, not pd_inputs",
        id_ok and issue_d_ok,
        f"id_OK={id_ok}, issue_d_OK={issue_d_ok}",
        "both in identifiers",
    ))

    available = set(loans.columns)
    pd_missing = [c for c in pd_inputs if c not in available and c != "issue_year"]
    out_missing = [c for c in fc.get("outcome_only", []) if c not in available]
    results.append(make(
        "4.6", "pd_inputs and outcome_only resolvable to columns",
        not pd_missing and not out_missing,
        (f"pd_missing={pd_missing}, out_missing={out_missing}"
         if (pd_missing or out_missing) else "all present"),
        "all present (issue_year derivable)",
    ))

    leaks = [c for c in LEAKAGE_COLS if c in pd_inputs]
    results.append(make(
        "4.7", "No leakage columns in pd_inputs",
        not leaks,
        f"leaks={leaks}" if leaks else "no leaks",
        "empty",
    ))

    return results


# ---------- Task 5 ----------

def task_5_distributional(loans: pd.DataFrame) -> list:
    results = []

    rate = float(loans["default_flag"].mean())
    results.append(make(
        "5.1", "Overall default rate in 19–21%",
        0.19 <= rate <= 0.21,
        f"{rate * 100:.2f}%", "19–21%",
    ))

    by_grade = (loans.groupby("grade", observed=True)["default_flag"].mean() * 100).round(2).sort_index()
    diffs = by_grade.diff().dropna()
    monotonic = bool((diffs >= 0).all())
    results.append(make(
        "5.2", "Default rate monotonic A→G",
        monotonic, f"{by_grade.to_dict()}", "non-decreasing",
    ))

    bins = [660, 700, 740, 780, 1000]
    labels_ = ["660-700", "700-740", "740-780", "780+"]
    band = pd.cut(loans["fico_range_low"], bins=bins, labels=labels_,
                  right=False, include_lowest=True)
    by_band = (loans.groupby(band, observed=True)["default_flag"].mean() * 100).round(2).dropna()
    diffs = by_band.diff().dropna()
    monotonic = bool((diffs <= 0).all())
    results.append(make(
        "5.3", "Default rate monotonic ↓ across FICO bands",
        monotonic, f"{by_band.to_dict()}", "non-increasing",
    ))

    yr = loans["issue_d"].dt.year
    by_year = loans.groupby(yr)["default_flag"].mean()
    bad = by_year[(by_year < 0.05) | (by_year > 0.30)]
    results.append(make(
        "5.4", "All years have 5–30% default rate",
        bad.empty,
        (f"out_of_range={bad.round(4).to_dict()}" if not bad.empty
         else f"{(by_year * 100).round(1).to_dict()}"),
        "5–30%",
    ))

    bounds = {
        "unrate": (3.0, 10.5),
        "fedfunds": (0.0, 6.0),
        "hpi_yoy": (-20, 15),
        "gdp_yoy": (-5, 5),
    }
    bad: list[str] = []
    for m, (lo, hi) in bounds.items():
        col_min = float(loans[m].min())
        col_max = float(loans[m].max())
        if not (col_min >= lo and col_max <= hi):
            bad.append(f"{m}=[{col_min:.2f},{col_max:.2f}] expected [{lo},{hi}]")
    results.append(make(
        "5.5", "Macros within plausible US ranges",
        not bad,
        "; ".join(bad) if bad else "all in range",
        "see bounds",
    ))

    macro_nulls = {m: int(loans[m].isna().sum()) for m in
                   ["unrate", "gdp_yoy", "fedfunds", "hpi_yoy"]}
    bad_macros = {m: n for m, n in macro_nulls.items() if n > 0}
    results.append(make(
        "5.6", "All macro columns zero nulls",
        not bad_macros,
        f"bad={bad_macros}" if bad_macros else "0 nulls in all macros",
        "0 nulls",
    ))

    return results


# ---------- Task 6 ----------

def task_6_consistency(loans_ready: pd.DataFrame, loans_macros: pd.DataFrame,
                       step7_md: str, step8_md: str) -> list:
    results = []

    actual_n7 = len(loans_ready)
    cited_n7 = _parse_int(step7_md, [
        r"\*\*[Ff]inal rows:?\*\*[:\s]+([\d,]+)",
        r"[Ff]inal rows:?\*?\*?[\s]+([\d,]+)",
    ])
    if cited_n7 is None:
        results.append(make(
            "6.1", "Step 7 cited row count matches actual",
            False, f"actual={actual_n7:,}; no citation parsed", f"≈ {actual_n7:,}",
        ))
    else:
        ok = abs(cited_n7 - actual_n7) <= 10
        results.append(make(
            "6.1", "Step 7 cited row count matches actual",
            ok, f"actual={actual_n7:,}, cited={cited_n7:,}", "within ±10",
        ))

    actual_n8 = len(loans_macros)
    cited_n8 = _parse_int(step8_md, [
        r"\*\*[Ss]hape:?\*\*[:\s]+([\d,]+)\s*rows",
        r"[Ss]hape:\s*([\d,]+)\s*rows",
    ])
    if cited_n8 is None:
        results.append(make(
            "6.2", "Step 8 cited row count matches actual",
            False, f"actual={actual_n8:,}; no citation parsed", f"≈ {actual_n8:,}",
        ))
    else:
        ok = abs(cited_n8 - actual_n8) <= 10
        results.append(make(
            "6.2", "Step 8 cited row count matches actual",
            ok, f"actual={actual_n8:,}, cited={cited_n8:,}", "within ±10",
        ))

    actual_rate = float(loans_ready["default_flag"].mean()) * 100
    m = re.search(r"\*\*[Dd]efault rate:?\*\*[:\s]+([\d.]+)\s*%", step7_md)
    if m:
        cited_rate = float(m.group(1))
        ok = abs(cited_rate - actual_rate) <= 0.1
        results.append(make(
            "6.3", "Step 7 cited default rate matches actual",
            ok,
            f"actual={actual_rate:.2f}%, cited={cited_rate:.2f}%",
            "within 0.1pp",
        ))
    else:
        results.append(make(
            "6.3", "Step 7 cited default rate matches actual",
            False,
            f"actual={actual_rate:.2f}%; no citation parsed",
            "within 0.1pp",
        ))

    df = loans_macros.copy()
    df["issue_year"] = df["issue_d"].dt.year
    g = df.groupby("issue_year")
    dr = df["default_flag"] - g["default_flag"].transform("mean")
    actual_corrs: dict[str, float] = {}
    for macro in ["unrate", "gdp_yoy", "fedfunds", "hpi_yoy"]:
        mr = df[macro] - g[macro].transform("mean")
        actual_corrs[macro] = float(pd.Series(dr).corr(pd.Series(mr)))
    cited = _parse_within_corrs(step8_md)
    if not cited:
        results.append(make(
            "6.4", "Within-year correlations match cited values",
            False,
            f"actual={ {k: round(v, 4) for k, v in actual_corrs.items()} }; "
            "no §4 table parsed",
            "within ±0.001",
        ))
    else:
        diffs = {m: abs(actual_corrs[m] - cited[m])
                 for m in actual_corrs if m in cited}
        max_diff = max(diffs.values()) if diffs else float("inf")
        ok = max_diff <= 0.001
        results.append(make(
            "6.4", "Within-year correlations match cited values",
            ok,
            f"actual={ {k: round(v, 4) for k, v in actual_corrs.items()} }, "
            f"cited={ {k: round(v, 4) for k, v in cited.items()} }, "
            f"max_diff={max_diff:.4f}",
            "within ±0.001",
        ))

    return results


def _parse_int(md: str, patterns: list[str]) -> int | None:
    for p in patterns:
        m = re.search(p, md)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def _parse_within_corrs(md: str) -> dict[str, float]:
    """Pick the year-FE column from the §4 robustness table.

    Table rows look like: | unrate | -0.0638 | -0.0062 | -0.0101 | -0.0098 |
    """
    pat = re.compile(
        r"^\|\s*(unrate|gdp_yoy|fedfunds|hpi_yoy)\s*\|"
        r"\s*[+\-]?[\d.]+\s*\|"
        r"\s*([+\-]?[\d.]+)\s*\|",
        re.MULTILINE,
    )
    return {macro: float(val) for macro, val in pat.findall(md)}


# ---------- Task 7 ----------

def task_7_readiness(loans: pd.DataFrame, fc: dict) -> list:
    results = []

    n = len(loans)
    results.append(make(
        "7.1", "Sample size > 500K", n > 500_000, f"{n:,}", "> 500,000",
    ))

    n_def = int(loans["default_flag"].sum())
    results.append(make(
        "7.2", "Defaults ≥ 50K", n_def >= 50_000, f"{n_def:,}", "≥ 50,000",
    ))

    train_mask = loans["issue_d"] < pd.Timestamp("2016-01-01")
    n_train = int(train_mask.sum())
    n_test = int((~train_mask).sum())
    n_train_def = int(loans.loc[train_mask, "default_flag"].sum())
    n_test_def = int(loans.loc[~train_mask, "default_flag"].sum())
    ok = (n_train > 100_000 and n_test > 100_000
          and n_train_def > 10_000 and n_test_def > 10_000)
    results.append(make(
        "7.3", "Time-split feasible (issue_d < 2016 vs ≥ 2016)",
        ok,
        f"train={n_train:,} ({n_train_def:,} def), "
        f"test={n_test:,} ({n_test_def:,} def)",
        "both > 100K rows + 10K defaults",
    ))

    pd_inputs = fc.get("pd_inputs", [])
    available = set(loans.columns)
    derivable = {"issue_year"}
    bad = [c for c in pd_inputs if c not in available and c not in derivable]
    results.append(make(
        "7.4", "All pd_inputs present or derivable",
        not bad,
        f"non_derivable={bad}" if bad else "all OK",
        "all OK",
    ))

    # earliest_cr_line is a datetime column with ~700 unique month-year values.
    # The methodology notes it will be feature-engineered to credit_history_years
    # in Step 9; treating it as raw categorical here would falsely fail the
    # "2-100 unique" check. Datetimes only need ≥2 unique values.
    bad_degenerate: list[str] = []
    for col in pd_inputs:
        if col not in available:
            continue
        s = loans[col]
        nu = int(s.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            if nu < 2:
                bad_degenerate.append(f"{col}={nu} (numeric/datetime needs ≥2)")
        else:
            if nu < 2 or nu > 100:
                bad_degenerate.append(f"{col}={nu} (cat needs 2–100)")
    results.append(make(
        "7.5", "pd_inputs non-degenerate",
        not bad_degenerate,
        "; ".join(bad_degenerate) if bad_degenerate else "all OK",
        "≥2 numeric/datetime; 2–100 cat",
    ))

    too_high, warn_missing = [], []
    for col in pd_inputs:
        if col not in available:
            continue
        rate = float(loans[col].isna().mean())
        if rate > 0.50:
            too_high.append(f"{col}={rate * 100:.1f}%")
        elif rate > 0.30:
            warn_missing.append(f"{col}={rate * 100:.1f}%")
    parts = []
    if too_high:
        parts.append(f"FAIL>50%={too_high}")
    if warn_missing:
        parts.append(f"WARN>30%={warn_missing}")
    results.append(make(
        "7.6", "No pd_input column > 50% missing",
        not too_high,
        "; ".join(parts) if parts else "all < 30%",
        "≤ 50%",
    ))

    return results


# ---------- Task 8 (Step 9a artifacts) ----------

def task_8_step9a() -> list:
    results = []

    artifacts = [
        ("8.1", "models/binning_process.pkl", BINNING_PKL, "pkl"),
        ("8.2", "data/train.parquet", TRAIN_PARQUET, "parquet"),
        ("8.3", "data/test.parquet", TEST_PARQUET, "parquet"),
        ("8.4", "data/train_woe.parquet", TRAIN_WOE, "parquet"),
        ("8.5", "data/test_woe.parquet", TEST_WOE, "parquet"),
        ("8.6", "docs/binning_summary.json", BINNING_SUMMARY, "json"),
        ("8.7", "docs/binning_tables/_summary.csv",
         BINNING_TABLES_DIR / "_summary.csv", "csv"),
    ]
    all_files_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind if kind != "pkl" else "pkl")
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_files_ok = False

    if not all_files_ok:
        results.append(make(
            "8.8", "WoE parquets and per-feature CSVs (skipped)",
            False, "skipped — missing artifact above", "all checks ok",
        ))
        return results

    train = pd.read_parquet(TRAIN_PARQUET, columns=["id"])
    test = pd.read_parquet(TEST_PARQUET, columns=["id"])
    train_woe = pd.read_parquet(TRAIN_WOE)
    test_woe = pd.read_parquet(TEST_WOE)

    n_train, n_test = len(train), len(test)
    n_train_woe, n_test_woe = len(train_woe), len(test_woe)
    results.append(make(
        "8.8", "train_woe row count matches train",
        n_train == n_train_woe,
        f"train={n_train:,}, train_woe={n_train_woe:,}", "equal",
    ))
    results.append(make(
        "8.9", "test_woe row count matches test",
        n_test == n_test_woe,
        f"test={n_test:,}, test_woe={n_test_woe:,}", "equal",
    ))

    woe_cols = [c for c in train_woe.columns if c not in {"id", "issue_d", "default_flag"}]
    train_nulls = int(train_woe[woe_cols].isna().sum().sum())
    test_nulls = int(test_woe[woe_cols].isna().sum().sum())
    results.append(make(
        "8.10", "Zero nulls in WoE parquets",
        train_nulls == 0 and test_nulls == 0,
        f"train={train_nulls}, test={test_nulls}", "0",
    ))

    summary = json.loads(BINNING_SUMMARY.read_text())
    iv_values = [e["iv"] for e in summary["iv_table"]]
    bad_neg = [e["feature"] for e in summary["iv_table"] if e["iv"] < 0]
    bad_high = [e["feature"] for e in summary["iv_table"] if e["iv"] > 0.8]
    iv_ok = not bad_neg and not bad_high
    results.append(make(
        "8.11", "IV values reasonable (≥0 and ≤0.8)",
        iv_ok,
        (f"neg={bad_neg}, >0.8={bad_high}" if not iv_ok
         else f"min={min(iv_values):.4f}, max={max(iv_values):.4f}"),
        "0 ≤ iv ≤ 0.8",
    ))

    selected = [e["feature"] for e in summary["iv_table"]
                if e["status"] in ("selected", "selected_forced")]
    missing_csvs = [
        f for f in selected
        if not (BINNING_TABLES_DIR / f"{f}.csv").exists()
    ]
    n_iv = sum(1 for e in summary["iv_table"] if e["status"] == "selected")
    n_forced = sum(1 for e in summary["iv_table"] if e["status"] == "selected_forced")
    results.append(make(
        "8.12", "Per-feature binning CSV exists for every selected feature",
        not missing_csvs,
        (f"missing={missing_csvs}" if missing_csvs
         else f"{len(selected)} CSVs OK ({n_iv} IV + {n_forced} forced)"),
        "all present",
    ))

    return results


# ---------- Task 9 (Step 9b artifacts) ----------

def task_9_step9b() -> list:
    results = []

    artifacts = [
        ("9.1", "models/pd_logistic.pkl", PD_LR, "pkl"),
        ("9.2", "models/pd_xgboost.pkl", PD_XGB, "pkl"),
        ("9.3", "data/test_predictions.parquet", TEST_PREDICTIONS, "parquet"),
        ("9.4", "docs/model_evaluation.json", EVAL_JSON, "json"),
        ("9.5", "docs/coefficients_lr.csv", COEFS_CSV, "csv"),
        ("9.6", "docs/feature_importance_xgb.csv", IMPORTANCE_CSV, "csv"),
        ("9.7", "docs/step9b_methodology.md", STEP9B_MD, "md"),
    ]
    all_files_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_files_ok = False

    if not all_files_ok:
        results.append(make(
            "9.8", "Downstream Step 9b checks (skipped)",
            False, "skipped — missing artifact above", "all checks ok",
        ))
        return results

    import joblib
    from sklearn.linear_model import LogisticRegression
    model_lr = joblib.load(PD_LR)
    results.append(make(
        "9.8", "pd_logistic.pkl is a LogisticRegression",
        isinstance(model_lr, LogisticRegression),
        f"type={type(model_lr).__name__}", "LogisticRegression",
    ))

    model_xgb = joblib.load(PD_XGB)
    has_predict_proba = hasattr(model_xgb, "predict_proba")
    results.append(make(
        "9.9", "pd_xgboost.pkl is a sklearn classifier with predict_proba",
        has_predict_proba,
        f"type={type(model_xgb).__name__}", "has predict_proba",
    ))

    test_pred = pd.read_parquet(TEST_PREDICTIONS)
    expected_cols = {"id", "issue_d", "default_flag", "pd_lr", "pd_xgb"}
    has_cols = expected_cols.issubset(set(test_pred.columns))
    null_pd = int(test_pred[["pd_lr", "pd_xgb"]].isna().sum().sum()) if has_cols else -1
    results.append(make(
        "9.10", "test_predictions.parquet has required columns + no PD nulls",
        has_cols and null_pd == 0,
        f"cols={sorted(test_pred.columns)}, null_pd={null_pd}",
        f"{sorted(expected_cols)}, 0 nulls",
    ))

    eval_payload = json.loads(EVAL_JSON.read_text())
    auc_lr = eval_payload["models"]["logistic_regression"]["auc"]
    auc_xgb = eval_payload["models"]["xgboost"]["auc"]
    auc_ok = (0.65 <= auc_lr <= 0.85) and (0.65 <= auc_xgb <= 0.85)
    results.append(make(
        "9.11", "Both models AUC in [0.65, 0.85]",
        auc_ok,
        f"LR={auc_lr:.4f}, GBM={auc_xgb:.4f}", "[0.65, 0.85]",
    ))

    coefs = pd.read_csv(COEFS_CSV)
    importance = pd.read_csv(IMPORTANCE_CSV)
    expected_n = 19
    coef_ok = len(coefs) == expected_n
    imp_ok = len(importance) == expected_n
    results.append(make(
        "9.12", f"coefficients_lr.csv has {expected_n} rows",
        coef_ok, f"rows={len(coefs)}", str(expected_n),
    ))
    results.append(make(
        "9.13", f"feature_importance_xgb.csv has {expected_n} rows",
        imp_ok, f"rows={len(importance)}", str(expected_n),
    ))

    return results


# ---------- Task 10 (Step 9c artifacts) ----------

def task_10_step9c() -> list:
    results = []

    artifacts = [
        ("10.1", "models/pd_logistic_calibrator.pkl", CAL_LR, "pkl"),
        ("10.2", "models/pd_xgboost_calibrator.pkl", CAL_XGB, "pkl"),
        ("10.3", "docs/calibration_comparison.json", CAL_COMP_JSON, "json"),
        ("10.4", "docs/calibrators.json", CALIBRATORS_JSON, "json"),
        ("10.5", "docs/reliability_combined.csv", REL_COMBINED, "csv"),
        ("10.6", "docs/step9c_methodology.md", STEP9C_MD, "md"),
    ]
    all_files_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_files_ok = False

    if not all_files_ok:
        results.append(make("10.7", "Downstream Step 9c checks (skipped)",
                             False, "skipped — missing artifact above", "all checks ok"))
        return results

    import joblib
    from sklearn.linear_model import LogisticRegression
    cal_lr = joblib.load(CAL_LR)
    cal_xgb = joblib.load(CAL_XGB)
    results.append(make(
        "10.7", "Calibrators are LogisticRegression instances",
        isinstance(cal_lr, LogisticRegression) and isinstance(cal_xgb, LogisticRegression),
        f"LR={type(cal_lr).__name__}, GBM={type(cal_xgb).__name__}",
        "LogisticRegression",
    ))

    test_pred = pd.read_parquet(TEST_PREDICTIONS)
    needed = {"pd_lr_calibrated", "pd_xgb_calibrated"}
    has_cols = needed.issubset(set(test_pred.columns))
    null_cal = (int(test_pred[list(needed)].isna().sum().sum()) if has_cols else -1)
    results.append(make(
        "10.8", "test_predictions has calibrated columns with no nulls",
        has_cols and null_cal == 0,
        f"cols={has_cols}, nulls={null_cal}",
        "both cols present, 0 nulls",
    ))

    lr_in_range = test_pred["pd_lr_calibrated"].between(0, 1).all() if has_cols else False
    xgb_in_range = test_pred["pd_xgb_calibrated"].between(0, 1).all() if has_cols else False
    results.append(make(
        "10.9", "Calibrated PDs in [0, 1]",
        lr_in_range and xgb_in_range,
        (f"LR=[{test_pred['pd_lr_calibrated'].min():.4f}, {test_pred['pd_lr_calibrated'].max():.4f}], "
         f"GBM=[{test_pred['pd_xgb_calibrated'].min():.4f}, {test_pred['pd_xgb_calibrated'].max():.4f}]"
         if has_cols else "missing cols"),
        "[0, 1]",
    ))

    cmp = json.loads(CAL_COMP_JSON.read_text())
    lr_pre = cmp["logistic_regression"]["pre"]["ece"]
    lr_post = cmp["logistic_regression"]["post"]["ece"]
    xgb_pre = cmp["xgboost"]["pre"]["ece"]
    xgb_post = cmp["xgboost"]["post"]["ece"]
    ece_ok = (lr_post <= lr_pre + 1e-6) and (xgb_post <= xgb_pre + 1e-6)
    results.append(make(
        "10.10", "Post-ECE ≤ Pre-ECE (calibration didn't worsen)",
        ece_ok,
        f"LR: {lr_pre:.4f}→{lr_post:.4f}; GBM: {xgb_pre:.4f}→{xgb_post:.4f}",
        "post ≤ pre",
    ))

    rel = pd.read_csv(REL_COMBINED)
    results.append(make(
        "10.11", "reliability_combined.csv has 10 deciles",
        len(rel) == 10, f"rows={len(rel)}", "10",
    ))

    from sklearn.metrics import roc_auc_score
    auc_lr_raw = float(roc_auc_score(test_pred["default_flag"], test_pred["pd_lr"]))
    auc_lr_cal = float(roc_auc_score(test_pred["default_flag"], test_pred["pd_lr_calibrated"]))
    auc_xgb_raw = float(roc_auc_score(test_pred["default_flag"], test_pred["pd_xgb"]))
    auc_xgb_cal = float(roc_auc_score(test_pred["default_flag"], test_pred["pd_xgb_calibrated"]))
    auc_preserved = (abs(auc_lr_raw - auc_lr_cal) < 0.001
                     and abs(auc_xgb_raw - auc_xgb_cal) < 0.001)
    results.append(make(
        "10.12", "AUC preserved within ±0.001 by calibration",
        auc_preserved,
        f"LR Δ={auc_lr_cal - auc_lr_raw:+.2e}, GBM Δ={auc_xgb_cal - auc_xgb_raw:+.2e}",
        "|Δ| < 0.001",
    ))

    return results


# ---------- Task 11 (Step 10 LGD) ----------

def task_11_step10() -> list:
    results = []

    artifacts = [
        ("11.1", "data/loans_with_lgd.parquet", LOANS_LGD, "parquet"),
        ("11.2", "docs/lgd_lookup.csv", LGD_LOOKUP, "csv"),
        ("11.3", "docs/lgd_histogram.csv", LGD_HIST, "csv"),
        ("11.4", "docs/lgd_backtest.csv", LGD_BACKTEST, "csv"),
        ("11.5", "docs/lgd_sensitivity.csv", LGD_SENSITIVITY, "csv"),
        ("11.6", "docs/step10_methodology.md", STEP10_MD, "md"),
    ]
    all_files_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_files_ok = False

    if not all_files_ok:
        results.append(make("11.7", "Downstream Step 10 checks (skipped)",
                             False, "skipped — missing artifact above", "all checks ok"))
        return results

    macros = pd.read_parquet(LOANS_MACROS, columns=["id"])
    lgd = pd.read_parquet(LOANS_LGD, columns=["id", "lgd_predicted"])
    n_macros, n_lgd = len(macros), len(lgd)
    results.append(make(
        "11.7", "loans_with_lgd row count == loans_with_macros",
        n_macros == n_lgd, f"macros={n_macros:,}, lgd={n_lgd:,}", "equal",
    ))

    null_lgd = int(lgd["lgd_predicted"].isna().sum())
    in_range = lgd["lgd_predicted"].between(0, 1).all()
    results.append(make(
        "11.8", "lgd_predicted has no nulls and is in [0, 1]",
        null_lgd == 0 and in_range,
        f"nulls={null_lgd}, range=[{lgd['lgd_predicted'].min():.4f}, "
        f"{lgd['lgd_predicted'].max():.4f}]",
        "0 nulls, [0, 1]",
    ))

    test_pred = pd.read_parquet(TEST_PREDICTIONS)
    has_lgd = "lgd_predicted" in test_pred.columns
    null_in_tp = int(test_pred["lgd_predicted"].isna().sum()) if has_lgd else -1
    results.append(make(
        "11.9", "test_predictions has lgd_predicted, no nulls",
        has_lgd and null_in_tp == 0,
        f"has_col={has_lgd}, nulls={null_in_tp}",
        "present + 0 nulls",
    ))

    lookup = pd.read_csv(LGD_LOOKUP)
    no_null_estimates = lookup["lgd_estimate"].notna().all()
    results.append(make(
        "11.10", "lgd_lookup.csv has non-null estimates",
        no_null_estimates and len(lookup) > 0,
        f"rows={len(lookup)}, nulls={int(lookup['lgd_estimate'].isna().sum())}",
        "non-empty, no nulls",
    ))

    backtest = pd.read_csv(LGD_BACKTEST)
    if "n" in backtest.columns and "observed" in backtest.columns and "predicted" in backtest.columns:
        obs_total = float((backtest["observed"] * backtest["n"]).sum() / backtest["n"].sum())
        pred_total = float((backtest["predicted"] * backtest["n"]).sum() / backtest["n"].sum())
        diff = abs(pred_total - obs_total)
        results.append(make(
            "11.11", "Validation predicted ≈ observed within ±0.05",
            diff <= 0.05,
            f"observed={obs_total:.4f}, predicted={pred_total:.4f}, |diff|={diff:.4f}",
            "≤ 0.05",
        ))
    else:
        results.append(make("11.11", "Validation predicted ≈ observed within ±0.05",
                             False, "backtest CSV missing required columns",
                             "n + observed + predicted columns"))

    return results


# ---------- Task 12 (Step 11 EAD) ----------

def task_12_step11() -> list:
    results = []

    artifacts = [
        ("12.1", "data/loans_with_ead.parquet", LOANS_EAD, "parquet"),
        ("12.2", "docs/ead_histogram.csv", EAD_HIST, "csv"),
        ("12.3", "docs/ead_months_remaining_distribution.csv", EAD_MONTHS_DIST, "csv"),
        ("12.4", "docs/ead_status_breakdown.csv", EAD_STATUS_BREAK, "csv"),
        ("12.5", "docs/step11_methodology.md", STEP11_MD, "md"),
    ]
    all_files_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_files_ok = False

    if not all_files_ok:
        results.append(make("12.6", "Downstream Step 11 checks (skipped)",
                             False, "skipped — missing artifact above", "all checks ok"))
        return results

    macros = pd.read_parquet(LOANS_LGD, columns=["id"])
    ead = pd.read_parquet(LOANS_EAD, columns=[
        "id", "ead_12m", "ead_at_month_12", "ead_lifetime_undiscounted_total",
        "ead_lifetime_discounted_total", "months_remaining",
        "out_prncp", "funded_amnt",
    ])
    results.append(make(
        "12.6", "loans_with_ead row count == loans_with_lgd",
        len(macros) == len(ead),
        f"lgd={len(macros):,}, ead={len(ead):,}", "equal",
    ))

    needed_cols = {"ead_12m", "ead_at_month_12", "ead_lifetime_undiscounted_total",
                   "ead_lifetime_discounted_total", "months_remaining"}
    has_cols = needed_cols.issubset(set(ead.columns))
    nulls = int(ead[list(needed_cols)].isna().sum().sum())
    results.append(make(
        "12.7", "EAD columns present and non-null",
        has_cols and nulls == 0,
        f"missing={sorted(needed_cols - set(ead.columns))}, nulls={nulls}",
        "all present, 0 nulls",
    ))

    n_neg = int((ead["ead_12m"] < 0).sum())
    results.append(make(
        "12.8", "ead_12m ≥ 0 for all loans",
        n_neg == 0, f"negative={n_neg}", "0",
    ))

    full_ead = pd.read_parquet(LOANS_EAD, columns=[
        "id", "ead_lifetime_path", "discount_factors", "months_remaining",
    ])
    sample = full_ead.sample(n=min(2000, len(full_ead)), random_state=42)
    bad_path_len = 0
    bad_disc_len = 0
    for _, r in sample.iterrows():
        if len(r["ead_lifetime_path"]) != int(r["months_remaining"]):
            bad_path_len += 1
        if len(r["discount_factors"]) != int(r["months_remaining"]):
            bad_disc_len += 1
    results.append(make(
        "12.9", "ead_lifetime_path length == months_remaining (2K sample)",
        bad_path_len == 0,
        f"bad_path_len={bad_path_len}", "0",
    ))
    results.append(make(
        "12.10", "discount_factors length == months_remaining (2K sample)",
        bad_disc_len == 0,
        f"bad_disc_len={bad_disc_len}", "0",
    ))

    inactive = full_ead[full_ead["months_remaining"] == 0].sample(
        n=min(500, (full_ead["months_remaining"] == 0).sum()), random_state=42)
    bad_inactive = sum(1 for _, r in inactive.iterrows()
                        if len(r["ead_lifetime_path"]) != 0)
    results.append(make(
        "12.11", "Inactive loans have empty path",
        bad_inactive == 0, f"non_empty={bad_inactive}", "0",
    ))

    total_ead_12m = float(ead["ead_12m"].sum())
    total_funded = float(ead["funded_amnt"].sum())
    ratio = total_ead_12m / total_funded if total_funded else 0.0
    in_band = 0.05 <= ratio <= 0.95
    results.append(make(
        "12.12", "Aggregate ratio EAD_12m / funded in [0.05, 0.95]",
        in_band, f"ratio={ratio:.4f}", "[0.05, 0.95]",
    ))

    test_pred = pd.read_parquet(TEST_PREDICTIONS)
    needed_in_tp = {"ead_12m", "ead_lifetime_discounted_total", "months_remaining"}
    has_in_tp = needed_in_tp.issubset(set(test_pred.columns))
    nulls_in_tp = (int(test_pred[list(needed_in_tp)].isna().sum().sum())
                    if has_in_tp else -1)
    results.append(make(
        "12.13", "test_predictions has new EAD columns, no nulls",
        has_in_tp and nulls_in_tp == 0,
        f"has_cols={has_in_tp}, nulls={nulls_in_tp}",
        "present + 0 nulls",
    ))

    return results


# ---------- Task 13 (Step 12 ECL) ----------

def task_13_step12() -> list:
    results = []

    artifacts = [
        ("13.1", "data/loans_with_ecl.parquet", LOANS_ECL, "parquet"),
        ("13.2", "docs/ecl_headline.json", ECL_HEADLINE, "json"),
        ("13.3", "docs/ecl_by_stage.csv", ECL_BY_STAGE, "csv"),
        ("13.4", "docs/ecl_by_grade.csv", ECL_BY_GRADE, "csv"),
        ("13.5", "docs/ecl_by_vintage.csv", ECL_BY_VINTAGE, "csv"),
        ("13.6", "docs/ecl_by_purpose.csv", ECL_BY_PURPOSE, "csv"),
        ("13.7", "docs/step12_methodology.md", STEP12_MD, "md"),
    ]
    all_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_ok = False
    if not all_ok:
        results.append(make("13.8", "Downstream Step 12 checks (skipped)",
                             False, "skipped", "all files present"))
        return results

    macros = pd.read_parquet(LOANS_EAD, columns=["id"])
    ecl = pd.read_parquet(LOANS_ECL, columns=[
        "id", "pd_lifetime", "pd_12m", "ecl_12m", "ecl_lifetime",
        "ifrs9_stage", "ecl_total", "lgd_predicted",
        "ead_lifetime_undiscounted_total", "default_flag",
        "months_remaining",
    ])
    results.append(make(
        "13.8", "loans_with_ecl row count == loans_with_ead",
        len(macros) == len(ecl),
        f"ead={len(macros):,}, ecl={len(ecl):,}", "equal",
    ))

    new_cols = ["pd_lifetime", "pd_12m", "ecl_12m", "ecl_lifetime",
                "ifrs9_stage", "ecl_total"]
    nulls = int(ecl[new_cols].isna().sum().sum())
    n_neg = int((ecl[["ecl_12m", "ecl_lifetime", "ecl_total"]] < 0).sum().sum())
    results.append(make(
        "13.9", "ECL columns present, non-null, non-negative",
        nulls == 0 and n_neg == 0,
        f"nulls={nulls}, negatives={n_neg}", "0 / 0",
    ))

    headline = json.loads(ECL_HEADLINE.read_text())
    cited_total = float(headline["total_ecl"])
    actual_total = float(ecl["ecl_total"].sum())
    diff = abs(cited_total - actual_total)
    results.append(make(
        "13.10", "Headline JSON total_ecl matches parquet sum",
        diff < 1.0,
        f"json=${cited_total:,.2f}, parquet=${actual_total:,.2f}, |diff|=${diff:.4f}",
        "|diff| < $1",
    ))

    stage_sum = float(ecl.groupby("ifrs9_stage")["ecl_total"].sum().sum())
    diff2 = abs(stage_sum - actual_total)
    results.append(make(
        "13.11", "Σ stage ECL == total ECL",
        diff2 < 1.0, f"|diff|=${diff2:.4f}", "|diff| < $1",
    ))

    long_loans = ecl[ecl["months_remaining"] > 12]
    bad_inv = int((long_loans["ecl_12m"] > long_loans["ecl_lifetime"] + 1.0).sum())
    results.append(make(
        "13.12", "ecl_12m ≤ ecl_lifetime (long loans)",
        bad_inv == 0, f"violations={bad_inv}", "0",
    ))

    max_loss = ecl["lgd_predicted"] * ecl["ead_lifetime_undiscounted_total"]
    bad_max = int((ecl["ecl_total"] > max_loss + 1.0).sum())
    results.append(make(
        "13.13", "ECL ≤ EAD × LGD invariant",
        bad_max == 0, f"violations={bad_max}", "0",
    ))

    test_pred = pd.read_parquet(TEST_PREDICTIONS)
    needed = {"pd_lifetime", "pd_12m", "ecl_12m", "ecl_lifetime",
              "ifrs9_stage", "ecl_total"}
    has_cols = needed.issubset(set(test_pred.columns))
    null_in_tp = (int(test_pred[list(needed)].isna().sum().sum())
                   if has_cols else -1)
    results.append(make(
        "13.14", "test_predictions has Step 12 columns, no nulls",
        has_cols and null_in_tp == 0,
        f"has={has_cols}, nulls={null_in_tp}", "present + 0 nulls",
    ))

    return results


# ---------- Task 14 (Step 13 Overlay) ----------

def task_14_step13() -> list:
    results = []
    artifacts = [
        ("14.1", "data/loans_with_ecl_overlay.parquet", LOANS_OVERLAY, "parquet"),
        ("14.2", "docs/ecl_overlay_headline.json", OVERLAY_HEADLINE, "json"),
        ("14.3", "docs/ecl_overlay_by_stage.csv", OVERLAY_BY_STAGE, "csv"),
        ("14.4", "docs/ecl_overlay_by_grade.csv", OVERLAY_BY_GRADE, "csv"),
        ("14.5", "docs/ecl_overlay_by_vintage.csv", OVERLAY_BY_VINTAGE, "csv"),
        ("14.6", "docs/step13_methodology.md", STEP13_MD, "md"),
    ]
    all_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_ok = False
    if not all_ok:
        results.append(make("14.7", "Downstream Step 13 checks (skipped)",
                             False, "skipped", "all files present"))
        return results

    overlay = pd.read_parquet(LOANS_OVERLAY, columns=[
        "id", "ifrs9_stage", "ecl_total",
        "pd_lifetime_baseline", "pd_lifetime_adverse", "pd_lifetime_severe",
        "pd_12m_baseline", "pd_12m_adverse", "pd_12m_severe",
        "ecl_total_baseline", "ecl_total_adverse", "ecl_total_severe",
        "ecl_final",
    ])
    macros = pd.read_parquet(LOANS_ECL, columns=["id"])
    results.append(make(
        "14.7", "loans_with_ecl_overlay row count == loans_with_ecl",
        len(overlay) == len(macros),
        f"overlay={len(overlay):,}, ecl={len(macros):,}", "equal",
    ))

    new_cols = ["pd_lifetime_baseline", "pd_lifetime_adverse", "pd_lifetime_severe",
                "pd_12m_baseline", "pd_12m_adverse", "pd_12m_severe",
                "ecl_total_baseline", "ecl_total_adverse", "ecl_total_severe", "ecl_final"]
    nulls = int(overlay[new_cols].isna().sum().sum())
    n_neg = int((overlay[new_cols] < 0).sum().sum())
    results.append(make(
        "14.8", "Overlay columns non-null and non-negative",
        nulls == 0 and n_neg == 0,
        f"nulls={nulls}, negatives={n_neg}", "0 / 0",
    ))

    headline = json.loads(OVERLAY_HEADLINE.read_text())
    weights_sum = float(headline["scenario_weights_sum"])
    results.append(make(
        "14.9", "Scenario weights sum to 1.0",
        abs(weights_sum - 1.0) < 1e-9, f"sum={weights_sum}", "1.0",
    ))

    base_total = float(overlay["ecl_total_baseline"].sum())
    step12 = float(overlay["ecl_total"].sum())
    diff = abs(base_total - step12)
    results.append(make(
        "14.10", "Baseline ECL matches Step 12 within $1",
        diff < 1.0, f"|diff|=${diff:.4f}", "< $1",
    ))

    final_total = float(overlay["ecl_final"].sum())
    sev_total = float(overlay["ecl_total_severe"].sum())
    # Direction-agnostic ordering: Step 13 §3 documents that the LC
    # underwriting-reaction effect inverts the conventional baseline→severe
    # direction in this dataset. Assert ordered totals (in either direction).
    order_normal = base_total <= final_total <= sev_total
    order_inverted = sev_total <= final_total <= base_total
    range_ok = order_normal or order_inverted
    direction = ("normal" if order_normal else
                  ("inverted (LC underwriting-reaction)" if order_inverted else
                   "non-monotonic"))
    results.append(make(
        "14.11", "Aggregate base/final/severe ordered (monotonic, either direction)",
        range_ok,
        f"base=${base_total:,.0f}, final=${final_total:,.0f}, sev=${sev_total:,.0f} — {direction}",
        "ordered (direction documented)",
    ))

    test_pred = pd.read_parquet(TEST_PREDICTIONS)
    needed = {"ecl_total_baseline", "ecl_total_adverse", "ecl_total_severe", "ecl_final"}
    has_cols = needed.issubset(set(test_pred.columns))
    null_in_tp = (int(test_pred[list(needed)].isna().sum().sum()) if has_cols else -1)
    results.append(make(
        "14.12", "test_predictions has overlay columns, no nulls",
        has_cols and null_in_tp == 0,
        f"has={has_cols}, nulls={null_in_tp}", "present + 0 nulls",
    ))

    return results


# ---------- Task 15 (Step 14 Validation) ----------

def task_15_step14() -> list:
    results = []
    artifacts = [
        ("15.1", "validation_discrimination.json", VAL_DISCRIM, "json"),
        ("15.2", "validation_calibration.json", VAL_CALIB, "json"),
        ("15.3", "validation_auc_by_vintage.csv", VAL_AUC_VINTAGE, "csv"),
        ("15.4", "validation_auc_by_grade.csv", VAL_AUC_GRADE, "csv"),
        ("15.5", "validation_gain_curve.csv", VAL_GAIN, "csv"),
        ("15.6", "validation_reliability_test.csv", VAL_RELIAB, "csv"),
        ("15.7", "validation_sicr_sensitivity.csv", VAL_SICR, "csv"),
        ("15.8", "validation_overlay_weights.csv", VAL_WEIGHTS, "csv"),
        ("15.9", "validation_regulatory_overlay.json", VAL_REG_OVERLAY, "json"),
        ("15.10", "validation_psi_over_time.csv", VAL_PSI, "csv"),
        ("15.11", "validation_single_feature_stress.csv", VAL_FEATURE_STRESS, "csv"),
        ("15.12", "validation_stage_migration.csv", VAL_STAGE_MIGRATION, "csv"),
        ("15.13", "final_validation_report.md", FINAL_REPORT, "md"),
    ]
    all_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_ok = False
    if not all_ok:
        results.append(make("15.14", "Final report content checks (skipped)",
                             False, "skipped", "all files ok"))
        return results

    rep = FINAL_REPORT.read_text()
    required_headers = [
        "## 1. Executive Summary",
        "## 2. Headline numbers",
        "## 3. Discrimination validation",
        "## 4. Calibration validation",
        "## 5. Sensitivity analyses",
        "## 6. The macro-overlay finding",
        "## 7. Methodological assumptions",
        "## 8. Limitations",
        "## 9. Appendix",
        "## 10. Sign-off",
    ]
    missing_headers = [h for h in required_headers if h not in rep]
    results.append(make(
        "15.14", "Final report has all 10 sections",
        not missing_headers,
        f"missing={missing_headers}" if missing_headers else "all 10 sections present",
        "all 10 headers",
    ))

    headline = json.loads(ECL_HEADLINE.read_text())
    overlay_h = json.loads(OVERLAY_HEADLINE.read_text())
    reg_h = json.loads(VAL_REG_OVERLAY.read_text())
    h1 = f"${headline['total_ecl']:,.0f}"
    h2 = f"${overlay_h['final_ecl']:,.0f}"
    h3 = f"${reg_h['weighted_final_ecl']:,.0f}"
    in_report = (h1 in rep and h2 in rep and h3 in rep)
    results.append(make(
        "15.15", "Three headline numbers present in final report",
        in_report,
        f"baseline={h1 in rep}, overlay={h2 in rep}, regulatory={h3 in rep}",
        "all three present",
    ))

    reg_ecl = float(reg_h["weighted_final_ecl"])
    base_ecl = float(reg_h["step12_baseline_ecl"])
    results.append(make(
        "15.16", "Regulatory overlay ECL > baseline (validates plausibility)",
        reg_ecl > base_ecl,
        f"reg=${reg_ecl:,.0f} > base=${base_ecl:,.0f} = {reg_ecl > base_ecl}",
        "reg > base",
    ))

    sicr_df = pd.read_csv(VAL_SICR)
    sicr_cols_ok = {"threshold_rule", "stage2_share_pct",
                     "total_ecl_baseline", "change_from_2x_pct"}.issubset(sicr_df.columns)
    results.append(make(
        "15.17", "validation_sicr_sensitivity.csv has expected columns",
        sicr_cols_ok,
        f"cols={sorted(sicr_df.columns)}", "expected columns",
    ))

    return results


# ---------- Task 16 (Step 15 Dashboard) ----------

def task_16_step15() -> list:
    results = []
    artifacts = [
        ("16.1", "data/dashboard/loans_summary.csv", DASH_LOANS, "csv"),
        ("16.2", "data/dashboard/headline_metrics.csv", DASH_HEADLINE, "csv"),
        ("16.3", "data/dashboard/discrimination_metrics.csv", DASH_DISCRIM, "csv"),
        ("16.4", "data/dashboard/calibration_table.csv", DASH_CALIB, "csv"),
        ("16.5", "data/dashboard/sensitivity_table.csv", DASH_SENS, "csv"),
        ("16.6", "docs/dashboard_spec.md", DASHBOARD_SPEC, "md"),
    ]
    all_ok = True
    for cid, name, path, kind in artifacts:
        ok, value, expected = _check_file(path, kind)
        results.append(make(cid, name, ok, value, expected))
        if not ok:
            all_ok = False
    if not all_ok:
        results.append(make("16.7", "Downstream Step 15 checks (skipped)",
                             False, "skipped", "all files ok"))
        return results

    overlay_n = pd.read_parquet(LOANS_OVERLAY, columns=["id"]).shape[0]
    loans_n = sum(1 for _ in DASH_LOANS.open()) - 1  # minus header
    results.append(make(
        "16.7", "loans_summary.csv row count matches loans_with_ecl_overlay",
        loans_n == overlay_n,
        f"dashboard={loans_n:,}, parquet={overlay_n:,}", "equal",
    ))

    headline = pd.read_csv(DASH_HEADLINE)
    versions_ok = set(headline["version"].unique()) >= {"baseline", "data_overlay", "regulatory"}
    results.append(make(
        "16.8", "headline_metrics has rows for all three versions",
        versions_ok,
        f"versions={sorted(headline['version'].unique())}",
        "{'baseline', 'data_overlay', 'regulatory'} ⊆ versions",
    ))

    spec = DASHBOARD_SPEC.read_text()
    sections_ok = all(s in spec for s in [
        "Section 1 — Setup", "Section 2 — Page 1", "Section 3 — Page 2",
        "Section 4 — Page 3", "Section 5 — Page 4",
        "Section 6 — Theme", "Section 7 — Final touches",
    ])
    results.append(make(
        "16.9", "dashboard_spec.md has all 7 sections",
        sections_ok,
        f"all sections present" if sections_ok else "missing sections",
        "all 7 sections",
    ))

    return results


# ---------- Task 17 (Project Summary) ----------

def task_17_summary() -> list:
    results = []
    ok, value, expected = _check_file(PROJECT_SUMMARY, "md")
    results.append(make("17.1", "docs/project_summary.md exists, non-empty",
                         ok, value, expected))
    if not ok:
        return results

    md = PROJECT_SUMMARY.read_text()
    required = [f"## Section {i} —" for i in range(1, 13)]
    missing = [h for h in required if h not in md]
    results.append(make(
        "17.2", "All 12 H2 section headers present",
        not missing,
        f"missing={missing}" if missing else "all 12 sections",
        "12 H2 headers",
    ))

    h_base = json.loads(ECL_HEADLINE.read_text())
    h_overlay = json.loads(OVERLAY_HEADLINE.read_text())
    h_reg = json.loads(VAL_REG_OVERLAY.read_text())
    headlines = [
        f"${float(h_base['total_ecl']):,.0f}",
        f"${float(h_overlay['final_ecl']):,.0f}",
        f"${float(h_reg['weighted_final_ecl']):,.0f}",
    ]
    missing_h = [h for h in headlines if h not in md]
    results.append(make(
        "17.3", "Three headline numbers present in summary",
        not missing_h,
        f"missing={missing_h}" if missing_h else "all 3 present",
        "3 headlines",
    ))

    return results


def task_18_dossier() -> list:
    results = []
    ok, value, expected = _check_file(FINAL_DOSSIER, "md")
    results.append(make("18.1", "docs/final_project_dossier.md exists, non-empty",
                         ok, value, expected))
    if not ok:
        return results

    md = FINAL_DOSSIER.read_text()

    required = [f"## Section {i} —" for i in range(15)]
    missing = [h for h in required if h not in md]
    results.append(make(
        "18.2", "All 15 H2 section headers present (Section 0–14)",
        not missing,
        f"missing={missing}" if missing else "all 15 sections",
        "15 H2 headers",
    ))

    word_count = len(md.split())
    in_range = 5_000 <= word_count <= 10_000
    results.append(make(
        "18.3", "Word count within 5,000–10,000 range",
        in_range,
        f"{word_count:,} words",
        "5,000 ≤ wc ≤ 10,000",
    ))

    h_base = json.loads(ECL_HEADLINE.read_text())
    h_overlay = json.loads(OVERLAY_HEADLINE.read_text())
    h_reg = json.loads(VAL_REG_OVERLAY.read_text())
    headlines = [
        f"${float(h_base['total_ecl']):,.0f}",
        f"${float(h_overlay['final_ecl']):,.0f}",
        f"${float(h_reg['weighted_final_ecl']):,.0f}",
    ]
    missing_h = [h for h in headlines if h not in md]
    results.append(make(
        "18.4", "Three headline numbers present in dossier",
        not missing_h,
        f"missing={missing_h}" if missing_h else "all 3 present",
        "3 headlines",
    ))

    n_verified = md.count("[verified ✓ — source:")
    results.append(make(
        "18.5", "≥50 inline verification tags present",
        n_verified >= 50,
        f"{n_verified} tags",
        "≥50",
    ))

    n_failed = md.count("[VERIFY FAILED")
    results.append(make(
        "18.6", "Zero VERIFY FAILED tags in body",
        n_failed == 0,
        f"{n_failed} failed",
        "0",
    ))

    appendix_present = "## Section 13 —" in md and "| # | Claim |" in md
    results.append(make(
        "18.7", "Verification appendix table present (Section 13)",
        appendix_present,
        "appendix table found" if appendix_present else "appendix table missing",
        "appendix table in Section 13",
    ))

    file_index_present = "## Section 12 —" in md
    file_index_categories = ["Cleaned data", "Models", "Validation", "Source code"]
    cats_missing = [c for c in file_index_categories if c not in md]
    results.append(make(
        "18.8", "File index (Section 12) present with category groupings",
        file_index_present and not cats_missing,
        f"missing={cats_missing}" if cats_missing else "all 4 categories present",
        "Cleaned data / Models / Validation / Source code",
    ))

    return results


# ---------- Reporting ----------

def emit_console(all_results: list[tuple[str, list]]) -> None:
    for task_name, checks in all_results:
        print(f"\n=== {task_name} ===")
        for cid, name, ok, value, expected in checks:
            tag = "[ ✓ PASS ]" if ok else "[ ✗ FAIL ]"
            label = f"{cid} {name}"
            if ok:
                print(f"  {tag}  {label:<60s} ({value})")
            else:
                print(f"  {tag}  {label:<60s} ({value}; expected {expected})")


def emit_summary(all_results: list[tuple[str, list]]) -> None:
    total = passed = 0
    failures = []
    for _, checks in all_results:
        for cid, name, ok, _, _ in checks:
            total += 1
            if ok:
                passed += 1
            else:
                failures.append(f"{cid} {name}")
    print("\n=== Validation Summary ===")
    print(f"  Total checks: {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {total - passed}")
    if failures:
        print("  Failures:")
        for f in failures:
            print(f"    - {f}")


def write_report(all_results: list[tuple[str, list]]) -> None:
    parts: list[str] = [
        "# Pipeline Validation Report — Steps 7 and 8\n",
        f"\nRun timestamp: `{datetime.now().isoformat(timespec='seconds')}`\n",
    ]
    total = passed = 0
    failures: list[tuple[str, str, str, str]] = []
    for task_name, checks in all_results:
        parts.append(f"\n## {task_name}\n\n")
        parts.append("| Status | ID | Check | Value | Expected |\n")
        parts.append("|---|---|---|---|---|\n")
        for cid, name, ok, value, expected in checks:
            total += 1
            if ok:
                passed += 1
                tag = "✓ PASS"
            else:
                tag = "✗ FAIL"
                failures.append((cid, name, value, expected))
            value_md = str(value).replace("|", "\\|")
            expected_md = str(expected).replace("|", "\\|")
            parts.append(f"| {tag} | {cid} | {name} | {value_md} | {expected_md} |\n")

    parts.append("\n## Validation Summary\n\n")
    parts.append(f"- Total checks: **{total}**\n")
    parts.append(f"- Passed: **{passed}**\n")
    parts.append(f"- Failed: **{total - passed}**\n")

    if failures:
        parts.append("\n## ⚠️ Action Required\n\n")
        parts.append("The following failures must be resolved before proceeding to Step 9:\n\n")
        for cid, name, value, expected in failures:
            parts.append(f"- **{cid} {name}** — got `{value}`, expected `{expected}`\n")

    REPORT.write_text("".join(parts))


# ---------- Main ----------

def main() -> None:
    print("=== Pipeline Validation — Steps 7 and 8 ===")
    print(f"Run: {datetime.now().isoformat(timespec='seconds')}")

    file_results, files_ok = task_1_files()

    if not files_ok:
        all_results = [("Task 1 — File existence and integrity", file_results)]
        emit_console(all_results)
        emit_summary(all_results)
        write_report(all_results)
        print("\nFATAL: Task 1 failed; downstream tasks skipped.")
        sys.exit(1)

    print("\nLoading parquets and JSON…")
    loans_ready = pd.read_parquet(LOANS_READY)
    loans_macros = pd.read_parquet(LOANS_MACROS)
    fc = json.loads(FC_JSON.read_text())
    step7_md = STEP7_MD.read_text()
    step8_md = STEP8_MD.read_text()

    all_results = [
        ("Task 1 — File existence and integrity", file_results),
        ("Task 2 — Schema validation", task_2_schema(loans_ready, loans_macros)),
        ("Task 3 — Data quality re-verification", task_3_quality(loans_macros)),
        ("Task 4 — Feature classification JSON validity", task_4_classification(fc, loans_macros)),
        ("Task 5 — Distributional sanity", task_5_distributional(loans_macros)),
        ("Task 6 — Cross-deliverable consistency",
         task_6_consistency(loans_ready, loans_macros, step7_md, step8_md)),
        ("Task 7 — Modeling-readiness check", task_7_readiness(loans_macros, fc)),
        ("Task 8 — Step 9a artifacts (WoE binning)", task_8_step9a()),
        ("Task 9 — Step 9b artifacts (PD models)", task_9_step9b()),
        ("Task 10 — Step 9c artifacts (Platt calibration)", task_10_step9c()),
        ("Task 11 — Step 10 artifacts (LGD)", task_11_step10()),
        ("Task 12 — Step 11 artifacts (EAD)", task_12_step11()),
        ("Task 13 — Step 12 artifacts (ECL)", task_13_step12()),
        ("Task 14 — Step 13 artifacts (Overlay)", task_14_step13()),
        ("Task 15 — Step 14 artifacts (Validation pack)", task_15_step14()),
        ("Task 16 — Step 15 artifacts (Dashboard data)", task_16_step15()),
        ("Task 17 — Unified project summary", task_17_summary()),
        ("Task 18 — Final project dossier", task_18_dossier()),
    ]

    emit_console(all_results)
    emit_summary(all_results)
    write_report(all_results)

    any_failed = any(not ok for _, checks in all_results for _, _, ok, _, _ in checks)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
