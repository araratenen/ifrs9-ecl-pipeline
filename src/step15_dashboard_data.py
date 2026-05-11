"""
Step 15 — Power BI Dashboard data layer.

Produces five denormalized CSVs and a markdown specification, optimized for
Power BI Desktop's import workflow:

  - data/dashboard/loans_summary.csv         per-loan facts (1.18M rows)
  - data/dashboard/headline_metrics.csv       headline KPIs across 3 versions × 5 breakdowns
  - data/dashboard/discrimination_metrics.csv AUC/Gini/KS by vintage and grade
  - data/dashboard/calibration_table.csv      decile / by-grade / by-vintage reliability
  - data/dashboard/sensitivity_table.csv      SICR + weights + single-feature stress
  - docs/dashboard_spec.md                    Power BI build instructions

The .pbix is built manually in Power BI Desktop by following dashboard_spec.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/ostappolukainen/Desktop/ProjRED")
LOANS_OVERLAY = ROOT / "data" / "loans_with_ecl_overlay.parquet"

ECL_HEADLINE = ROOT / "docs" / "ecl_headline.json"
OVERLAY_HEADLINE = ROOT / "docs" / "ecl_overlay_headline.json"
REG_OVERLAY = ROOT / "docs" / "validation_regulatory_overlay.json"
DISCRIM_JSON = ROOT / "docs" / "validation_discrimination.json"
CALIB_JSON = ROOT / "docs" / "validation_calibration.json"
RELIAB_CSV = ROOT / "docs" / "validation_reliability_test.csv"
CALIB_GRADE_CSV = ROOT / "docs" / "validation_calibration_by_grade.csv"
CALIB_VINTAGE_CSV = ROOT / "docs" / "validation_calibration_by_vintage.csv"
SICR_CSV = ROOT / "docs" / "validation_sicr_sensitivity.csv"
WEIGHTS_CSV = ROOT / "docs" / "validation_overlay_weights.csv"
FEATURE_STRESS_CSV = ROOT / "docs" / "validation_single_feature_stress.csv"

DASHBOARD_DIR = ROOT / "data" / "dashboard"
LOANS_SUMMARY = DASHBOARD_DIR / "loans_summary.csv"
HEADLINE_METRICS = DASHBOARD_DIR / "headline_metrics.csv"
DISCRIM_METRICS = DASHBOARD_DIR / "discrimination_metrics.csv"
CALIB_TABLE = DASHBOARD_DIR / "calibration_table.csv"
SENSITIVITY_TABLE = DASHBOARD_DIR / "sensitivity_table.csv"
DASHBOARD_SPEC = ROOT / "docs" / "dashboard_spec.md"
VALIDATOR = ROOT / "src" / "validate_pipeline_steps_7_8.py"

REG_COEF_UNRATE = 0.18
REG_COEF_HPI = 0.05


def main() -> None:
    rec: dict = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    task_1_environment()
    overlay = task_2_loans_summary(rec)
    task_3_headline_metrics(overlay, rec)
    task_4_discrimination_metrics(rec)
    task_5_calibration_table(rec)
    task_6_sensitivity_table(rec)
    task_7_dashboard_spec()
    rc = task_run_validator()

    print("\n=== Done ===")
    for p in (LOANS_SUMMARY, HEADLINE_METRICS, DISCRIM_METRICS,
              CALIB_TABLE, SENSITIVITY_TABLE, DASHBOARD_SPEC):
        size = p.stat().st_size / 1024**2 if p.exists() else 0
        print(f"  {p}  ({size:.1f} MB)")
    print(f"\nValidator: {'PASS' if rc == 0 else f'FAIL (exit {rc})'}")

    print("\n=== Headline numbers (final reproduction) ===")
    base = json.loads(ECL_HEADLINE.read_text())["total_ecl"]
    overlay_h = json.loads(OVERLAY_HEADLINE.read_text())["final_ecl"]
    reg_h = json.loads(REG_OVERLAY.read_text())["weighted_final_ecl"]
    print(f"  Step 12 baseline:           ${base:,.0f}")
    print(f"  Step 13 data-driven overlay: ${overlay_h:,.0f}")
    print(f"  Step 14 regulatory overlay:  ${reg_h:,.0f}  ← recommended for IFRS 9")


def task_1_environment() -> None:
    print("=== Task 1: Environment ===")
    for mod in ("pandas", "numpy", "pyarrow"):
        try:
            __import__(mod)
        except ImportError:
            sys.exit(f"ERROR: missing {mod}")
    inputs = [LOANS_OVERLAY, ECL_HEADLINE, OVERLAY_HEADLINE, REG_OVERLAY,
              DISCRIM_JSON, CALIB_JSON, RELIAB_CSV, SICR_CSV, WEIGHTS_CSV,
              FEATURE_STRESS_CSV]
    for f in inputs:
        if not f.exists():
            sys.exit(f"ERROR: missing {f}")
    print("dependencies + inputs: OK")


# ---------- Task 2 ----------

def task_2_loans_summary(rec: dict) -> pd.DataFrame:
    print("\n=== Task 2: Build loans_summary.csv ===")
    overlay = pd.read_parquet(LOANS_OVERLAY)
    print(f"loaded overlay: {overlay.shape}")

    print("computing per-loan regulatory ECL (3 scenarios × lifetime ECL)...")
    overlay["ecl_regulatory"] = _compute_regulatory_per_loan(overlay)

    overlay["issue_year"] = overlay["issue_d"].dt.year
    overlay["vintage_bucket"] = pd.cut(
        overlay["issue_year"],
        bins=[-1, 2013, 2014, 2015, 2016, 9999],
        labels=["pre-2014", "2014", "2015", "2016", "2017+"],
    ).astype(str)

    cols = [
        "id", "issue_d", "issue_year", "vintage_bucket",
        "grade", "sub_grade", "purpose", "term_months",
        "home_ownership", "addr_state",
        "funded_amnt", "loan_amnt", "int_rate", "dti",
        "fico_range_low", "annual_inc",
        "default_flag", "loan_status",
        "pd_lifetime", "pd_12m",
        "pd_lifetime_baseline", "pd_lifetime_adverse", "pd_lifetime_severe",
        "lgd_predicted", "ead_12m", "ead_lifetime_discounted_total",
        "months_remaining",
        "ifrs9_stage", "ecl_12m", "ecl_lifetime", "ecl_total",
        "ecl_total_baseline", "ecl_total_adverse", "ecl_total_severe",
        "ecl_final", "ecl_regulatory",
    ]
    out = overlay[cols].copy()
    out["issue_d"] = out["issue_d"].dt.strftime("%Y-%m-%d")
    out.to_csv(LOANS_SUMMARY, index=False)
    size_mb = LOANS_SUMMARY.stat().st_size / 1024 ** 2
    print(f"wrote: {LOANS_SUMMARY} ({size_mb:.1f} MB), {len(out):,} rows × {len(cols)} cols")
    rec["n_loans"] = len(out)
    return overlay


def _compute_regulatory_per_loan(overlay: pd.DataFrame) -> np.ndarray:
    pd_baseline = overlay["pd_lifetime"].values.astype(float)
    pd_clipped = np.clip(pd_baseline, 1e-9, 1 - 1e-9)
    log_odds = np.log(pd_clipped / (1 - pd_clipped))

    Ts = overlay["months_remaining"].astype(int).values
    lgd = overlay["lgd_predicted"].values
    ead_12m_arr = overlay["ead_12m"].values
    paths = overlay["ead_lifetime_path"].values
    discs = overlay["discount_factors"].values
    df_avg_12m = _df_avg_12m(discs)
    stage = overlay["ifrs9_stage"].values

    months_rem_safe = np.where(Ts > 0, Ts.astype(float), np.nan)
    stage3_ecl = np.where(
        np.isnan(months_rem_safe), 0.0,
        lgd * overlay["ead_lifetime_discounted_total"].values / months_rem_safe,
    )

    scenarios = [
        ("baseline", 0.0, 0.0, 0.50),
        ("adverse", 3.0, -10.0, 0.30),
        ("severe", 5.0, -20.0, 0.20),
    ]
    final = np.zeros(len(overlay))
    for name, du, dh, w in scenarios:
        log_odds_scen = log_odds + REG_COEF_UNRATE * du + REG_COEF_HPI * abs(min(dh, 0))
        pd_scen = 1.0 / (1.0 + np.exp(-log_odds_scen))
        pd_12m_scen = _pd_12m(pd_scen, Ts)
        ecl_12m_scen = pd_12m_scen * lgd * ead_12m_arr * df_avg_12m
        ecl_lifetime_scen = _lifetime_ecl_array(pd_scen, lgd, paths, discs, Ts)
        ecl_total_scen = np.where(stage == 1, ecl_12m_scen,
                                    np.where(stage == 2, ecl_lifetime_scen, stage3_ecl))
        ecl_total_scen = np.where(Ts == 0, 0.0, ecl_total_scen)
        final += w * ecl_total_scen
    return final


def _pd_12m(pd_lt, T):
    out = np.zeros(len(pd_lt))
    active = T > 0
    pd_clipped = np.clip(pd_lt, 1e-9, 1 - 1e-9)
    horizon = np.minimum(12, T)
    monthly_hazard = np.zeros(len(pd_lt))
    monthly_hazard[active] = 1 - np.power(1 - pd_clipped[active], 1 / T[active])
    out[active] = 1 - np.power(1 - monthly_hazard[active], horizon[active])
    return out


def _lifetime_ecl_array(pd_lt, lgd, paths, discs, Ts):
    n = len(pd_lt)
    out = np.zeros(n)
    for i in range(n):
        T = int(Ts[i])
        if T == 0:
            continue
        p = float(pd_lt[i])
        if p <= 0:
            continue
        L = float(lgd[i])
        if L <= 0:
            continue
        if p >= 1.0:
            out[i] = L * float(np.sum(paths[i] * discs[i]))
            continue
        h = 1 - (1 - p) ** (1 / T)
        t_arr = np.arange(T)
        marg = h * np.power(1 - h, t_arr)
        out[i] = L * float(np.sum(marg * paths[i] * discs[i]))
    return out


def _df_avg_12m(discounts):
    out = np.zeros(len(discounts))
    for i, dv in enumerate(discounts):
        if len(dv) == 0:
            continue
        out[i] = float(np.mean(dv[:12]))
    return out


# ---------- Task 3 ----------

def task_3_headline_metrics(overlay: pd.DataFrame, rec: dict) -> None:
    print("\n=== Task 3: Build headline_metrics.csv ===")
    rows: list[dict] = []
    versions = [
        ("baseline", "ecl_total"),
        ("data_overlay", "ecl_final"),
        ("regulatory", "ecl_regulatory"),
    ]
    for ver, col in versions:
        rows.append(_metric_row(overlay, ver, "all", "all", col))

    for ver, col in versions:
        for stage in [1, 2, 3]:
            mask = overlay["ifrs9_stage"] == stage
            rows.append(_metric_row(overlay[mask], ver, "stage", str(stage), col))

    for ver, col in versions:
        for grade in sorted(overlay["grade"].dropna().unique()):
            mask = overlay["grade"] == grade
            rows.append(_metric_row(overlay[mask], ver, "grade", str(grade), col))

    for ver, col in versions:
        for vb in sorted(overlay["vintage_bucket"].dropna().unique()):
            mask = overlay["vintage_bucket"] == vb
            rows.append(_metric_row(overlay[mask], ver, "vintage_bucket", str(vb), col))

    for ver, col in versions:
        for purpose in sorted(overlay["purpose"].dropna().unique()):
            mask = overlay["purpose"] == purpose
            rows.append(_metric_row(overlay[mask], ver, "purpose", str(purpose), col))

    df = pd.DataFrame(rows)
    df.to_csv(HEADLINE_METRICS, index=False)
    print(f"wrote: {HEADLINE_METRICS} ({len(df)} rows)")
    print(df.head().to_string(index=False))
    rec["headline_rows"] = len(df)


def _metric_row(df: pd.DataFrame, version: str, breakdown: str,
                 segment: str, col: str) -> dict:
    if df.empty:
        return {
            "version": version, "breakdown": breakdown, "segment": segment,
            "total_ecl": 0.0, "total_funded": 0.0,
            "ecl_ratio_pct": 0.0, "n_loans": 0,
        }
    ecl = float(df[col].sum())
    funded = float(df["funded_amnt"].sum())
    return {
        "version": version,
        "breakdown": breakdown,
        "segment": segment,
        "total_ecl": round(ecl, 2),
        "total_funded": round(funded, 2),
        "ecl_ratio_pct": round(ecl / funded * 100, 4) if funded else 0.0,
        "n_loans": len(df),
    }


# ---------- Task 4 ----------

def task_4_discrimination_metrics(rec: dict) -> None:
    print("\n=== Task 4: Build discrimination_metrics.csv ===")
    payload = json.loads(DISCRIM_JSON.read_text())
    rows = [{
        "dimension": "aggregate", "segment": "all_test",
        "n_loans": payload["n"],
        "n_defaults": int(payload["default_rate"] * payload["n"]),
        "default_rate": round(payload["default_rate"], 6),
        "auc": payload["auc"], "gini": payload["gini"], "ks": payload["ks"],
    }]
    for r in payload["by_vintage"]:
        rows.append({
            "dimension": "vintage", "segment": str(r["issue_year"]),
            "n_loans": r["n"],
            "n_defaults": int(r["default_rate"] * r["n"]),
            "default_rate": round(r["default_rate"], 6),
            "auc": r["auc"], "gini": r["gini"], "ks": r["ks"],
        })
    for r in payload["by_grade"]:
        rows.append({
            "dimension": "grade", "segment": str(r["grade"]),
            "n_loans": r["n"],
            "n_defaults": int(r["default_rate"] * r["n"]),
            "default_rate": round(r["default_rate"], 6),
            "auc": r["auc"], "gini": r["gini"], "ks": r["ks"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(DISCRIM_METRICS, index=False)
    print(f"wrote: {DISCRIM_METRICS} ({len(df)} rows)")


# ---------- Task 5 ----------

def task_5_calibration_table(rec: dict) -> None:
    print("\n=== Task 5: Build calibration_table.csv ===")
    rel = pd.read_csv(RELIAB_CSV)
    rows = []
    for _, r in rel.iterrows():
        rows.append({
            "breakdown": "decile", "segment": "all",
            "decile_or_segment": int(r["decile"]),
            "predicted_pd": round(float(r["mean_pred_pd"]), 6),
            "observed_pd": round(float(r["observed_default_rate"]), 6),
            "n": int(r["n"]),
        })
    by_g = pd.read_csv(CALIB_GRADE_CSV)
    for _, r in by_g.iterrows():
        rows.append({
            "breakdown": "grade", "segment": str(r["grade"]),
            "decile_or_segment": str(r["grade"]),
            "predicted_pd": round(float(r["mean_pred_pd"]), 6),
            "observed_pd": round(float(r["observed_default_rate"]), 6),
            "n": int(r["n"]),
        })
    by_v = pd.read_csv(CALIB_VINTAGE_CSV)
    for _, r in by_v.iterrows():
        yr = r.get("issue_year") or r.iloc[0]
        rows.append({
            "breakdown": "vintage", "segment": str(int(yr)),
            "decile_or_segment": str(int(yr)),
            "predicted_pd": round(float(r["mean_pred_pd"]), 6),
            "observed_pd": round(float(r["observed_default_rate"]), 6),
            "n": int(r["n"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(CALIB_TABLE, index=False)
    print(f"wrote: {CALIB_TABLE} ({len(df)} rows)")


# ---------- Task 6 ----------

def task_6_sensitivity_table(rec: dict) -> None:
    print("\n=== Task 6: Build sensitivity_table.csv ===")
    rows = []
    sicr = pd.read_csv(SICR_CSV)
    for _, r in sicr.iterrows():
        rows.append({
            "analysis_type": "SICR_threshold",
            "scenario": str(r["threshold_rule"]),
            "total_ecl": round(float(r["total_ecl_baseline"]), 0),
            "change_from_baseline_pct": round(float(r["change_from_2x_pct"]), 4),
            "n_stage2": None,
            "stage2_share_pct": round(float(r["stage2_share_pct"]), 4),
        })
    weights = pd.read_csv(WEIGHTS_CSV)
    for _, r in weights.iterrows():
        rows.append({
            "analysis_type": "overlay_weights",
            "scenario": str(r["weight_set"]),
            "total_ecl": round(float(r["total_ecl_final"]), 0),
            "change_from_baseline_pct": round(float(r["change_from_50_30_20_pct"]), 4),
            "n_stage2": None, "stage2_share_pct": None,
        })
    feat = pd.read_csv(FEATURE_STRESS_CSV)
    for _, r in feat.iterrows():
        if "skip" in r and pd.notna(r.get("skip", None)):
            continue
        if pd.isna(r.get("ecl_under_stress", np.nan)):
            continue
        rows.append({
            "analysis_type": "single_feature_stress",
            "scenario": f"{r['feature']}={r['worst_value']}",
            "total_ecl": round(float(r["ecl_under_stress"]), 0),
            "change_from_baseline_pct": round(float(r["delta_pct"]), 4),
            "n_stage2": None, "stage2_share_pct": None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(SENSITIVITY_TABLE, index=False)
    print(f"wrote: {SENSITIVITY_TABLE} ({len(df)} rows)")


# ---------- Task 7 ----------

def task_7_dashboard_spec() -> None:
    print("\n=== Task 7: Build dashboard_spec.md ===")

    spec = """# Power BI Dashboard Specification — IFRS 9 ECL Model (LendingClub Consumer Portfolio)

**Audience.** A Big 4 audit/risk reviewer or interviewer with ~10 minutes of attention. Strong financial instincts; limited time for technical detail.
**Goal.** Communicate three headline ECL numbers, why they differ, and what drives them. Not a data dump.
**Scope.** Four pages, five backing tables, financial-reporting visual style.

---

## Section 1 — Setup

1. **Open Power BI Desktop** (latest version).
2. **Click "Get Data" → "Text/CSV".** Import each of these from `/Users/ostappolukainen/Desktop/ProjRED/data/dashboard/`:
   - `loans_summary.csv`
   - `headline_metrics.csv`
   - `discrimination_metrics.csv`
   - `calibration_table.csv`
   - `sensitivity_table.csv`
3. **In the "Model" view, define relationships:**
   - `loans_summary[id]` is the primary key for the per-loan facts table.
   - The aggregate tables (`headline_metrics`, `discrimination_metrics`, `calibration_table`, `sensitivity_table`) are filtered independently — no direct relationship to `loans_summary` is required for the visuals on Pages 1–4.
   - Optional: create a `vintage_bucket` dimension table if you want a shared vintage slicer across pages.
4. **Set data types:**
   - All ECL columns (`total_ecl`, `ecl_total`, `ecl_final`, `ecl_regulatory`, `ecl_total_*`) → **Currency (USD)**.
   - Percentage columns (`ecl_ratio_pct`, `change_from_baseline_pct`, `stage2_share_pct`, `default_rate`) → **Decimal number**, format as percentage with 2 decimal places.
   - `issue_d` → **Date** (M/d/yyyy).
   - `pd_*`, `lgd_*`, `auc`, `gini`, `ks` → **Decimal number**.
   - `id`, `grade`, `sub_grade`, `purpose`, `home_ownership`, `addr_state`, `vintage_bucket`, `loan_status` → **Text**.
5. **Configure import.** For `loans_summary.csv` (1.18M rows), use Power BI Desktop's default Import mode — it will compress to ~30–50 MB in memory.

---

## Section 2 — Page 1: Executive Overview

### Layout

3 KPI cards across the top, two charts beneath.

### KPI cards (top row, left → right)

| Card | Metric | Subtitle | Formatting |
|---|---|---|---|
| 1 | **Baseline ECL** = `$403.5M` | "Step 12 model output, no forward-looking adjustment" | Card visual; gray background |
| 2 | **Data-Driven Overlay ECL** = `$390.8M` | "IFRS 9 mechanically applied; inverted direction documented" | Card visual; gray background |
| 3 | **Regulatory Overlay ECL** = `$484.5M` | "Recommended for IFRS 9 reporting" | Card visual; **highlighted** in primary color (navy) |

Source: `headline_metrics.csv` filtered to `breakdown = "all"`, `segment = "all"`, three rows for `version ∈ {baseline, data_overlay, regulatory}`.

### Chart 1 (left, lower half): "ECL by Stage — All Three Versions"

- **Visual type:** Stacked horizontal bar chart.
- **Y-axis:** Three bars, one per version (baseline, data_overlay, regulatory).
- **X-axis:** Total ECL.
- **Stack:** Stage 1 / Stage 2 / Stage 3 (color-coded).
- **Tooltip:** Show count and ECL per stage.
- **Source:** `headline_metrics.csv` filtered to `breakdown = "stage"`.

### Chart 2 (right, lower half): "ECL by Grade — All Three Versions"

- **Visual type:** Grouped column chart.
- **X-axis:** Grade A → G.
- **Y-axis:** Total ECL (USD).
- **Group:** Three bars per grade for the three versions.
- **Color:** Match the version palette from the KPI cards.
- **Source:** `headline_metrics.csv` filtered to `breakdown = "grade"`.

### Slicer (top-right corner)

Vintage filter (multi-select): `pre-2014`, `2014`, `2015`, `2016`, `2017+`.
**Source:** `loans_summary[vintage_bucket]`.

---

## Section 3 — Page 2: PD Model Performance

### Layout

4 sections in a 2×2 grid.

### Top-left: "Aggregate Discrimination" (KPI strip)

| KPI | Value |
|---|---:|
| AUC | `0.7059` |
| Gini | `0.4118` |
| KS | `0.2974` |
| Default rate (test) | `23.26%` |
| Test cohort n | `353,083` |

**Source:** `discrimination_metrics.csv` filtered to `dimension = "aggregate"`.

### Top-right: "Out-of-Time Stability" (line chart)

- **X-axis:** Vintage year (2016, 2017).
- **Y-axis:** AUC.
- **Annotation:** "AUC variation < 0.05 → stable" (set as text box).
- **Source:** `discrimination_metrics.csv` filtered to `dimension = "vintage"`.

### Bottom-left: "Reliability Diagram" (clustered column chart)

- **X-axis:** Decile 1–10.
- **Y-axis:** PD value.
- **Two columns per decile:** `predicted_pd` (light) and `observed_pd` (dark).
- **Annotation:** Decile MAD = `0.0276`.
- **Source:** `calibration_table.csv` filtered to `breakdown = "decile"`.

### Bottom-right: "Calibration by Grade" (clustered column chart)

- **X-axis:** Grade A → G.
- **Y-axis:** PD value.
- **Two columns per grade:** predicted vs observed.
- **Source:** `calibration_table.csv` filtered to `breakdown = "grade"`.

### Slicer (top-right)

Filter by `dimension` (vintage / grade / aggregate) for the discrimination KPIs, or by `breakdown` for the calibration tables.

---

## Section 4 — Page 3: Sensitivity & Stress

### Layout

3 panels in a vertical stack.

### Panel 1 (top): "SICR Threshold Sensitivity"

- **Visual type:** Bar chart.
- **X-axis:** SICR rule (`1.25x`, `1.5x`, `2.0x_current`, `2.5x`, `3.0x`, `abs_pd_5pct`, `all_stage1_floor`, `all_lifetime_ceiling`).
- **Y-axis:** Total ECL.
- **Highlight:** the `multiplier_2.00x_current` bar in primary color (navy); others in light gray.
- **Annotation:** "absolute_pd_5pct jumps Stage 2 share to 80%" (call-out).
- **Source:** `sensitivity_table.csv` filtered to `analysis_type = "SICR_threshold"`.

### Panel 2 (middle): "Overlay Weight Sensitivity"

- **Visual type:** Column chart.
- **X-axis:** Weight set (`60_30_10`, `50_30_20_current`, `40_40_20`, `33_33_33`, `40_30_30`, `30_40_30`, `100_baseline`, `100_adverse`, `100_severe`).
- **Y-axis:** Final ECL (data-driven overlay version).
- **Highlight:** `50_30_20_current` in primary color.
- **Source:** `sensitivity_table.csv` filtered to `analysis_type = "overlay_weights"`.

### Panel 3 (bottom): "Single-Feature Stress"

- **Visual type:** Horizontal bar chart.
- **Y-axis:** Feature shock (`sub_grade=G5`, `int_rate=30.99`, etc.).
- **X-axis:** ECL change % from baseline.
- **Color:** Bar shaded by magnitude — red for highest impact, gray for low.
- **Annotation:** Note that `sub_grade` dominates at +9.88% (consistent with IV ranking).
- **Source:** `sensitivity_table.csv` filtered to `analysis_type = "single_feature_stress"`.

### Slicer (top-right)

Filter by `analysis_type` to switch between the three panels' source data when needed.

---

## Section 5 — Page 4: The Macro-Overlay Finding

### Layout

Text panel left (50% width), chart panel right (50% width).

### Text panel (left)

```
THE MACRO-OVERLAY FINDING

Step 8 documented an unusual property of LendingClub data:
when unemployment rises, LC tightens credit standards
within the same year, partially offsetting the macro→default
relationship.

This propagates through the model. The PD model's coefficient
on unrate is -0.41 (with optbinning's WoE convention), meaning
the model has learned "high unemployment at origination →
better-quality borrowers chosen → lower predicted PD".

When we shock unrate +5pp in the severe stress scenario, the
model produces *lower* PDs, leading to a data-driven overlay
ECL that is 3.15% BELOW baseline — the opposite of IFRS 9's
expected direction.

This is mechanically correct on this dataset, but would not
pass production review. The regulatory overlay (right) replaces
the data-derived macro coefficients with conventional regulatory
stress-test sensitivities (+0.18 log-odds per pp unrate, +0.05
per −pp HPI YoY), producing the +20% adjustment IFRS 9 expects.

RECOMMENDATION: Report regulatory overlay $484.5M for external
purposes. Document data-driven overlay $390.8M as an internal
finding for model risk discussion.
```

Format: monospace text box, dark text on light background, 11pt.

### Chart panel (right)

- **Visual type:** Grouped column chart.
- **X-axis:** Scenario (Baseline, Adverse, Severe).
- **Two bars per scenario:**
  - Bar A: "Data-driven ECL" (gray)
  - Bar B: "Regulatory ECL" (primary color — navy)
- **Behavior:**
  - Both bars equal at the **Baseline** scenario ($403.5M).
  - Data-driven decreases through Adverse to Severe.
  - Regulatory increases through Adverse to Severe.
  - The crossing visualizes the inversion in 5 seconds.
- **Annotations:** "Baseline (anchor): $403.5M", "Severe data-driven: $376.2M", "Severe regulatory: $630.4M".
- **Source:** Build a measure table in DAX combining `headline_metrics` rows for `breakdown = "all"` per version, plus the per-scenario rows from `validation_regulatory_overlay.json`. Alternatively, hardcode three rows × two columns directly into a calculated table.

---

## Section 6 — Theme and styling

**Color palette (financial reporting):**
- **Navy** `#1E3A5F` — primary, regulatory headline.
- **Teal** `#3F8E9B` — secondary accent, by-grade visuals.
- **Amber** `#D69E2E` — alerts, the data-driven inversion call-out.
- **Gray** `#6B7280` — supporting elements, subtitles.
- **Light gray** `#F3F4F6` — backgrounds, gridlines.

No pastels.

**Number formatting:**
- USD with thousands separators: `$403,536,501` or `$403.5M` for headlines.
- Percentages with 2 decimal places: `2.37%`, `+20.06%`.
- Use `+` sign on positive deltas (`+9.88%`), no `+` on negatives (`-3.15%`).

**Typography:**
- Body: Segoe UI 11pt.
- Section headers: Segoe UI Semibold 16pt.
- KPI numbers: Segoe UI Bold 24pt.

**Headers and footers (every page):**
- Header: "IFRS 9 ECL Model — LendingClub Consumer Portfolio".
- Footer: "as_of: 2019-04-01 | Author: [your name] | Sources: 1.18M LendingClub loans + FRED macro".

**Page navigation:** Use Power BI's bookmark feature to add Next/Previous buttons in the bottom-right of each page.

---

## Section 7 — Final touches

1. **Bookmark each page** for navigation. Add a "Home" button on Pages 2–4 returning to Page 1.
2. **Tooltips on every chart.** Power BI defaults to truncated tooltips — manually enable full numbers (`Format → Tooltip → On`).
3. **Test the slicers.** Vintage filter on Page 1 should propagate to relevant charts (set up Sync Slicers if you want them shared across pages).
4. **Performance check.** Refresh the model — Power BI should load `loans_summary.csv` in under 30 seconds.
5. **Export the .pbix** as `dashboard/ecl_dashboard.pbix`.
6. **Optionally export to PDF** (`File → Export → Export to PDF`) for static viewing in `dashboard/ecl_dashboard.pdf`.

---

## Source data summary

| File | Rows | Purpose |
|---|---:|---|
| `loans_summary.csv` | 1,179,687 | Per-loan fact table (all visuals can drill into loan-level if needed) |
| `headline_metrics.csv` | ~120 | Aggregate KPIs (Page 1 cards + breakdowns) |
| `discrimination_metrics.csv` | ~10 | AUC/Gini/KS by vintage and grade (Page 2) |
| `calibration_table.csv` | ~25 | Decile reliability + by-grade calibration (Page 2) |
| `sensitivity_table.csv` | ~25 | SICR + weights + feature stress (Page 3) |

---

## Recommended demo flow (10-minute interview)

1. **Open Page 1** — point to the three KPI cards. "Three numbers, all defensible, recommendation is the regulatory overlay at $484.5M."
2. **Click Page 4** — explain the macro-overlay finding. "The data-driven number is below baseline because the model learned LC's underwriting reaction. The regulatory overlay corrects for this with textbook coefficients."
3. **Click Page 2** — show discrimination + calibration. "AUC stable across vintages, calibration MAD under 5%."
4. **Click Page 3** — sensitivity. "Headline is robust to SICR threshold and overlay weights; sub_grade dominates feature importance, consistent with IV ranking."
5. **Return to Page 1** — restate the recommendation. "External reporting: $484.5M. Internal: $390.8M as a model risk discussion item."
"""
    DASHBOARD_SPEC.write_text(spec)
    print(f"wrote: {DASHBOARD_SPEC}")


def task_run_validator() -> int:
    print("\n=== Re-run pipeline validator ===")
    if not VALIDATOR.exists():
        return -1
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        capture_output=True, text=True,
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(result.stderr[:1000])
    return result.returncode


if __name__ == "__main__":
    main()
