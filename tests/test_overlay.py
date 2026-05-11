"""Unit tests for the regulatory macro overlay (Step 14).

Three macro scenarios are run; per-loan PDs are shocked according to
regulatory log-odds coefficients (CCAR/EBA-style). The reportable ECL is
the weighted average across scenarios:

    ECL_weighted = w_baseline * ECL_baseline
                  + w_adverse * ECL_adverse
                  + w_severe  * ECL_severe

with weights summing to 1 (typically 0.5 / 0.3 / 0.2).
"""
import numpy as np
import pandas as pd
import pytest


def weighted_overlay(scenario_ecls: dict, weights: dict) -> float:
    return sum(weights[k] * scenario_ecls[k] for k in scenario_ecls)


# ---------- Math sanity ----------

def test_overlay_weights_sum_to_one(regulatory_overlay: dict):
    weights = {k: v["weight"] for k, v in regulatory_overlay["scenarios"].items()}
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_overlay_three_scenarios(regulatory_overlay: dict):
    assert set(regulatory_overlay["scenarios"]) == {"baseline", "adverse", "severe"}


def test_overlay_baseline_shock_zero(regulatory_overlay: dict):
    base = regulatory_overlay["scenarios"]["baseline"]
    assert base["unrate_shock_pp"] == 0.0
    assert base["hpi_yoy_shock_pp"] == 0.0


def test_overlay_severe_shock_more_extreme(regulatory_overlay: dict):
    """Severe scenario must shock more than adverse."""
    adv = regulatory_overlay["scenarios"]["adverse"]
    sev = regulatory_overlay["scenarios"]["severe"]
    assert abs(sev["unrate_shock_pp"]) > abs(adv["unrate_shock_pp"])
    assert abs(sev["hpi_yoy_shock_pp"]) > abs(adv["hpi_yoy_shock_pp"])


def test_overlay_simple_weighted_math():
    scenario_ecls = {"baseline": 100.0, "adverse": 200.0, "severe": 300.0}
    weights = {"baseline": 0.5, "adverse": 0.3, "severe": 0.2}
    result = weighted_overlay(scenario_ecls, weights)
    # 0.5*100 + 0.3*200 + 0.2*300 = 170
    assert result == pytest.approx(170.0)


# ---------- Reconcile with pipeline output ----------

def test_regulatory_headline_reconciles_from_per_scenario(regulatory_overlay: dict):
    """The weighted_final_ecl JSON value must equal w·ECL_per_scenario."""
    scenarios = regulatory_overlay["scenarios"]
    ecls = {k: v["total_ecl"] for k, v in scenarios.items()}
    weights = {k: v["weight"] for k, v in scenarios.items()}
    cited = float(regulatory_overlay["weighted_final_ecl"])
    recomputed = weighted_overlay(ecls, weights)
    assert cited == pytest.approx(recomputed, abs=0.01)


def test_regulatory_headline_higher_than_baseline(regulatory_overlay: dict):
    """Adverse + severe scenarios shock UP, so weighted ECL > baseline."""
    weighted = float(regulatory_overlay["weighted_final_ecl"])
    baseline = float(regulatory_overlay["step12_baseline_ecl"])
    assert weighted > baseline


def test_regulatory_overlay_multiplier_consistent(regulatory_overlay: dict):
    """The overlay_multiplier must equal weighted/baseline."""
    weighted = float(regulatory_overlay["weighted_final_ecl"])
    baseline = float(regulatory_overlay["step12_baseline_ecl"])
    cited = float(regulatory_overlay["overlay_multiplier_vs_baseline"])
    assert cited == pytest.approx(weighted / baseline, rel=1e-6)


# ---------- Data-driven overlay (Step 13) ----------

def test_data_overlay_headline_matches_parquet_sum(overlay_headline, data_dir):
    df = pd.read_parquet(data_dir / "loans_with_ecl_overlay.parquet",
                          columns=["ecl_final"])
    cited = float(overlay_headline["final_ecl"])
    recomputed = float(df["ecl_final"].sum())
    assert cited == pytest.approx(recomputed, abs=0.01)


def test_data_overlay_per_scenario_columns_present(data_dir):
    df = pd.read_parquet(data_dir / "loans_with_ecl_overlay.parquet")
    for col in ["ecl_total_baseline", "ecl_total_adverse", "ecl_total_severe"]:
        assert col in df.columns, f"missing per-scenario column {col}"
