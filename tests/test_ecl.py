"""Unit tests for the ECL combination math (Step 12).

Per-loan ECL is a function of stage:

    Stage 1: ECL = PD_12m * LGD * EAD_12m * DF_avg_12m
    Stage 2: ECL = lifetime ECL (path integral over marginal default hazard)
    Stage 3: ECL = LGD * EAD_lifetime_discounted_total / months_remaining

with PD_12m derived from the lifetime PD via constant monthly hazard:

    monthly_hazard = 1 - (1 - PD_lifetime)^(1/T)
    PD_12m         = 1 - (1 - monthly_hazard)^min(12, T)

These tests verify the formulas in isolation and reconcile one loan from
the artifact.
"""
import json

import numpy as np
import pandas as pd
import pytest


def lifetime_to_12m_pd(pd_lifetime: float, T: int) -> float:
    """Constant-hazard conversion used in Step 12."""
    if T <= 0:
        return 0.0
    p = float(np.clip(pd_lifetime, 1e-9, 1 - 1e-9))
    monthly = 1 - (1 - p) ** (1 / T)
    months_for_12m = min(12, T)
    return 1 - (1 - monthly) ** months_for_12m


def stage_1_ecl(pd_12m: float, lgd: float, ead_12m: float, df_avg_12m: float) -> float:
    return pd_12m * lgd * ead_12m * df_avg_12m


def stage_3_ecl(lgd: float, ead_lifetime_discounted: float, months_remaining: int) -> float:
    if months_remaining == 0:
        return 0.0
    return lgd * ead_lifetime_discounted / months_remaining


# ---------- Closed-form sanity ----------

def test_pd_12m_never_exceeds_lifetime_pd():
    for pd_lt in [0.01, 0.05, 0.20, 0.50, 0.90]:
        for T in [6, 12, 24, 36, 60]:
            assert lifetime_to_12m_pd(pd_lt, T) <= pd_lt + 1e-12


def test_pd_12m_equals_lifetime_when_T_is_12():
    """If T == 12, the 12-month PD equals the lifetime PD."""
    assert lifetime_to_12m_pd(0.05, 12) == pytest.approx(0.05)
    assert lifetime_to_12m_pd(0.30, 12) == pytest.approx(0.30)


def test_pd_12m_below_lifetime_when_T_above_12():
    """If T > 12, PD_12m must be strictly less than PD_lifetime."""
    assert lifetime_to_12m_pd(0.20, 60) < 0.20


def test_pd_12m_zero_for_zero_input():
    assert lifetime_to_12m_pd(0.0, 36) == pytest.approx(0, abs=1e-6)


def test_stage_1_ecl_zero_when_any_factor_zero():
    assert stage_1_ecl(0, 0.9, 5000, 0.95) == 0
    assert stage_1_ecl(0.05, 0, 5000, 0.95) == 0
    assert stage_1_ecl(0.05, 0.9, 0, 0.95) == 0


def test_stage_1_ecl_known_value():
    # PD_12m = 5%, LGD = 90%, EAD_12m = $5000, DF = 0.95
    assert stage_1_ecl(0.05, 0.9, 5000, 0.95) == pytest.approx(213.75)


def test_stage_3_ecl_zero_for_terminated_loan():
    assert stage_3_ecl(0.9, 1000.0, 0) == 0


def test_stage_3_ecl_proportional_to_lgd():
    a = stage_3_ecl(0.5, 1000, 12)
    b = stage_3_ecl(1.0, 1000, 12)
    assert b == pytest.approx(2 * a)


# ---------- Reconcile with pipeline output ----------

def test_loans_with_ecl_stage_1_reconciles(data_dir):
    """Pick a Stage-1 loan from the parquet and re-derive ecl_total from inputs.

    Note: `df_avg_12m` is a Step-12 intermediate dropped before save, so we
    recompute it from `discount_factors` (first-12-month average).
    """
    df = pd.read_parquet(data_dir / "loans_with_ecl.parquet")
    s1 = df[(df["ifrs9_stage"] == 1) & (df["ecl_total"] > 0) &
            (df["months_remaining"] > 0)]
    if s1.empty:
        pytest.skip("no positive Stage-1 ECL loans available")
    row = s1.iloc[0]
    discounts = np.asarray(row["discount_factors"])
    df_avg_12m = float(np.mean(discounts[:12])) if len(discounts) else 0.0
    expected = (row["pd_12m"] * row["lgd_predicted"]
                 * row["ead_12m"] * df_avg_12m)
    assert row["ecl_12m"] == pytest.approx(expected, rel=1e-6)
    assert row["ecl_total"] == pytest.approx(row["ecl_12m"], rel=1e-9)


def test_loans_with_ecl_stage_3_reconciles(data_dir):
    df = pd.read_parquet(data_dir / "loans_with_ecl.parquet")
    s3 = df[(df["ifrs9_stage"] == 3) & (df["months_remaining"] > 0) &
            (df["ecl_total"] > 0)]
    if s3.empty:
        pytest.skip("no positive Stage-3 ECL loans with months_remaining > 0")
    row = s3.iloc[0]
    expected = stage_3_ecl(
        row["lgd_predicted"],
        row["ead_lifetime_discounted_total"],
        int(row["months_remaining"]),
    )
    assert row["ecl_total"] == pytest.approx(expected, rel=1e-9)


def test_loans_with_ecl_pd_conversion_reconciles(data_dir):
    """pd_12m in the parquet must match the closed-form conversion from pd_lifetime."""
    df = pd.read_parquet(data_dir / "loans_with_ecl.parquet")
    sample = df[df["months_remaining"] > 0].sample(50, random_state=0)
    for _, row in sample.iterrows():
        expected = lifetime_to_12m_pd(
            float(row["pd_lifetime"]),
            int(row["months_remaining"]),
        )
        assert row["pd_12m"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_baseline_headline_matches_parquet_sum(ecl_headline, data_dir):
    """The headline ECL JSON must equal the sum of ecl_total in the parquet."""
    df = pd.read_parquet(data_dir / "loans_with_ecl.parquet",
                          columns=["ecl_total"])
    cited = float(ecl_headline["total_ecl"])
    recomputed = float(df["ecl_total"].sum())
    assert cited == pytest.approx(recomputed, abs=0.01)
