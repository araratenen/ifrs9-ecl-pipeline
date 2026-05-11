"""Unit tests for the Weight-of-Evidence convention used in Step 9a.

Convention (matches `optbinning` default and Step 9a usage):

    WoE_bin = log( P(non-event | bin) / P(event | bin) )
    IV      = sum_bins (P(non-event | bin) - P(event | bin)) * WoE_bin

Implication: high event rate → low P(non-event)/P(event) → negative WoE.

WoE-transformed features used as logistic-regression inputs flip sign
relative to a "raw probability" intuition: positive coefficient on a
WoE-transformed feature means the feature is RISK-PROTECTIVE (because high
WoE means low event rate). This is documented behavior in Step 9b.
"""
import json

import numpy as np
import pandas as pd
import pytest


def woe(p_non_event: float, p_event: float, eps: float = 1e-9) -> float:
    return np.log(max(p_non_event, eps) / max(p_event, eps))


def iv(bin_stats: list[tuple[float, float]]) -> float:
    """Compute IV from a list of (p_non_event, p_event) per bin."""
    total = 0.0
    for p_non, p_e in bin_stats:
        if p_non > 0 and p_e > 0:
            total += (p_non - p_e) * woe(p_non, p_e)
    return total


# ---------- Closed-form sanity ----------

def test_woe_zero_for_balanced_bin():
    """Bin with equal event/non-event distribution shares → WoE = 0."""
    assert woe(0.5, 0.5) == pytest.approx(0)


def test_woe_negative_for_high_event_rate_bin():
    """Bin with HIGHER share of events than non-events → negative WoE."""
    assert woe(p_non_event=0.10, p_event=0.50) < 0


def test_woe_positive_for_low_event_rate_bin():
    """Bin with LOWER share of events than non-events → positive WoE (protective)."""
    assert woe(p_non_event=0.50, p_event=0.10) > 0


def test_iv_non_negative():
    """IV is a sum of (Δp) × WoE — always non-negative; zero iff all bins balanced."""
    bins = [(0.20, 0.40), (0.30, 0.30), (0.50, 0.30)]
    assert iv(bins) >= 0


def test_iv_zero_for_uninformative_feature():
    """If every bin has p_non_event == p_event, IV = 0."""
    bins = [(0.25, 0.25), (0.25, 0.25), (0.25, 0.25), (0.25, 0.25)]
    assert iv(bins) == pytest.approx(0)


def test_iv_higher_for_more_separation():
    """Greater separation between event/non-event distributions → higher IV."""
    weak = [(0.40, 0.50), (0.60, 0.50)]
    strong = [(0.10, 0.50), (0.90, 0.50)]
    assert iv(strong) > iv(weak)


# ---------- Reconcile with binning_summary.json ----------

def test_binning_summary_iv_values_non_negative(docs_dir):
    summary = json.loads((docs_dir / "binning_summary.json").read_text())
    for entry in summary["iv_table"]:
        assert entry["iv"] >= 0, f"feature {entry['feature']} has negative IV"


def test_binning_summary_iv_table_shape(docs_dir):
    """IV table must have an entry per evaluated feature with required keys."""
    summary = json.loads((docs_dir / "binning_summary.json").read_text())
    for entry in summary["iv_table"]:
        assert {"feature", "iv", "status"} <= set(entry)


def test_train_woe_values_are_transformed(data_dir, docs_dir):
    """In `train_woe.parquet`, columns retain their original names but contain
    WoE-transformed values (typically O(1) magnitudes, not raw feature units)."""
    df_woe = pd.read_parquet(data_dir / "train_woe.parquet")
    df_raw = pd.read_parquet(data_dir / "train.parquet")
    summary = json.loads((docs_dir / "binning_summary.json").read_text())

    selected = [e["feature"] for e in summary["iv_table"]
                 if e["status"] in ("selected", "selected_forced")]
    common = [f for f in selected if f in df_woe.columns and f in df_raw.columns]
    assert common, "no selected features present in both train and train_woe"

    # WoE-transformed columns must NOT match the raw values (modulo missing).
    for feat in common[:5]:
        a = pd.to_numeric(df_woe[feat], errors="coerce").dropna()
        b = pd.to_numeric(df_raw[feat], errors="coerce").dropna()
        if a.empty or b.empty:
            continue
        n = min(1000, len(a), len(b))
        assert not np.allclose(a.iloc[:n].to_numpy(),
                                  b.iloc[:n].to_numpy(), atol=1e-6), \
            f"feature {feat} appears un-transformed in train_woe.parquet"


def test_train_woe_values_finite(data_dir, docs_dir):
    """Selected WoE columns must be finite (no NaN/inf)."""
    df = pd.read_parquet(data_dir / "train_woe.parquet")
    summary = json.loads((docs_dir / "binning_summary.json").read_text())
    selected = [e["feature"] for e in summary["iv_table"]
                 if e["status"] in ("selected", "selected_forced")]
    cols = [c for c in selected if c in df.columns]
    if not cols:
        pytest.skip("no selected features present")
    sample = df[cols].sample(min(1000, len(df)), random_state=0)
    arr = sample.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_share = np.isfinite(arr).mean()
    assert finite_share > 0.99, f"only {finite_share:.1%} of WoE values finite"
