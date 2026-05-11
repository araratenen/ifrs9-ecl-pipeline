"""Unit tests for IFRS 9 stage assignment (Step 12).

The proxy SICR rule is:

    sicr_threshold(grade) = grade_average_pd_lifetime(grade) * SICR_MULTIPLIER

    Stage 1 — pd_lifetime ≤ threshold and not defaulted (performing)
    Stage 2 — pd_lifetime  > threshold and not defaulted (significant deterioration)
    Stage 3 — default_flag == 1                          (defaulted)
"""
import numpy as np
import pandas as pd
import pytest


def assign_stage(pd_lt: float, threshold: float, default_flag: int) -> int:
    if default_flag == 1:
        return 3
    if pd_lt > threshold:
        return 2
    return 1


# ---------- Sanity ----------

def test_default_flag_always_stage_3():
    assert assign_stage(0.001, 0.50, default_flag=1) == 3
    assert assign_stage(0.999, 0.10, default_flag=1) == 3


def test_low_risk_stage_1():
    assert assign_stage(pd_lt=0.05, threshold=0.20, default_flag=0) == 1


def test_high_risk_stage_2():
    assert assign_stage(pd_lt=0.30, threshold=0.20, default_flag=0) == 2


def test_at_threshold_is_stage_1():
    """Convention is `>` strict — equality lands in Stage 1."""
    assert assign_stage(0.20, 0.20, default_flag=0) == 1


# ---------- Reconciliation ----------

def test_stage_counts_partition_population(ecl_headline: dict):
    """Stage counts must sum to total loan count."""
    by_stage = ecl_headline["by_stage"]
    s1 = int(by_stage["stage_1"]["count"])
    s2 = int(by_stage["stage_2"]["count"])
    s3 = int(by_stage["stage_3"]["count"])
    total = int(ecl_headline["total_loans"])
    assert s1 + s2 + s3 == total


def test_stage_3_count_equals_defaulters(data_dir, ecl_headline: dict):
    """Stage 3 count must equal the number of default_flag==1 loans."""
    df = pd.read_parquet(data_dir / "loans_with_ecl.parquet",
                          columns=["default_flag", "ifrs9_stage"])
    n_default_flag = int((df["default_flag"] == 1).sum())
    n_stage_3 = int((df["ifrs9_stage"] == 3).sum())
    assert n_default_flag == n_stage_3


def test_stage_3_ecl_in_headline_matches_parquet_sum(ecl_headline: dict, data_dir):
    df = pd.read_parquet(data_dir / "loans_with_ecl.parquet")
    cited = float(ecl_headline["by_stage"]["stage_3"]["ecl"])
    recomputed = float(df.loc[df["ifrs9_stage"] == 3, "ecl_total"].sum())
    assert cited == pytest.approx(recomputed, abs=0.01)


# Constant from src/step12_ecl_combination.py — kept here to avoid a circular
# import in tests; if the production constant changes, this test catches it
# via the assignment reconciliation below.
SICR_MULTIPLIER = 2.0


def _grade_thresholds(df: pd.DataFrame) -> dict:
    """Reproduce step12's threshold table (grade-average pd_lifetime × SICR_MULTIPLIER)."""
    active = df[df["months_remaining"] > 0]
    grade_avg = active.groupby("grade", observed=True)["pd_lifetime"].mean()
    return (grade_avg * SICR_MULTIPLIER).to_dict()


def test_pipeline_stage_assignment_matches_reference(data_dir):
    """For a sample of loans, the pipeline's stage must match the reference rule."""
    df = pd.read_parquet(data_dir / "loans_with_ecl.parquet")
    thresholds = _grade_thresholds(df)
    sample = df.sample(500, random_state=0)
    for _, row in sample.iterrows():
        threshold = thresholds.get(row["grade"])
        if threshold is None:
            continue
        expected = assign_stage(
            float(row["pd_lifetime"]),
            float(threshold),
            int(row["default_flag"]),
        )
        assert int(row["ifrs9_stage"]) == expected, (
            f"mismatch: grade={row['grade']}, pd_lt={row['pd_lifetime']:.4f}, "
            f"threshold={threshold:.4f}, default={row['default_flag']}, "
            f"expected={expected}, got={row['ifrs9_stage']}"
        )


def test_sicr_threshold_recomputable_per_grade(data_dir):
    """The grade-average × SICR_MULTIPLIER table must reproduce the partition
    of Stage-1 vs Stage-2 within each grade for non-defaulted active loans."""
    df = pd.read_parquet(data_dir / "loans_with_ecl.parquet")
    thresholds = _grade_thresholds(df)
    perf = df[(df["default_flag"] == 0) & (df["months_remaining"] > 0)]
    for grade, threshold in thresholds.items():
        sub = perf[perf["grade"] == grade]
        if sub.empty:
            continue
        # Every row above threshold should be Stage 2; below or equal, Stage 1
        s1_misclassified = ((sub["pd_lifetime"] <= threshold)
                              & (sub["ifrs9_stage"] != 1)).sum()
        s2_misclassified = ((sub["pd_lifetime"] > threshold)
                              & (sub["ifrs9_stage"] != 2)).sum()
        assert s1_misclassified == 0, (
            f"grade {grade}: {s1_misclassified} loans below threshold not in Stage 1"
        )
        assert s2_misclassified == 0, (
            f"grade {grade}: {s2_misclassified} loans above threshold not in Stage 2"
        )
