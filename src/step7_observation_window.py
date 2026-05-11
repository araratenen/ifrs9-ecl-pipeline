"""
Step 7 — Observation/Performance Windows + Data Quality Fixes.

Produces, deterministically, from the source parquet:
  - data/loans_modeling_ready.parquet
  - docs/feature_classification.json
  - docs/step7_methodology.md

Re-runnable: deleting the outputs and running this script regenerates them
identically. The source parquet (data/accepted_labeled.parquet) is read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


ROOT = Path("/Users/ostappolukainen/Desktop/ProjRED")
SRC = ROOT / "data" / "accepted_labeled.parquet"
OUT_PARQUET = ROOT / "data" / "loans_modeling_ready.parquet"
OUT_JSON = ROOT / "docs" / "feature_classification.json"
OUT_MD = ROOT / "docs" / "step7_methodology.md"


PD_INPUTS = [
    "loan_amnt", "funded_amnt", "term_months", "int_rate", "installment",
    "grade", "sub_grade", "purpose",
    "annual_inc", "dti",
    "fico_range_low", "fico_range_high",
    "emp_length_years", "home_ownership", "verification_status", "addr_state",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
    "pub_rec_bankruptcies", "mort_acc", "tax_liens",
    "application_type", "earliest_cr_line",
]

IDENTIFIERS = ["id", "issue_d", "last_pymnt_d"]

OUTCOME_ONLY = [
    "last_fico_range_low", "last_fico_range_high",
    "loan_status",
    "total_pymnt", "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee",
    "out_prncp",
    "chargeoff_within_12_mths", "collections_12_mths_ex_med",
    "hardship_flag", "debt_settlement_flag",
]

LABEL = "default_flag"


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("=== Loading source ===")
    df = pd.read_parquet(SRC)
    print(f"loaded: {len(df):,} rows × {df.shape[1]} cols from {SRC.name}")

    rec: dict = {"rows_initial": len(df)}

    df, rec = task_1(df, rec)
    df, rec = task_2(df, rec)
    rec = sanity_checks(df, rec)

    write_classification()
    write_parquet(df)
    write_methodology(rec)

    print("\n=== Done ===")
    for p in (OUT_PARQUET, OUT_JSON, OUT_MD):
        print(f"  {p}")


def task_1(df: pd.DataFrame, rec: dict) -> tuple[pd.DataFrame, dict]:
    print("\n=== Task 1.1: Drop DNMCP rows ===")
    before = len(df)
    mask = df["loan_status"].astype(str).str.startswith("Does not meet the credit policy")
    n = int(mask.sum())
    df = df.loc[~mask].copy()
    print(f"rows before:   {before:>10,}")
    print(f"rows after:    {len(df):>10,}")
    print(f"DNMCP dropped: {n:>10,}")
    rec["task_1_1"] = {"before": before, "after": len(df), "dropped": n}

    print("\n=== Task 1.2: Sentinel cleanup on dti ===")
    n = int((df["dti"] == 999).sum())
    df.loc[df["dti"] == 999, "dti"] = np.nan
    print(f"dti==999 → NaN: {n:,} cells")
    rec["task_1_2"] = {"affected": n}

    print("\n=== Task 1.3: Cap revol_util at 100 ===")
    n = int((df["revol_util"] > 100).sum())
    df.loc[df["revol_util"] > 100, "revol_util"] = 100.0
    print(f"revol_util > 100 → 100: {n:,} cells")
    rec["task_1_3"] = {"affected": n}

    print("\n=== Task 1.4: Winsorize annual_inc at p99 ===")
    p99 = float(df["annual_inc"].quantile(0.99))
    n = int((df["annual_inc"] > p99).sum())
    df.loc[df["annual_inc"] > p99, "annual_inc"] = p99
    print(f"p99 (post-Task-1.1): ${p99:,.2f}")
    print(f"capped: {n:,} cells")
    rec["task_1_4"] = {"affected": n, "p99": p99}

    print("\n=== Task 1.5: Drop FICO < 660 ===")
    before = len(df)
    n = int((df["fico_range_low"] < 660).sum())
    df = df.loc[df["fico_range_low"] >= 660].copy()
    print(f"rows before:    {before:>10,}")
    print(f"rows after:     {len(df):>10,}")
    print(f"FICO<660 drop:  {n:>10,}")
    rec["task_1_5"] = {"before": before, "after": len(df), "dropped": n}

    return df, rec


def task_2(df: pd.DataFrame, rec: dict) -> tuple[pd.DataFrame, dict]:
    print("\n=== Task 2.1: as_of ===")
    as_of = df["last_pymnt_d"].max() + DateOffset(months=1)
    print(f"as_of: {as_of.date()}")
    rec["as_of"] = str(as_of.date())

    print("\n=== Task 2.2: months_observable ===")
    df = df.copy()
    df["months_observable"] = (
        (as_of.year - df["issue_d"].dt.year) * 12
        + (as_of.month - df["issue_d"].dt.month)
    )
    print(f"min={int(df['months_observable'].min())}, "
          f"max={int(df['months_observable'].max())}, "
          f"median={int(df['months_observable'].median())}")

    print("\n=== Task 2.3: maturity filter (months_observable >= 24) ===")
    df_pre = df.copy()
    before = len(df)
    df = df.loc[df["months_observable"] >= 24].copy()
    print(f"rows before: {before:>10,}")
    print(f"rows after:  {len(df):>10,}")
    print(f"dropped:     {before - len(df):>10,}")
    rec["task_2_3"] = {"before": before, "after": len(df), "dropped": before - len(df)}

    print("\n=== Task 2.4: by-year before/after ===")
    yr_pre = df_pre["issue_d"].dt.year
    yr_post = df["issue_d"].dt.year
    summary = pd.DataFrame({
        "rows_before": df_pre.groupby(yr_pre).size(),
        "rows_after": df.groupby(yr_post).size(),
        "rate_before_%": df_pre.groupby(yr_pre)["default_flag"].mean() * 100,
        "rate_after_%": df.groupby(yr_post)["default_flag"].mean() * 100,
    })
    summary["rows_after"] = summary["rows_after"].fillna(0).astype(int)
    summary["rows_dropped"] = summary["rows_before"] - summary["rows_after"]
    summary = summary[["rows_before", "rows_after", "rows_dropped",
                       "rate_before_%", "rate_after_%"]]
    summary.index.name = "issue_year"
    summary["rate_before_%"] = summary["rate_before_%"].round(2)
    summary["rate_after_%"] = summary["rate_after_%"].round(2)
    print(summary.to_string())
    rec["by_year"] = summary

    return df, rec


def sanity_checks(df: pd.DataFrame, rec: dict) -> dict:
    print("\n=== Task 6: Sanity checks ===")
    final_rows = len(df)
    final_cols = df.shape[1]
    overall_rate = df["default_flag"].mean() * 100
    print(f"final rows: {final_rows:,}")
    print(f"final cols: {final_cols}")
    print(f"overall default rate: {overall_rate:.2f}%")

    by_grade = (df.groupby("grade", observed=True)["default_flag"].mean() * 100).sort_index()
    print(f"\nby grade:\n{by_grade.round(2).to_string()}")
    diffs = by_grade.diff().dropna()
    assert (diffs >= 0).all(), (
        f"FAIL: default rate by grade not monotonic A→G: {by_grade.round(2).to_dict()}"
    )

    bins = [0, 660, 700, 740, 780, 1000]
    labels = ["<660", "660-700", "700-740", "740-780", ">=780"]
    band = pd.cut(df["fico_range_low"], bins=bins, labels=labels, right=False)
    by_fico = (df.groupby(band, observed=True)["default_flag"].mean() * 100).dropna()
    print(f"\nby FICO band:\n{by_fico.round(2).to_string()}")
    diffs = by_fico.diff().dropna()
    assert (diffs <= 0).all(), (
        f"FAIL: default rate by FICO band not monotonically decreasing: {by_fico.round(2).to_dict()}"
    )

    by_year_post = (df.groupby(df["issue_d"].dt.year)["default_flag"].mean() * 100).round(2)
    print(f"\nby issue year (post-filter):\n{by_year_post.to_string()}")

    rec["final_rows"] = final_rows
    rec["final_cols"] = final_cols
    rec["overall_rate"] = overall_rate
    rec["by_grade"] = by_grade.round(2)
    rec["by_fico"] = by_fico.round(2)
    rec["by_year_post"] = by_year_post

    return rec


def write_classification() -> None:
    print(f"\n=== Task 3: feature classification → {OUT_JSON.name} ===")
    classification = {
        "pd_inputs": PD_INPUTS,
        "identifiers": IDENTIFIERS,
        "outcome_only": OUTCOME_ONLY,
        "label": LABEL,
    }
    OUT_JSON.write_text(json.dumps(classification, indent=2) + "\n")
    print(f"wrote: {OUT_JSON}")


def write_parquet(df: pd.DataFrame) -> None:
    print(f"\n=== Task 4: model-ready parquet → {OUT_PARQUET.name} ===")
    keep = PD_INPUTS + IDENTIFIERS + OUTCOME_ONLY + [LABEL]
    missing = [c for c in keep if c not in df.columns]
    assert not missing, f"missing columns from feature lists: {missing}"
    out = df[keep].copy()
    # pyarrow + pandas "string" extension dtype have a known compat hiccup inside
    # Jupyter kernels. Writing as plain object sidesteps it without changing
    # on-disk content (parquet stores both as plain UTF-8 strings).
    for col in out.select_dtypes(include=["string"]).columns:
        out[col] = out[col].astype(object)
    out.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / 1024**2
    print(f"shape: {out.shape}")
    print(f"size:  {size_mb:.1f} MB")


def write_methodology(rec: dict) -> None:
    print(f"\n=== Task 5: methodology → {OUT_MD.name} ===")
    by_year = rec["by_year"].to_string()
    by_grade = rec["by_grade"].to_string()
    by_fico = rec["by_fico"].to_string()

    md = (
        "# Step 7 — Observation/Performance Windows + Data Quality Fixes\n"
        "\n"
        "## 1. Observation and performance window choice\n"
        "\n"
        "**Observation date:** each loan's `issue_d` (origination date).\n"
        "**Performance window:** the contractual term of the loan (36 or 60 months).\n"
        "**Label:** `default_flag` — whether the loan transitioned to default "
        "(\"Charged Off\" or \"Default\") at any point during its term.\n"
        "\n"
        "This is a **lifetime PD** formulation: the model produces the probability "
        "that a loan defaults at any point during its term, conditional on attributes "
        "observable at origination.\n"
        "\n"
        "The alternative — a **snapshot-based 12-month PD** with monthly behavioral "
        "features (the typical approach for a live retail bank) — is not feasible on "
        "this dataset. Lending Club provides only **terminal** loan status, not monthly "
        "payment history; without monthly observations we cannot construct snapshot "
        "features as of an arbitrary date or measure a 12-month-forward PD directly. "
        "Conversion of the lifetime PD to a 12-month PD for IFRS 9 Stage 1 ECL is "
        "performed downstream using vintage hazard rates.\n"
        "\n"
        "## 2. Data quality fixes applied\n"
        "\n"
        "| Step | Action | Rows / cells affected |\n"
        "|---|---|---:|\n"
        f"| 1.1 | Drop \"Does not meet the credit policy\" rows (pre-2009 underwriting regime) | {rec['task_1_1']['dropped']:,} dropped |\n"
        f"| 1.2 | Set `dti = NaN` where `dti == 999` (sentinel for \"not computable\") | {rec['task_1_2']['affected']:,} cells |\n"
        f"| 1.3 | Cap `revol_util` at 100 (over-limit revolvers) | {rec['task_1_3']['affected']:,} cells |\n"
        f"| 1.4 | Winsorize `annual_inc` at p99 = ${rec['task_1_4']['p99']:,.0f} | {rec['task_1_4']['affected']:,} cells |\n"
        f"| 1.5 | Drop `fico_range_low < 660` (LC issuance minimum is 660) | {rec['task_1_5']['dropped']:,} dropped |\n"
        "\n"
        "**Rationale:**\n"
        "- DNMCP rows belong to a pre-2009 credit policy. Mixing them with the modern regime would contaminate the model's view of current underwriting standards. Dropping also corrects an earlier label inversion where DNMCP:Fully Paid was incorrectly mapped to default = 1.\n"
        "- `dti == 999` is documented as \"not computable\". Treated as missing rather than as a real DTI of 999.\n"
        "- `revol_util > 100%` indicates an over-limit revolving account; capping at 100% preserves the high-utilization signal while removing implausible magnitudes.\n"
        "- `annual_inc` had a max near $11M against a 99th percentile near $250K — clear data-entry artifacts. Winsorizing preserves the row but caps outlier influence.\n"
        "- The `fico_range_low < 660` band held only 338 rows but a 99.4% default rate against LC's policy of refusing applicants below 660. These are data anomalies whose retention would distort low-FICO modeling signal.\n"
        "\n"
        "`dti` is left as NaN. No imputation is applied at this stage — downstream WoE binning will handle missingness explicitly as its own bucket.\n"
        "\n"
        "## 3. Maturity filter\n"
        "\n"
        f"**`as_of` date:** `{rec['as_of']}` — derived as `max(last_pymnt_d) + 1 month`, used as the proxy for the dataset cutoff.\n"
        "\n"
        "**Rule:** keep loans where `months_observable = (as_of − issue_d) in calendar months ≥ 24`.\n"
        "\n"
        "**Rationale (survivorship bias):** the labeled population already excludes loans whose status is `Current`, `Late`, or `In Grace Period`. For recent vintages (2017–2018), most slow-performing loans are still `Current` and were excluded in the labeling step, leaving only loans that defaulted quickly. Using these vintages directly would over-represent fast defaulters and inflate apparent default rates. Requiring at least 24 months of observation lets slow defaulters surface; loans not in default by month 24 typically run to maturity without defaulting (consistent with vintage curves seen in EDA — the curve flattens after ~24 months).\n"
        "\n"
        f"**Rows before:** {rec['task_2_3']['before']:,}  \n"
        f"**Rows after:** {rec['task_2_3']['after']:,}  \n"
        f"**Rows dropped:** {rec['task_2_3']['dropped']:,}\n"
        "\n"
        "**Per-vintage breakdown (counts and default rate, before vs. after):**\n"
        "\n"
        "```\n"
        f"{by_year}\n"
        "```\n"
        "\n"
        "Pre-filter rates for 2017–2018 are inflated relative to mature vintages because only fast defaulters are present in the labeled set; post-filter, those years are either reduced to a smaller, more representative sub-vintage or removed entirely.\n"
        "\n"
        "## 4. Feature classification\n"
        "\n"
        "Three explicit lists, by downstream use.\n"
        "\n"
        f"### PD model inputs (origination features only — {len(PD_INPUTS)} features)\n"
        "\n"
        "```\n"
        f"{json.dumps(PD_INPUTS, indent=2)}\n"
        "```\n"
        "\n"
        f"### Identifiers and dates ({len(IDENTIFIERS)} columns; for joins, not features)\n"
        "\n"
        "```\n"
        f"{json.dumps(IDENTIFIERS, indent=2)}\n"
        "```\n"
        "\n"
        f"### Outcome / downstream-use only ({len(OUTCOME_ONLY)} columns; never as PD inputs)\n"
        "\n"
        "```\n"
        f"{json.dumps(OUTCOME_ONLY, indent=2)}\n"
        "```\n"
        "\n"
        "### Label\n"
        "\n"
        f"`{LABEL}`\n"
        "\n"
        "**Why outcome columns cannot be used as PD model inputs (data leakage):** outcome columns encode information observable only **after** the loan has run — total payments received, recoveries collected, last-credit-pull FICO, hardship and settlement flags, etc. A model trained on these features would achieve near-perfect in-sample accuracy by reading off the answer (e.g., a loan with high `recoveries` and `last_fico_range_low ≈ 0` is obviously charged off), but would not generalize at origination time when none of these are known. They are retained in the parquet only because the LGD and EAD models in subsequent steps require them.\n"
        "\n"
        "## 5. Output\n"
        "\n"
        "- **Path:** `data/loans_modeling_ready.parquet`\n"
        f"- **Final rows:** {rec['final_rows']:,}\n"
        f"- **Final columns:** {rec['final_cols']}\n"
        f"- **Default rate:** {rec['overall_rate']:.2f}%\n"
        "\n"
        "**Default rate by grade (post-filter):**\n"
        "\n"
        "```\n"
        f"{by_grade}\n"
        "```\n"
        "\n"
        "**Default rate by FICO band (post-filter):**\n"
        "\n"
        "```\n"
        f"{by_fico}\n"
        "```\n"
        "\n"
        "## 6. Limitations\n"
        "\n"
        "- The model produces **lifetime PD** because that is what the label measures. Conversion to **12-month PD** for IFRS 9 Stage 1 ECL is performed downstream using vintage hazard rates (transition matrices fit by year-of-origination).\n"
        "- A live bank would build a snapshot-based 12-month PD with monthly behavioral features (payment trends, utilization changes, late-payment counts). The Lending Club dataset's terminal-status structure does not provide monthly payment history, so this approach is not available here. The lifetime-PD-then-convert workaround is the standard treatment for terminal-status retail credit datasets.\n"
        "- DNMCP loans were dropped to avoid mixing the pre-2009 (looser) and post-2009 credit-policy regimes in training data. The remaining sample reflects post-2009 underwriting only.\n"
    )
    OUT_MD.write_text(md)
    print(f"wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
