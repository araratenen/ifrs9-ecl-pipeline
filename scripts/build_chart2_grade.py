"""Chart 2 — ECL by grade (two-panel).

Top: bar of total_ecl by grade A-G
Bottom: line of coverage_ratio (= ecl_ratio_pct / 100)

Source: data/dashboard/headline_metrics.csv (baseline + breakdown=='grade')
Intermediate: docs/aggregates/ecl_by_grade.csv
Output: docs/figures/chart2_ecl_by_grade.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from chart_style import PALETTE, apply_style, fmt_money_short

ROOT = Path(__file__).resolve().parent.parent


def main():
    apply_style()
    src = pd.read_csv(ROOT / "data/dashboard/headline_metrics.csv", low_memory=False)
    df = src[(src["breakdown"] == "grade") & (src["version"] == "baseline")].copy()
    df = df.rename(columns={"segment": "grade", "n_loans": "count"})
    df["coverage_ratio"] = df["ecl_ratio_pct"] / 100.0
    df["ecl_per_loan"] = df["total_ecl"] / df["count"]
    df = df[["grade", "count", "total_ecl", "total_funded", "ecl_per_loan", "coverage_ratio"]]
    df = df.sort_values("grade").reset_index(drop=True)
    df.to_csv(ROOT / "docs/aggregates/ecl_by_grade.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})

    ax1.bar(df["grade"], df["total_ecl"], color=PALETTE["primary"], edgecolor="white", width=0.62)
    ax1.set_ylabel("Total ECL (USD)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: fmt_money_short(x)))
    ax1.set_title("IFRS 9 ECL by grade — volume vs. risk rate", loc="left")
    for i, row in df.iterrows():
        ax1.text(i, row["total_ecl"] * 1.02, fmt_money_short(row["total_ecl"]),
                 ha="center", va="bottom", fontsize=8.5, color=PALETTE["neutral_text"])
    ax1.set_ylim(top=df["total_ecl"].max() * 1.18)

    ax2.plot(df["grade"], df["coverage_ratio"] * 100, marker="o", color=PALETTE["secondary"], linewidth=1.7)
    ax2.set_ylabel("Coverage (ECL / funded)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}%"))
    ax2.set_xlabel("Grade")
    for i, row in df.iterrows():
        ax2.text(i, row["coverage_ratio"] * 100 + 0.4, f"{row['coverage_ratio']*100:.2f}%",
                 ha="center", va="bottom", fontsize=8.5, color="#555555")
    ax2.set_ylim(top=df["coverage_ratio"].max() * 100 * 1.18)

    fig.text(
        0.02, 0.01,
        "ECL concentrates in middle grades by absolute volume, but the coverage rate climbs monotonically A → G.",
        fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = ROOT / "docs/figures/chart2_ecl_by_grade.png"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
