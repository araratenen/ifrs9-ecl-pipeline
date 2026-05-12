"""Chart 7 — Simpson's paradox: between-vintage vs. within-vintage correlation
of unemployment and default rate.

Source: data/loans_with_macros.parquet
Intermediate: docs/aggregates/simpsons_table.csv (per-year aggregates)
              docs/aggregates/simpsons_correlations.json (raw + residualized corrs)
Output: docs/figures/chart7_simpsons_paradox.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chart_style import PALETTE, apply_style

ROOT = Path(__file__).resolve().parent.parent


def main():
    apply_style()
    df = pd.read_parquet(
        ROOT / "data/loans_with_macros.parquet",
        columns=["issue_d", "default_flag", "unrate"],
    )
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    df["vintage_year"] = df["issue_d"].dt.year
    df = df.dropna(subset=["unrate", "default_flag", "vintage_year"])

    yearly = (
        df.groupby("vintage_year")
          .agg(mean_unrate=("unrate", "mean"),
               default_rate=("default_flag", "mean"),
               n=("default_flag", "size"))
          .reset_index()
          .sort_values("vintage_year")
    )

    # raw between-vintage correlation (unweighted across years)
    raw_corr = float(yearly["mean_unrate"].corr(yearly["default_rate"]))

    # residualized within-year correlation at loan level
    year_means_u = df.groupby("vintage_year")["unrate"].transform("mean")
    year_means_d = df.groupby("vintage_year")["default_flag"].transform("mean")
    df["unrate_resid"] = df["unrate"] - year_means_u
    df["default_resid"] = df["default_flag"] - year_means_d
    resid_corr = float(df["unrate_resid"].corr(df["default_resid"]))

    # save intermediates
    yearly.to_csv(ROOT / "docs/aggregates/simpsons_table.csv", index=False)
    (ROOT / "docs/aggregates/simpsons_correlations.json").write_text(
        json.dumps({
            "raw_between_year_corr": round(raw_corr, 4),
            "residualized_within_year_corr": round(resid_corr, 4),
            "n_vintages": int(yearly.shape[0]),
            "n_loans": int(df.shape[0]),
            "vintage_min": int(yearly["vintage_year"].min()),
            "vintage_max": int(yearly["vintage_year"].max()),
        }, indent=2)
    )

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ---- Panel 1: raw between-vintage scatter ----
    ax = axes[0]
    ax.scatter(
        yearly["mean_unrate"], yearly["default_rate"],
        s=70, color=PALETTE["data_overlay"], edgecolor="white", zorder=3,
    )
    for _, r in yearly.iterrows():
        ax.annotate(
            f"{int(r['vintage_year'])}",
            (r["mean_unrate"], r["default_rate"]),
            xytext=(6, 4), textcoords="offset points", fontsize=8.5, color="#333333",
        )
    # OLS fit
    x = yearly["mean_unrate"].to_numpy()
    y = yearly["default_rate"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() * 0.98, x.max() * 1.02, 50)
    ax.plot(xs, intercept + slope * xs, color=PALETTE["data_overlay"], linewidth=1.2, alpha=0.55)
    ax.set_xlabel("Vintage-year mean unemployment rate (%)")
    ax.set_ylabel("Vintage-year default rate")
    ax.set_title(f"Raw between-vintage  ·  r = {raw_corr:+.3f}", loc="left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v*100:.0f}%"))

    # ---- Panel 2: residualized within-vintage ----
    ax = axes[1]
    # bin unrate_resid into 25 deciles-equivalent and show mean default_resid per bin
    bins = np.linspace(df["unrate_resid"].quantile(0.005), df["unrate_resid"].quantile(0.995), 26)
    df["urate_bin"] = pd.cut(df["unrate_resid"], bins=bins, include_lowest=True)
    binned = df.groupby("urate_bin", observed=True).agg(
        mean_unrate_resid=("unrate_resid", "mean"),
        mean_default_resid=("default_resid", "mean"),
        n=("default_resid", "size"),
    ).dropna().reset_index(drop=True)
    sizes = (binned["n"] / binned["n"].max() * 120) + 12
    ax.scatter(
        binned["mean_unrate_resid"], binned["mean_default_resid"],
        s=sizes, color=PALETTE["regulatory"], edgecolor="white", alpha=0.85, zorder=3,
    )
    xs2 = np.linspace(binned["mean_unrate_resid"].min(), binned["mean_unrate_resid"].max(), 50)
    s2, i2 = np.polyfit(binned["mean_unrate_resid"], binned["mean_default_resid"], 1)
    ax.plot(xs2, i2 + s2 * xs2, color=PALETTE["regulatory"], linewidth=1.2, alpha=0.55)
    ax.axhline(0, color="#bbbbbb", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="#bbbbbb", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Unemployment rate, residualized within vintage year")
    ax.set_ylabel("Default flag, residualized within vintage year")
    ax.set_title(f"Within-vintage (loan-level residuals)  ·  r = {resid_corr:+.3f}", loc="left")

    fig.suptitle(
        "Simpson's paradox: unemployment co-moves with vintage-year defaults in aggregate,\n"
        "but within a vintage the relationship vanishes — the macro overlay's signal is between-vintage, not within.",
        x=0.02, y=0.99, ha="left", va="top", fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_png = ROOT / "docs/figures/chart7_simpsons_paradox.png"
    fig.savefig(out_png)

    print(f"raw between-vintage r = {raw_corr:+.4f}")
    print(f"residualized within-vintage r = {resid_corr:+.4f}")
    print(f"vintages: {int(yearly['vintage_year'].min())}-{int(yearly['vintage_year'].max())}, n_loans={len(df):,}")
    print(f"wrote {out_png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
