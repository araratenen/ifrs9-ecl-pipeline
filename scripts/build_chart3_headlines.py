"""Chart 3 — Three headline ECL numbers (horizontal bars).

Source: data/dashboard/headline_metrics.csv (rows where breakdown == 'all').
Intermediate: docs/aggregates/ecl_headlines.json
Output: docs/figures/chart3_headlines.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chart_style import PALETTE, apply_style, fmt_money_short

ROOT = Path(__file__).resolve().parent.parent


def main():
    apply_style()
    src = pd.read_csv(ROOT / "data/dashboard/headline_metrics.csv", low_memory=False)
    top = src[src["breakdown"] == "all"].set_index("version")["total_ecl"].to_dict()
    headlines = {
        "baseline": float(top["baseline"]),
        "data_overlay": float(top["data_overlay"]),
        "regulatory": float(top["regulatory"]),
    }
    # save intermediate
    out_json = ROOT / "docs/aggregates/ecl_headlines.json"
    out_json.write_text(json.dumps(headlines, indent=2))

    labels = [
        ("Regulatory overlay\n(recommended for IFRS 9)", headlines["regulatory"], PALETTE["regulatory"]),
        ("Baseline ECL\n(model PD only)", headlines["baseline"], PALETTE["baseline"]),
        ("Data-driven overlay\n(macro-weighted)", headlines["data_overlay"], PALETTE["data_overlay"]),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = list(range(len(labels)))
    vals = [v for _, v, _ in labels]
    colors = [c for _, _, c in labels]
    names = [n for n, _, _ in labels]
    bars = ax.barh(y, vals, color=colors, edgecolor="white", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("Total ECL (USD)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: fmt_money_short(x)))
    ax.grid(axis="y", visible=False)

    baseline = headlines["baseline"]
    for bar, (_, v, _) in zip(bars, labels):
        delta = (v - baseline) / baseline * 100
        delta_txt = f"  ({delta:+.1f}% vs. baseline)" if v != baseline else ""
        ax.text(
            v + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{fmt_money_short(v)}{delta_txt}",
            va="center",
            fontsize=10,
            color=PALETTE["neutral_text"],
        )

    ax.set_title("IFRS 9 ECL — three overlay views", loc="left")
    fig.suptitle(
        "Data-driven overlay decreases vs. baseline (Simpson's paradox / underwriting-reaction effect — see methodology).\n"
        "Regulatory overlay applied for IFRS 9 reporting.",
        x=0.02, y=0.02, ha="left", va="bottom", fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_png = ROOT / "docs/figures/chart3_headlines.png"
    fig.savefig(out_png)
    print(f"wrote {out_json.relative_to(ROOT)}")
    print(f"wrote {out_png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
