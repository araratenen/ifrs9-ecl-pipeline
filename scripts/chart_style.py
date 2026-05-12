"""Shared matplotlib styling for ECL and audit dashboards.

Clean, minimal, white background, readable fonts, tight layouts.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "baseline": "#9e9e9e",
    "data_overlay": "#e69500",
    "regulatory": "#2e7d32",
    "primary": "#1f4e79",
    "secondary": "#c0504d",
    "accent": "#e69500",
    "muted": "#cccccc",
    "good": "#2e7d32",
    "warn": "#c0504d",
    "neutral_text": "#222222",
}


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "text.parse_math": False,
    })


def fmt_money_millions(x, _pos=None):
    return f"${x/1e6:,.1f}M"


def fmt_money_short(x):
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    if x >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:,.0f}"
