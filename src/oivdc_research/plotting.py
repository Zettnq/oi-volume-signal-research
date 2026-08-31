# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Plotting utilities with two switchable themes.

Themes:
- "light":    standard academic style (white background, grey grid);
- "ice_nine": dark brand style.

The theme is selected via ResearchConfig.plot_theme. All plot functions read
it from the config, so a single flag restyles the whole report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .config import ResearchConfig

__all__ = [
    "ICE_NINE",
    "LIGHT",
    "get_palette",
    "set_plot_style",
    "apply_axes_style",
    "apply_axes_style_all",
    "save_figure",
    "plot_event_study_curve",
    "plot_winrate_by_horizon",
    "plot_pnl_distribution",
    "plot_methodology_comparison",
    "plot_signal_counts_over_time",
    "plot_clustering_distribution",
]


# ---------------------------------------------------------------------- #
# Palettes
# ---------------------------------------------------------------------- #
ICE_NINE = {
    "background": "#161413",
    "panel": "#141010",
    "grid": "#787f7f",
    "grid_alpha": 0.3,
    "text": "#d0d4dc",
    "text_secondary": "#8a8f98",
    "combined": "#5b9bd5",
    "long": "#4ecdc4",
    "short": "#dc143c",
    "accent": "#dc143c",
}

LIGHT = {
    "background": "#ffffff",
    "panel": "#ffffff",
    "grid": "#d9d9d9",
    "grid_alpha": 0.6,
    "text": "#111111",
    "text_secondary": "#555555",
    "combined": "#1f77b4",
    "long": "#2ca02c",
    "short": "#d62728",
    "accent": "#d62728",
}

PALETTES = {"ice_nine": ICE_NINE, "light": LIGHT}

# Module-level active theme (set via set_plot_style).
_CURRENT = {"theme": "light"}


def get_palette(theme: str | None = None) -> dict:
    """Return a palette by name, or the currently active one."""
    name = theme if theme is not None else _CURRENT["theme"]
    if name not in PALETTES:
        raise ValueError(f"Unknown theme: {name}. Supported: {sorted(PALETTES)}.")
    return PALETTES[name]


def _rc_for(p: dict) -> dict:
    """Build matplotlib rcParams from a palette."""
    return {
        "figure.facecolor": p["background"],
        "figure.dpi": 150,
        "axes.facecolor": p["panel"],
        "axes.edgecolor": p["grid"],
        "axes.labelcolor": p["text"],
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlecolor": p["text"],
        "axes.grid": True,
        "grid.color": p["grid"],
        "grid.alpha": p["grid_alpha"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "xtick.color": p["text_secondary"],
        "ytick.color": p["text_secondary"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.facecolor": p["panel"],
        "legend.edgecolor": p["grid"],
        "legend.labelcolor": p["text"],
        "legend.fontsize": 10,
        "text.color": p["text"],
        "font.family": "sans-serif",
        "font.size": 11,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
    }


def set_plot_style(theme: str = "light") -> None:
    """Apply global rcParams for the given theme and make it active."""
    p = get_palette(theme)
    _CURRENT["theme"] = theme
    plt.rcParams.update(_rc_for(p))


def _theme_from_config(config: ResearchConfig) -> str:
    return getattr(config, "plot_theme", "light")


def apply_axes_style(ax) -> None:
    """Hide top/right spines and tint the rest to the active grid colour."""
    p = get_palette()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(p["grid"])
    ax.spines["left"].set_color(p["grid"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)


def apply_axes_style_all(axes) -> None:
    """Apply apply_axes_style to every axis in an array of axes."""
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            apply_axes_style(ax)
    else:
        apply_axes_style(axes)


def save_figure(fig, save_path: Path | None) -> None:
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none",
        )
        # short, personal-free path for public notebooks
        try:
            short = "/".join(save_path.parts[save_path.parts.index("reports"):])
        except ValueError:
            short = save_path.name
        print(f"Saved: {short}")


_SCOPE_COLORS = ("combined", "long", "short")


# ---------------------------------------------------------------------- #
# Plot functions
# ---------------------------------------------------------------------- #
def plot_event_study_curve(
    stats_df: pd.DataFrame,
    config: ResearchConfig,
    save_path: Path | None = None,
) -> None:
    """Mean PnL by horizon with bootstrap CI (episodes, non-overlapping)."""
    set_plot_style(_theme_from_config(config))
    p = get_palette()

    df = stats_df[
        (stats_df["method"] == "episodes")
        & (stats_df["sample_type"] == "nonoverlapping")
    ].copy()
    if df.empty:
        print("No data for event study curve.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for scope in _SCOPE_COLORS:
        scope_df = df[df["signal_scope"] == scope].sort_values("horizon")
        if scope_df.empty:
            continue
        ax.plot(
            scope_df["horizon"],
            scope_df["mean_pnl"] * 100,
            marker="o",
            markersize=7,
            label=f"{scope.capitalize()} signal",
            color=p[scope],
            alpha=0.9,
        )
        ax.fill_between(
            scope_df["horizon"],
            scope_df["bootstrap_ci_lower"] * 100,
            scope_df["bootstrap_ci_upper"] * 100,
            alpha=0.12,
            color=p[scope],
        )

    ax.axhline(y=0, color=p["text_secondary"], ls="--", lw=1, alpha=0.5)
    ax.axvline(
        x=config.primary_horizon,
        color=p["accent"],
        ls=":",
        lw=2,
        alpha=0.8,
        label=f"Primary horizon (h={config.primary_horizon})",
    )
    ax.set_xlabel("Horizon (bars)")
    ax.set_ylabel("Mean Strategy PnL (%)")
    ax.set_title(f"Event Study: Mean PnL by Horizon\n{config.asset.upper()} {config.timeframe}")
    apply_axes_style(ax)
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()


def plot_winrate_by_horizon(
    stats_df: pd.DataFrame,
    config: ResearchConfig,
    save_path: Path | None = None,
) -> None:
    """Winrate by horizon with 95% Wilson CI; scopes offset for readability."""
    set_plot_style(_theme_from_config(config))
    p = get_palette()

    df = stats_df[
        (stats_df["method"] == "episodes")
        & (stats_df["sample_type"] == "nonoverlapping")
    ].copy()
    if df.empty:
        print("No data for winrate plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    offsets = {"combined": -0.15, "long": 0.0, "short": 0.15}

    for scope in _SCOPE_COLORS:
        scope_df = df[df["signal_scope"] == scope].sort_values("horizon")
        if scope_df.empty:
            continue
        x_positions = scope_df["horizon"].to_numpy() + offsets[scope]
        ax.plot(
            x_positions,
            scope_df["winrate"] * 100,
            marker="o",
            markersize=7,
            label=f"{scope.capitalize()} signal",
            color=p[scope],
            alpha=0.9,
        )
        for i, (_, row) in enumerate(scope_df.iterrows()):
            ax.vlines(
                x=x_positions[i],
                ymin=row["wilson_ci_lower"] * 100,
                ymax=row["wilson_ci_upper"] * 100,
                color=p[scope],
                lw=3,
                alpha=0.6,
            )

    ax.axhline(y=50, color=p["text_secondary"], ls="--", lw=1.5, alpha=0.7, label="50% baseline")
    ax.axvline(
        x=config.primary_horizon,
        color=p["accent"],
        ls=":",
        lw=2,
        alpha=0.8,
        label=f"Primary horizon (h={config.primary_horizon})",
    )
    ax.set_xlabel("Horizon (bars)")
    ax.set_ylabel("Winrate (%)")
    ax.set_title(f"Winrate by Horizon with 95% Wilson CI\n{config.asset.upper()} {config.timeframe}")
    ax.set_xticks(sorted(df["horizon"].unique()))
    apply_axes_style(ax)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_ylim(45, 75)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()


def plot_pnl_distribution(
    events_wide: pd.DataFrame,
    config: ResearchConfig,
    save_path: Path | None = None,
) -> None:
    """PnL distribution at the primary horizon (combined / long / short)."""
    set_plot_style(_theme_from_config(config))
    p = get_palette()

    h = config.primary_horizon
    pnl_col = f"strategy_pnl_{h}"
    if pnl_col not in events_wide.columns:
        print(f"Column {pnl_col} not found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        ("combined", "Combined Signals"),
        ("long", "Long Signals"),
        ("short", "Short Signals"),
    ]

    for ax, (scope, title) in zip(axes, panels):
        if scope == "combined":
            pnl = events_wide[pnl_col].dropna()
        else:
            pnl = events_wide.loc[events_wide["signal_type"] == scope, pnl_col].dropna()

        if pnl.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=p["text"], fontsize=14, transform=ax.transAxes)
            continue

        color = p[scope]
        ax.hist(pnl * 100, bins=50, density=True, alpha=0.5,
                color=color, edgecolor=p["background"], linewidth=0.5)

        kde = gaussian_kde(pnl * 100)
        x_range = np.linspace(pnl.min() * 100, pnl.max() * 100, 200)
        ax.plot(x_range, kde(x_range), color=color, lw=2, alpha=0.9)

        mean_pnl = pnl.mean() * 100
        median_pnl = pnl.median() * 100
        ax.axvline(mean_pnl, color=p["text"], ls="--", lw=1.5, alpha=0.9,
                   label=f"Mean: {mean_pnl:.3f}%")
        ax.axvline(median_pnl, color=p["text_secondary"], ls=":", lw=1.5, alpha=0.9,
                   label=f"Median: {median_pnl:.3f}%")
        ax.axvline(0, color=p["grid"], ls="-", lw=1, alpha=0.5)

        ax.set_xlabel("Strategy PnL (%)")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} | h={h}, n={len(pnl)}")
        ax.legend(fontsize=9, loc="upper right")

    apply_axes_style_all(axes)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()


def plot_methodology_comparison(
    stats_df: pd.DataFrame,
    config: ResearchConfig,
    save_path: Path | None = None,
) -> None:
    """Raw signals vs episodes across key metrics (overlapping sample)."""
    set_plot_style(_theme_from_config(config))
    p = get_palette()

    df = stats_df[stats_df["sample_type"] == "overlapping"].copy()
    if df.empty:
        print("No data for methodology comparison.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ("n_events", "Number of Events"),
        ("winrate", "Winrate"),
        ("mean_pnl", "Mean PnL"),
        ("profit_factor", "Profit Factor"),
    ]
    methods = ["raw_signals", "episodes"]

    for ax, (metric, title) in zip(axes.flat, metrics):
        labels, values, alphas = [], [], []
        for scope in _SCOPE_COLORS:
            for method in methods:
                subset = df[(df["signal_scope"] == scope) & (df["method"] == method)]
                if subset.empty:
                    continue
                value = subset[metric].mean()
                if np.isnan(value):
                    continue
                method_label = "Raw" if method == "raw_signals" else "Episodes"
                labels.append(f"{scope.capitalize()}\n{method_label}")
                values.append(value)
                alphas.append(0.35 if method == "raw_signals" else 0.85)

        if not values:
            continue

        x_positions = np.arange(len(values))
        bars = ax.bar(x_positions, values, color=p["accent"],
                      edgecolor=p["background"], linewidth=0.5, width=0.7)
        for bar, alpha_val in zip(bars, alphas):
            bar.set_alpha(alpha_val)
        for bar, val in zip(bars, values):
            display = f"{int(val)}" if metric == "n_events" else f"{val:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.02,
                    display, ha="center", va="bottom", fontsize=8,
                    color=p["text_secondary"])

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} Comparison (Overlapping)", fontsize=12)
        for i in range(1, len(_SCOPE_COLORS)):
            ax.axvline(x=i * len(methods) - 0.5, color=p["grid"], ls="-", lw=0.5, alpha=0.5)

    apply_axes_style_all(axes)
    plt.tight_layout(w_pad=3, h_pad=3)
    save_figure(fig, save_path)
    plt.show()


def plot_signal_counts_over_time(
    df: pd.DataFrame,
    config: ResearchConfig,
    save_path: Path | None = None,
) -> None:
    """Monthly long/short signal counts (temporal stability)."""
    set_plot_style(_theme_from_config(config))
    p = get_palette()

    if config.timestamp_col not in df.columns:
        print("Timestamp column not found.")
        return

    df_signals = df[df["signal_type"] != "none"].copy()
    if df_signals.empty:
        print("No signals found.")
        return

    ts = pd.to_datetime(df_signals[config.timestamp_col])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df_signals["year_month"] = ts.dt.to_period("M")
    monthly = df_signals.groupby(["year_month", "signal_type"]).size().unstack(fill_value=0)
    for col in ("long", "short"):
        if col not in monthly.columns:
            monthly[col] = 0
    monthly = monthly[["long", "short"]]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(monthly))
    width = 0.35
    ax.bar(x - width / 2, monthly["long"], width, label="Long signals",
           color=p["long"], alpha=0.7, edgecolor=p["background"], linewidth=0.5)
    ax.bar(x + width / 2, monthly["short"], width, label="Short signals",
           color=p["short"], alpha=0.7, edgecolor=p["background"], linewidth=0.5)

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Signals")
    ax.set_title(f"Signal Frequency Over Time\n{config.asset.upper()} {config.timeframe}")

    step = 6
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(t) for t in monthly.index[::step]],
                       rotation=45, ha="right", fontsize=9)
    apply_axes_style(ax)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()


def plot_clustering_distribution(
    long_runs: pd.DataFrame,
    short_runs: pd.DataFrame,
    save_path: Path | None = None,
    theme: str = "light",
) -> None:
    """Episode-length distribution for long and short signals."""
    set_plot_style(theme)
    p = get_palette()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [
        (long_runs, "Long Signal Episodes", p["long"]),
        (short_runs, "Short Signal Episodes", p["short"]),
    ]

    for ax, (runs, title, color) in zip(axes, panels):
        if runs.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=p["text"], transform=ax.transAxes)
            continue

        counts = runs["length"].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=color, alpha=0.7,
               edgecolor=p["background"], linewidth=1)
        for length, count in counts.items():
            ax.text(length, count, str(count), ha="center", va="bottom",
                    fontsize=9, color=p["text_secondary"])

        ax.set_xlabel("Episode Length (bars)")
        ax.set_ylabel("Number of Episodes")
        ax.set_title(title)

        total = len(runs)
        clustered = int((runs["length"] > 1).sum())
        pct = clustered / total * 100 if total else 0.0
        stats_text = f"Total episodes: {total}\nClustered (len>1): {clustered} ({pct:.1f}%)"
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=p["text_secondary"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor=p["panel"],
                          edgecolor=p["grid"], alpha=0.8))

    apply_axes_style_all(axes)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()