# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Statistical evaluation of event PnL.

All metrics are computed on direction-adjusted PnL (strategy_pnl), so a
positive value means the signal predicted the move correctly regardless of
side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from .config import ResearchConfig
from .event_selection import melt_events_long, select_nonoverlapping_events

__all__ = [
    "compute_event_metrics",
    "compute_statistics_for_events_wide",
    "format_stats_table",
    "summarize_mfe_mae_symmetry",
]

_EMPTY_RESULT = {
    "n_events": 0,
    "winrate": np.nan,
    "mean_pnl": np.nan,
    "median_pnl": np.nan,
    "std_pnl": np.nan,
    "t_stat": np.nan,
    "p_value": np.nan,
    "wilson_ci_lower": np.nan,
    "wilson_ci_upper": np.nan,
    "bootstrap_ci_lower": np.nan,
    "bootstrap_ci_upper": np.nan,
    "avg_win": np.nan,
    "avg_loss": np.nan,
    "profit_factor": np.nan,
    "expectancy": np.nan,
}


def compute_event_metrics(
    pnl: pd.Series,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Full metric set for a single direction-adjusted PnL series.

    Includes winrate (+Wilson CI), mean/median/std, one-sided t-test,
    bootstrap CI of the mean, avg win/loss, profit factor and expectancy.
    """
    pnl = pnl.dropna()
    n = len(pnl)
    if n == 0:
        return dict(_EMPTY_RESULT)

    wins_mask = pnl > 0
    n_wins = int(wins_mask.sum())
    winrate = n_wins / n

    wilson_lower, wilson_upper = proportion_confint(
        count=n_wins, nobs=n, alpha=1 - ci_level, method="wilson"
    )

    mean_pnl = float(pnl.mean())
    median_pnl = float(pnl.median())
    std_pnl = float(pnl.std())

    wins_values = pnl[wins_mask]
    losses_values = pnl[~wins_mask]
    avg_win = float(wins_values.mean()) if n_wins > 0 else 0.0
    avg_loss = float(losses_values.mean()) if len(losses_values) > 0 else 0.0

    sum_wins = float(wins_values.sum()) if n_wins > 0 else 0.0
    sum_losses = float((-losses_values).sum()) if len(losses_values) > 0 else 0.0
    if sum_losses > 0:
        profit_factor = sum_wins / sum_losses
    else:
        profit_factor = np.inf if sum_wins > 0 else np.nan

    # One-sided t-test: H0 mean=0 vs H1 mean>0.
    if n > 1 and std_pnl > 0:
        t_stat_raw, p_two = stats.ttest_1samp(pnl.to_numpy(), 0.0)
        t_stat = float(t_stat_raw)
        p_value = float(p_two / 2) if t_stat > 0 else float(1 - p_two / 2)
    else:
        t_stat = np.nan
        p_value = np.nan

    # Bootstrap CI of the mean.
    rng = np.random.default_rng(random_state)
    pnl_values = pnl.to_numpy()
    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(pnl_values, size=n, replace=True)
        bootstrap_means[i] = sample.mean()
    alpha = 1 - ci_level
    bootstrap_ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    bootstrap_ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {
        "n_events": n,
        "winrate": winrate,
        "mean_pnl": mean_pnl,
        "median_pnl": median_pnl,
        "std_pnl": std_pnl,
        "t_stat": t_stat,
        "p_value": p_value,
        "wilson_ci_lower": float(wilson_lower),
        "wilson_ci_upper": float(wilson_upper),
        "bootstrap_ci_lower": bootstrap_ci_lower,
        "bootstrap_ci_upper": bootstrap_ci_upper,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": mean_pnl,
    }


def compute_statistics_for_events_wide(
    events_wide: pd.DataFrame,
    config: ResearchConfig,
    method_name: str,
    sample_type: str = "overlapping",  # "overlapping" | "nonoverlapping"
    n_bootstrap: int = 1000,
    random_state: int = 42,
    bar_col: str = "trigger_bar_index",
) -> pd.DataFrame:
    """Main statistics driver.

    For each horizon (applying non-overlap first when requested), melt to long
    format and compute metrics for long / short / combined scopes.
    """
    if sample_type not in {"overlapping", "nonoverlapping"}:
        raise ValueError(f"Unknown sample_type: {sample_type}")

    results = []
    for horizon in config.horizons:
        if sample_type == "nonoverlapping":
            events_for_horizon = select_nonoverlapping_events(
                events_wide, horizon=horizon, gap=horizon, bar_col=bar_col
            )
        else:
            events_for_horizon = events_wide
        if events_for_horizon.empty:
            continue

        events_long = melt_events_long(events_for_horizon, config)
        events_h = events_long[events_long["horizon"] == horizon]
        if events_h.empty:
            continue

        for sig_scope in ["long", "short", "combined"]:
            sub = (
                events_h
                if sig_scope == "combined"
                else events_h[events_h["signal_type"] == sig_scope]
            )
            if sub.empty:
                continue

            pnl = sub["strategy_pnl"].dropna()
            metrics = compute_event_metrics(
                pnl, n_bootstrap=n_bootstrap, random_state=random_state
            )
            metrics.update(
                horizon=horizon,
                signal_scope=sig_scope,
                method=method_name,
                sample_type=sample_type,
            )
            results.append(metrics)

    return pd.DataFrame(results) if results else pd.DataFrame()


def format_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """Order and sort the statistics table for readable output."""
    if df.empty:
        return df

    first_cols = [
        "method",
        "sample_type",
        "signal_scope",
        "horizon",
        "n_events",
        "winrate",
        "mean_pnl",
        "median_pnl",
        "std_pnl",
        "t_stat",
        "p_value",
    ]
    existing_first = [c for c in first_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[existing_first + other_cols].copy()

    sort_cols = [c for c in first_cols[:4] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def summarize_mfe_mae_symmetry(
    events: pd.DataFrame,
    horizons,
    thresholds=(0.0005, 0.001, 0.002, 0.003),
) -> pd.DataFrame:
    """Touch-probability symmetry diagnostic.

    For each horizon and threshold tau: P(MFE > tau) vs P(MAE < -tau) and the
    ratio. Ratio ~1.0 = symmetric random walk (no directed impulse);
    ratio >~1.2 = directed favorable impulse.
    """
    rows = []
    for h in horizons:
        if f"mfe_{h}" not in events.columns or f"mae_{h}" not in events.columns:
            continue
        sub = events.dropna(subset=[f"mfe_{h}", f"mae_{h}"])
        if sub.empty:
            continue
        mfe = sub[f"mfe_{h}"]
        mae = sub[f"mae_{h}"]
        for tau in thresholds:
            p_mfe = float((mfe > tau).mean())
            p_mae = float((mae < -tau).mean())
            rows.append(
                {
                    "horizon": h,
                    "threshold": tau,
                    "p_mfe": p_mfe,
                    "p_mae": p_mae,
                    "ratio": p_mfe / p_mae if p_mae > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)