# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Signal clustering: run detection and episode labelling.

Consecutive signal bars form "episodes" (clusters). Because bars inside an
episode are not independent, treating each as a separate event inflates
apparent significance. This module detects runs and assigns episode ids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "analyze_signal_runs",
    "get_clustering_summary",
    "add_episode_ids",
]

_RUN_COLUMNS = [
    "run_id",
    "start_bar_index",
    "end_bar_index",
    "length",
    "start_timestamp",
    "end_timestamp",
]


def analyze_signal_runs(
    df: pd.DataFrame,
    signal_col: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Detect contiguous runs of a given signal.

    Contiguity is strict: bar_index[i] == bar_index[i-1] + 1.

    Returns a DataFrame describing each run (id, start/end bar and timestamp,
    length).
    """
    if signal_col not in df.columns:
        raise ValueError(f"Column {signal_col} not found in DataFrame.")
    if "bar_index" not in df.columns:
        raise ValueError("Column 'bar_index' not found in DataFrame.")

    df_sig = df.loc[df[signal_col].astype(bool), ["bar_index", timestamp_col]].copy()
    if df_sig.empty:
        return pd.DataFrame(columns=_RUN_COLUMNS)

    # A gap in bar_index marks the start of a new run.
    df_sig["prev_bar"] = df_sig["bar_index"].shift(1)
    df_sig["is_new_run"] = (
        (df_sig["bar_index"] - df_sig["prev_bar"] != 1) | df_sig["prev_bar"].isna()
    )
    df_sig["run_id"] = df_sig["is_new_run"].cumsum()

    runs = (
        df_sig.groupby("run_id")
        .agg(
            start_bar_index=("bar_index", "first"),
            end_bar_index=("bar_index", "last"),
            length=("bar_index", "count"),
            start_timestamp=(timestamp_col, "first"),
            end_timestamp=(timestamp_col, "last"),
        )
        .reset_index()
    )
    return runs


def get_clustering_summary(
    long_runs: pd.DataFrame,
    short_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Distribution of episode lengths for long and short signals."""
    summary = []
    for label, runs in [("long", long_runs), ("short", short_runs)]:
        if runs.empty:
            continue
        counts = runs["length"].value_counts().sort_index()
        for length, count in counts.items():
            summary.append(
                {
                    "signal_type": label,
                    "run_length": int(length),
                    "num_episodes": int(count),
                    "total_signals_in_episodes": int(count * length),
                }
            )

    df_summary = pd.DataFrame(summary)
    if not df_summary.empty:
        for sig_type in ["long", "short"]:
            mask = df_summary["signal_type"] == sig_type
            total_sigs = df_summary.loc[mask, "total_signals_in_episodes"].sum()
            df_summary.loc[mask, "pct_of_signals"] = (
                df_summary.loc[mask, "total_signals_in_episodes"] / total_sigs * 100
                if total_sigs > 0
                else 0.0
            )
    return df_summary


def add_episode_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add episode_id_long / episode_id_short columns.

    Bars without a signal get NaN. Episode ids are contiguous integers per
    run, allowing bars to be linked back to their episode.
    """
    df = df.copy()

    for sig_col, ep_col in [
        ("long_signal", "episode_id_long"),
        ("short_signal", "episode_id_short"),
    ]:
        df[ep_col] = np.nan
        mask = df[sig_col].astype(bool)
        if not mask.any():
            continue

        indices = df.index[mask]
        bar_indices = df.loc[mask, "bar_index"].to_numpy()

        is_new_run = np.zeros(len(bar_indices), dtype=bool)
        is_new_run[0] = True
        if len(bar_indices) > 1:
            is_new_run[1:] = np.diff(bar_indices) != 1
        run_ids = np.cumsum(is_new_run)

        df.loc[indices, ep_col] = run_ids

    return df