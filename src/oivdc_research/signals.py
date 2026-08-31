# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Signal generation for the exhaustion-reversal pattern.

A long signal fires when price falls, volume delta is negative and open
interest declines; a short signal is the mirror image. The combination
"directional move + falling OI" is the core exhaustion hypothesis.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import ResearchConfig

__all__ = [
    "get_signal_input_columns",
    "add_signal_columns",
    "summarize_signal_counts",
    "signal_counts_by_month",
    "signal_counts_by_year",
    "save_signals",
    "load_signals",
]


def get_signal_input_columns(
    df: pd.DataFrame,
    config: ResearchConfig,
) -> tuple[str, str, str]:
    """Resolve the columns used for signal generation.

    Prefers the cleaned columns (return_clean / delta_oi_clean) when present,
    otherwise falls back to the raw config columns.

    Returns:
        (return_col, volume_delta_col, delta_oi_col)
    """
    return_col = "return_clean" if "return_clean" in df.columns else config.return_col
    delta_oi_col = (
        "delta_oi_clean" if "delta_oi_clean" in df.columns else config.delta_oi_col
    )
    volume_delta_col = config.volume_delta_col

    required = [return_col, volume_delta_col, delta_oi_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for signal generation: {missing}")

    return return_col, volume_delta_col, delta_oi_col


def add_signal_columns(
    df: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Add long/short/combined signal columns.

    long_signal:  return < 0 AND volume_delta < 0 AND delta_oi < 0
    short_signal: return > 0 AND volume_delta > 0 AND delta_oi < 0

    Adds: long_signal, short_signal, signal (+1/-1/0), signal_type, is_signal.
    """
    df = df.copy()
    return_col, volume_delta_col, delta_oi_col = get_signal_input_columns(df, config)

    ret = df[return_col]
    volume_delta = df[volume_delta_col]
    delta_oi = df[delta_oi_col]

    # Strict signs; NaN comparisons already yield False.
    long_signal = ((ret < 0) & (volume_delta < 0) & (delta_oi < 0)).fillna(False).astype(bool)
    short_signal = ((ret > 0) & (volume_delta > 0) & (delta_oi < 0)).fillna(False).astype(bool)

    # return cannot be <0 and >0 simultaneously, but guard anyway.
    n_overlap = int((long_signal & short_signal).sum())
    if n_overlap > 0:
        raise ValueError(f"Found {n_overlap} rows where long and short signals overlap.")

    df["long_signal"] = long_signal
    df["short_signal"] = short_signal

    df["signal"] = 0
    df.loc[long_signal, "signal"] = 1
    df.loc[short_signal, "signal"] = -1
    df["signal"] = df["signal"].astype(np.int8)

    signal_type_values = np.select(
        condlist=[long_signal, short_signal],
        choicelist=["long", "short"],
        default="none",
    )
    df["signal_type"] = pd.Series(signal_type_values, index=df.index).astype("category")
    df["is_signal"] = df["signal"] != 0

    # Session-level metadata (not persisted to parquet).
    df.attrs["signal_return_col"] = return_col
    df.attrs["signal_volume_delta_col"] = volume_delta_col
    df.attrs["signal_delta_oi_col"] = delta_oi_col

    return df


def summarize_signal_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return overall long / short / none counts and percentages."""
    total = len(df)
    if total == 0:
        return pd.DataFrame(
            {
                "signal_type": ["long", "short", "none"],
                "count": [0, 0, 0],
                "pct": [0.0, 0.0, 0.0],
            }
        )

    if "signal_type" not in df.columns:
        raise ValueError("DataFrame does not contain 'signal_type'. Run add_signal_columns first.")

    counts = df["signal_type"].value_counts().rename("count")
    out = pd.DataFrame({"signal_type": ["long", "short", "none"]})
    out = out.merge(counts.reset_index(), on="signal_type", how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    out["pct"] = out["count"] / total * 100.0
    return out


def signal_counts_by_month(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Return long/short signal counts per calendar month."""
    return _signal_counts_by_period(df, timestamp_col, freq="M")


def signal_counts_by_year(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Return long/short signal counts per calendar year."""
    return _signal_counts_by_period(df, timestamp_col, freq="Y")


def _signal_counts_by_period(
    df: pd.DataFrame,
    timestamp_col: str,
    freq: str,
) -> pd.DataFrame:
    """Shared implementation for monthly/yearly signal counts."""
    if "signal_type" not in df.columns:
        raise ValueError("DataFrame does not contain 'signal_type'. Run add_signal_columns first.")
    if timestamp_col not in df.columns:
        raise ValueError(f"DataFrame does not contain timestamp column: {timestamp_col}")

    signals = df.loc[df["signal_type"] != "none", [timestamp_col, "signal_type"]].copy()
    col = "year_month" if freq == "M" else "year"

    if signals.empty:
        return pd.DataFrame(columns=[col, "long", "short", "total"])

    ts = signals[timestamp_col]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    if freq == "M":
        signals[col] = ts.dt.to_period("M").astype(str)
    else:
        signals[col] = ts.dt.year

    grouped = signals.groupby([col, "signal_type"]).size().unstack(fill_value=0)
    for c in ("long", "short"):
        if c not in grouped.columns:
            grouped[c] = 0
    grouped = grouped[["long", "short"]]
    grouped["total"] = grouped["long"] + grouped["short"]
    return grouped.reset_index()


def save_signals(df: pd.DataFrame, config: ResearchConfig) -> Path:
    """Persist the signals dataset to parquet."""
    path = config.signals_path
    path.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()
    if "signal_type" in df_to_save.columns:
        df_to_save["signal_type"] = df_to_save["signal_type"].astype(str)

    df_to_save.to_parquet(path, index=False)
    return path


def load_signals(config: ResearchConfig) -> pd.DataFrame:
    """Load the previously saved signals dataset."""
    path = config.signals_path
    if not path.exists():
        raise FileNotFoundError(
            f"Signals dataset not found: {path}. Run signal generation first."
        )
    return pd.read_parquet(path)