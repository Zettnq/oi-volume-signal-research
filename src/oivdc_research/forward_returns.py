# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Forward returns and strategy PnL construction.

Entry convention: the signal is known at the close of bar t, so the position
is entered at the open of bar t+1 and held for h bars (exit at close t+h).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ResearchConfig

__all__ = [
    "add_entry_info",
    "add_forward_returns",
    "add_strategy_pnl",
    "add_forward_return_columns",
    "summarize_forward_return_availability",
    "save_forward_returns",
    "load_forward_returns",
]


def add_entry_info(df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Add entry timing/price columns.

    entry_time  = timestamp[t+1]
    entry_price = open[t+1]
    entry_bar_index = bar_index[t+1]
    """
    df = df.copy()

    if config.entry_mode != "next_open":
        raise ValueError(
            f"Unsupported entry_mode: {config.entry_mode}. "
            "Currently only 'next_open' is supported."
        )

    required = [config.timestamp_col, "open", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for entry info: {missing}")

    if "bar_index" not in df.columns:
        df["bar_index"] = np.arange(len(df), dtype=np.int64)

    df["entry_bar_index"] = df["bar_index"].shift(-1).astype("Int64")
    df["entry_time"] = df[config.timestamp_col].shift(-1)
    df["entry_price"] = df["open"].shift(-1)
    return df


def add_forward_returns(df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Add forward returns for each horizon h.

    For each h creates exit_time_{h}, exit_price_{h}, exit_bar_index_{h} and
    fwd_ret_{h}, where fwd_ret_h = close[t+h] / open[t+1] - 1.
    """
    df = df.copy()

    if "entry_price" not in df.columns:
        df = add_entry_info(df, config)
    if "bar_index" not in df.columns:
        df["bar_index"] = np.arange(len(df), dtype=np.int64)

    for h in config.horizons:
        exit_time_col = f"exit_time_{h}"
        exit_price_col = f"exit_price_{h}"
        fwd_ret_col = f"fwd_ret_{h}"

        df[exit_time_col] = df[config.timestamp_col].shift(-h)
        df[exit_price_col] = df["close"].shift(-h)
        df[f"exit_bar_index_{h}"] = df["bar_index"].shift(-h).astype("Int64")

        entry_price = df["entry_price"]
        exit_price = df[exit_price_col]

        with np.errstate(divide="ignore", invalid="ignore"):
            fwd_ret = exit_price / entry_price - 1.0

        invalid_mask = entry_price.isna() | exit_price.isna() | (entry_price <= 0)
        fwd_ret[invalid_mask] = np.nan

        df[fwd_ret_col] = fwd_ret

    return df


def add_strategy_pnl(df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Add direction-adjusted PnL: strategy_pnl_h = signal * fwd_ret_h."""
    df = df.copy()

    if "signal" not in df.columns:
        raise ValueError(
            "DataFrame does not contain 'signal'. Run add_signal_columns first."
        )

    for h in config.horizons:
        fwd_ret_col = f"fwd_ret_{h}"
        if fwd_ret_col not in df.columns:
            raise ValueError(
                f"DataFrame does not contain '{fwd_ret_col}'. "
                "Run add_forward_returns before add_strategy_pnl."
            )
        df[f"strategy_pnl_{h}"] = df["signal"] * df[fwd_ret_col]

    return df


def add_forward_return_columns(
    df: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Full pipeline: entry info + forward returns + strategy PnL."""
    df = add_entry_info(df, config)
    df = add_forward_returns(df, config)
    df = add_strategy_pnl(df, config)
    return df


def summarize_forward_return_availability(
    df: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Report how many valid forward returns exist per horizon.

    The last h bars have no future window, so some NaNs are expected.
    """
    total_bars = len(df)

    if "is_signal" in df.columns:
        signal_mask = df["is_signal"].astype(bool)
    elif "signal" in df.columns:
        signal_mask = df["signal"] != 0
    else:
        signal_mask = None

    rows = []
    for h in config.horizons:
        fwd_ret_col = f"fwd_ret_{h}"
        if fwd_ret_col not in df.columns:
            continue

        valid_all = int(df[fwd_ret_col].notna().sum())

        if signal_mask is not None:
            n_signals = int(signal_mask.sum())
            valid_signals = int(df.loc[signal_mask, fwd_ret_col].notna().sum())
        else:
            n_signals = valid_signals = None

        rows.append(
            {
                "horizon": h,
                "total_bars": total_bars,
                "valid_all": valid_all,
                "missing_all": total_bars - valid_all,
                "n_signals": n_signals,
                "valid_signals": valid_signals,
                "missing_signals": (
                    n_signals - valid_signals if n_signals is not None else None
                ),
            }
        )

    return pd.DataFrame(rows)


def save_forward_returns(df: pd.DataFrame, config: ResearchConfig) -> Path:
    """Persist the forward-returns dataset to parquet."""
    path = config.forward_returns_path
    path.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()
    for col in df_to_save.select_dtypes(include=["category"]).columns:
        df_to_save[col] = df_to_save[col].astype(str)

    df_to_save.to_parquet(path, index=False)
    return path


def load_forward_returns(config: ResearchConfig) -> pd.DataFrame:
    """Load the previously saved forward-returns dataset."""
    path = config.forward_returns_path
    if not path.exists():
        raise FileNotFoundError(
            f"Forward returns dataset not found: {path}. "
            "Run forward returns generation first."
        )
    return pd.read_parquet(path)