# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Loading, validation and preparation of the 5m dataset.

Mirrors the 1h validation philosophy (integrity + continuity) at 5m
resolution, so the 5m data is proven stable. The cleaned
dataset is persisted once (notebook 01) and reused downstream.

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ResearchConfig
from ..data import ValidationCheck, ValidationReport

__all__ = [
    "load_5m",
    "validate_5m",
    "prepare_5m_data",
    "save_5m_data",
    "load_5m_data",
    "hour_paths",
]

_PRICE_COLS = ["open", "high", "low", "close"]
_NUMERIC_5M = ["open", "high", "low", "close", "volume"]


def load_5m(config: ResearchConfig) -> pd.DataFrame:
    """Load the raw 5m CSV and coerce base dtypes."""
    if not config.raw_5m_path.exists():
        raise FileNotFoundError(
            f"Raw 5m CSV not found: {config.raw_5m_path}. "
            "Prepare it with the BVC repository (see data/README.md)."
        )
    df = pd.read_csv(config.raw_5m_path)
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], utc=True)
    cols = _NUMERIC_5M + (["volume_delta"] if "volume_delta" in df.columns else [])
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_5m(
    df5m: pd.DataFrame,
    config: ResearchConfig,
    df1h: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Full 5m data-quality check suite (critical + warning)."""
    checks: list[ValidationCheck] = []
    df = df5m.copy()
    ts = config.timestamp_col

    df = df.sort_values(ts, kind="mergesort", na_position="last").reset_index(drop=True)
    n_rows = len(df)
    checks.append(ValidationCheck(
        name="min_rows", passed=n_rows >= config.min_rows, severity="critical",
        n_issues=0 if n_rows >= config.min_rows else 1,
        message=f"Rows: {n_rows}. Minimum required: {config.min_rows}.",
    ))
    if n_rows == 0:
        return df, ValidationReport(checks=checks)

    ts_null = df[ts].isna()
    n_ts_null = int(ts_null.sum())
    checks.append(ValidationCheck(
        name="timestamp_not_null", passed=n_ts_null == 0, severity="critical",
        n_issues=n_ts_null,
        message="No null timestamps." if n_ts_null == 0 else f"{n_ts_null} null timestamps.",
    ))
    df = df.loc[~ts_null].copy()

    dup = df[ts].duplicated(keep="first")
    n_dup = int(dup.sum())
    checks.append(ValidationCheck(
        name="timestamp_duplicates", passed=n_dup == 0, severity="critical",
        n_issues=n_dup,
        message="No duplicate timestamps." if n_dup == 0 else f"{n_dup} duplicates (keep first).",
        examples=df.loc[dup, ts].head(5).tolist(),
    ))
    df = df.loc[~dup].copy()
    df = df.sort_values(ts, kind="mergesort").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)

    num = [c for c in _NUMERIC_5M if c in df.columns]
    n_nan = int(df[num].isna().sum().sum())
    checks.append(ValidationCheck(
        name="numeric_no_nan", passed=n_nan == 0, severity="critical",
        n_issues=n_nan,
        message="No NaN in numeric columns." if n_nan == 0 else f"{n_nan} NaN values.",
    ))
    n_inf = int(np.isinf(df[num].to_numpy()).sum())
    checks.append(ValidationCheck(
        name="numeric_finite", passed=n_inf == 0, severity="critical",
        n_issues=n_inf,
        message="All numeric values finite." if n_inf == 0 else f"{n_inf} infinite values.",
    ))
    df["_ok_numeric"] = np.isfinite(df[num].to_numpy()).all(axis=1)

    price_ok = (df[_PRICE_COLS] > 0).to_numpy().all(axis=1)
    n_bad_price = int((~price_ok).sum())
    checks.append(ValidationCheck(
        name="prices_positive", passed=n_bad_price == 0, severity="critical",
        n_issues=n_bad_price,
        message="All prices positive." if n_bad_price == 0 else f"{n_bad_price} non-positive rows.",
        examples=df.loc[~price_ok, ts].head(5).tolist(),
    ))
    df["_ok_prices_positive"] = price_ok

    vol_ok = (df["volume"] >= 0).to_numpy() if "volume" in df.columns else np.ones(len(df), bool)
    n_neg_vol = int((~vol_ok).sum())
    checks.append(ValidationCheck(
        name="volume_nonnegative", passed=n_neg_vol == 0, severity="critical",
        n_issues=n_neg_vol,
        message="volume non-negative." if n_neg_vol == 0 else f"{n_neg_vol} negative volume rows.",
    ))
    df["_ok_volume"] = vol_ok

    tol = config.ohlc_atol
    ohlc_ok = (
        (df["high"] + tol >= df["low"]) & (df["high"] + tol >= df["open"])
        & (df["high"] + tol >= df["close"]) & (df["low"] - tol <= df["open"])
        & (df["low"] - tol <= df["close"])
    ).to_numpy()
    n_ohlc = int((~ohlc_ok).sum())
    checks.append(ValidationCheck(
        name="ohlc_consistency", passed=n_ohlc == 0, severity="critical",
        n_issues=n_ohlc,
        message="OHLC consistent." if n_ohlc == 0 else f"{n_ohlc} inconsistent OHLC rows.",
        examples=df.loc[~ohlc_ok, ts].head(5).tolist(),
    ))
    df["_ok_ohlc"] = ohlc_ok

    # 5m grid alignment.
    misaligned = ((df[ts].dt.minute % 5 != 0) | (df[ts].dt.second != 0)).to_numpy()
    n_mis = int(misaligned.sum())
    checks.append(ValidationCheck(
        name="grid_aligned_5m", passed=n_mis == 0, severity="critical",
        n_issues=n_mis,
        message="All bars on the 5m grid." if n_mis == 0 else f"{n_mis} off-grid bars.",
        examples=df.loc[misaligned, ts].head(5).tolist(),
    ))

    # Continuity.
    if len(df) >= 2:
        expected = pd.Timedelta(minutes=5)
        gaps = df[ts].diff().dropna()
        n_small = int((gaps < expected).sum())
        n_large = int((gaps > expected).sum())
        large = gaps[gaps > expected]
        missing = int((np.floor(large / expected) - 1).clip(lower=0).sum()) if not large.empty else 0
        checks.append(ValidationCheck(
            name="no_overlapping_bars", passed=n_small == 0, severity="critical",
            n_issues=n_small,
            message="No overlapping bars." if n_small == 0 else f"{n_small} overlapping bars.",
        ))
        checks.append(ValidationCheck(
            name="timestamp_frequency", passed=(n_small + n_large) == 0, severity="warning",
            n_issues=n_large,
            message=(f"5m cadence clean. Larger gaps: {n_large}. "
                     f"Approx missing bars: {missing}."),
            examples=df.loc[large.index[:5], ts].tolist(),
        ))
    if df1h is not None:
        s5 = set(df[ts])
        missing_1h = int(sum(t not in s5 for t in df1h[ts]))
        checks.append(ValidationCheck(
            name="coverage_vs_1h", passed=missing_1h == 0, severity="warning",
            n_issues=missing_1h,
            message="5m covers the full 1h range." if missing_1h == 0
            else f"{missing_1h} 1h bars not covered by 5m.",
        ))

    return df, ValidationReport(checks=checks)


def prepare_5m_data(df_checked: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Clean the 5m dataset for downstream use."""
    df = df_checked.copy()
    if df.empty:
        return df
    df = df.dropna(subset=[config.timestamp_col])
    num = [c for c in _NUMERIC_5M if c in df.columns]
    df = df[np.isfinite(df[num].to_numpy()).all(axis=1)]
    for flag in ("_ok_prices_positive", "_ok_volume", "_ok_ohlc"):
        if flag in df.columns:
            df = df[df[flag]]
    df = df.sort_values(config.timestamp_col, kind="mergesort").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)
    return df


def save_5m_data(df: pd.DataFrame, config: ResearchConfig) -> Path:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.data_5m_path, index=False)
    return config.data_5m_path


def load_5m_data(config: ResearchConfig) -> pd.DataFrame:
    """Load the prepared 5m parquet (produced by notebook 01)."""
    if not config.data_5m_path.exists():
        raise FileNotFoundError(
            f"5m parquet not found: {config.data_5m_path}. Run notebook 01 first."
        )
    return pd.read_parquet(config.data_5m_path)


def hour_paths(df5m: pd.DataFrame, bars_per_hour: int = 12) -> dict:
    """Cumulative intrabar path per 1h bar: path_i = close_i / open_hour - 1."""
    df = df5m.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    out = {}
    for hour, g in df.groupby("hour"):
        if len(g) != bars_per_hour:
            continue
        o = g["open"].iloc[0]
        if o <= 0 or not np.isfinite(o):
            continue
        out[hour] = g["close"].to_numpy() / o - 1.0
    return out