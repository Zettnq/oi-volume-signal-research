# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Loading, validation and cleaning of the raw 1h dataset."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ResearchConfig


@dataclass
class ValidationCheck:
    """Result of a single data-quality check."""

    name: str
    passed: bool
    severity: str  # "critical" or "warning"
    n_issues: int
    message: str
    examples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "n_issues": self.n_issues,
            "message": self.message,
            "examples": "; ".join(str(x) for x in self.examples[:5]),
        }


@dataclass
class ValidationReport:
    """Collection of data-quality checks."""

    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def critical_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "critical")

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.checks])

    def show(self) -> None:
        with pd.option_context("display.max_colwidth", None):
            print(self.to_frame().to_string(index=False))


def load_raw_data(config: ResearchConfig) -> pd.DataFrame:
    """Load the raw CSV and coerce base dtypes."""
    if not config.raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {config.raw_csv_path}")

    df = pd.read_csv(config.raw_csv_path)

    missing = [c for c in config.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], utc=True)
    for col in config.numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_data(
    df: pd.DataFrame,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Run all data-quality checks.

    Returns the dataframe augmented with diagnostic `_ok_*` columns and a
    ValidationReport.
    """
    checks: list[ValidationCheck] = []
    df = df.copy()

    df = df.sort_values(config.timestamp_col, kind="mergesort", na_position="last")
    df = df.reset_index(drop=True)

    n_rows_raw = len(df)
    checks.append(ValidationCheck(
        name="min_rows",
        passed=n_rows_raw >= config.min_rows,
        severity="critical",
        n_issues=0 if n_rows_raw >= config.min_rows else 1,
        message=f"Rows: {n_rows_raw}. Minimum required: {config.min_rows}.",
    ))
    if n_rows_raw == 0:
        return df, ValidationReport(checks=checks)

    # 1. Timestamps must not be null.
    ts_null = df[config.timestamp_col].isna()
    n_ts_null = int(ts_null.sum())
    checks.append(ValidationCheck(
        name="timestamp_not_null",
        passed=n_ts_null == 0,
        severity="critical",
        n_issues=n_ts_null,
        message="No null timestamps." if n_ts_null == 0
        else f"Found {n_ts_null} null timestamps.",
    ))
    df = df.loc[~ts_null].copy()

    # 2. Duplicate timestamps.
    dup_mask = df[config.timestamp_col].duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    checks.append(ValidationCheck(
        name="timestamp_duplicates",
        passed=n_dup == 0,
        severity="critical",
        n_issues=n_dup,
        message="No duplicate timestamps." if n_dup == 0
        else f"Found {n_dup} duplicate timestamps. Keeping first occurrence.",
        examples=df.loc[dup_mask, config.timestamp_col].head(5).tolist(),
    ))
    df = df.loc[~dup_mask].copy()

    df = df.sort_values(config.timestamp_col, kind="mergesort").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)

    if len(df) == 0:
        checks.append(ValidationCheck(
            name="data_not_empty_after_cleaning",
            passed=False,
            severity="critical",
            n_issues=1,
            message="DataFrame is empty after removing null timestamps and duplicates.",
        ))
        return df, ValidationReport(checks=checks)

    # 3. NaN / inf in numeric columns.
    numeric_cols = [c for c in config.numeric_columns if c in df.columns]
    nan_counts = df[numeric_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    n_nans = int(nan_counts.sum())
    checks.append(ValidationCheck(
        name="numeric_no_nan",
        passed=n_nans == 0,
        severity="critical",
        n_issues=n_nans,
        message="No NaN values in numeric columns." if n_nans == 0
        else f"Found {n_nans} NaN values in numeric columns.",
        examples=[str(nan_counts.to_dict())],
    ))

    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_counts = inf_counts[inf_counts > 0]
    n_infs = int(inf_counts.sum())
    checks.append(ValidationCheck(
        name="numeric_finite",
        passed=n_infs == 0,
        severity="critical",
        n_issues=n_infs,
        message="All numeric values are finite." if n_infs == 0
        else f"Found {n_infs} infinite values in numeric columns.",
        examples=[str(inf_counts.to_dict())],
    ))
    df["_ok_numeric"] = np.isfinite(df[numeric_cols]).all(axis=1)

    # 4. Prices must be positive.
    price_cols = [c for c in config.price_columns if c in df.columns]
    price_ok = (df[price_cols] > 0).all(axis=1)
    n_bad_price = int((~price_ok).sum())
    checks.append(ValidationCheck(
        name="prices_positive",
        passed=n_bad_price == 0,
        severity="critical",
        n_issues=n_bad_price,
        message="All prices are positive." if n_bad_price == 0
        else f"Found {n_bad_price} rows with non-positive prices.",
        examples=df.loc[~price_ok, config.timestamp_col].head(5).tolist(),
    ))
    df["_ok_prices_positive"] = price_ok

    # 5. Volume / open interest must be non-negative.
    nonneg = pd.Series(True, index=df.index)
    if config.volume_col in df.columns:
        nonneg &= df[config.volume_col] >= 0
    if config.open_interest_col in df.columns:
        nonneg &= df[config.open_interest_col] >= 0
    n_neg = int((~nonneg).sum())
    checks.append(ValidationCheck(
        name="volume_and_oi_nonnegative",
        passed=n_neg == 0,
        severity="critical",
        n_issues=n_neg,
        message="volume and open_interest are non-negative." if n_neg == 0
        else f"Found {n_neg} rows with negative volume/open_interest.",
        examples=df.loc[~nonneg, config.timestamp_col].head(5).tolist(),
    ))
    df["_ok_nonnegative"] = nonneg

    # 6. OHLC consistency.
    tol = config.ohlc_atol
    if all(c in df.columns for c in ("open", "high", "low", "close")):
        ohlc_ok = (
            (df["high"] + tol >= df["low"])
            & (df["high"] + tol >= df["open"])
            & (df["high"] + tol >= df["close"])
            & (df["low"] - tol <= df["open"])
            & (df["low"] - tol <= df["close"])
        )
    else:
        ohlc_ok = pd.Series(False, index=df.index)
    n_ohlc_bad = int((~ohlc_ok).sum())
    checks.append(ValidationCheck(
        name="ohlc_consistency",
        passed=n_ohlc_bad == 0,
        severity="critical",
        n_issues=n_ohlc_bad,
        message="OHLC relations are consistent." if n_ohlc_bad == 0
        else f"Found {n_ohlc_bad} rows with inconsistent OHLC.",
        examples=df.loc[~ohlc_ok, config.timestamp_col].head(5).tolist(),
    ))
    df["_ok_ohlc"] = ohlc_ok

    # 7. Bar frequency and gaps.
    if len(df) >= 2:
        expected = config.expected_timedelta
        gaps = df[config.timestamp_col].diff().dropna()
        n_not_expected = int((gaps != expected).sum())
        n_smaller = int((gaps < expected).sum())
        n_larger = int((gaps > expected).sum())

        large = gaps[gaps > expected]
        if not large.empty:
            missing_bars = int((np.floor(large / expected) - 1).clip(lower=0).sum())
        else:
            missing_bars = 0

        checks.append(ValidationCheck(
            name="timestamp_frequency",
            passed=n_not_expected == 0,
            severity="warning",
            n_issues=n_not_expected,
            message=(
                f"Expected bar delta: {expected}. "
                f"Gaps != expected: {n_not_expected}. "
                f"Smaller gaps: {n_smaller}. Larger gaps: {n_larger}. "
                f"Approx missing bars: {missing_bars}."
            ),
            examples=df.loc[large.index[:5], config.timestamp_col].tolist(),
        ))
    else:
        checks.append(ValidationCheck(
            name="timestamp_frequency",
            passed=False,
            severity="warning",
            n_issues=1,
            message="Not enough rows to check timestamp frequency.",
        ))

    # 8. Provided return vs close-to-close computed return.
    df["return_computed"] = df["close"].pct_change()
    return_ok = pd.Series(True, index=df.index)
    verifiable = df["return_computed"].notna()
    return_ok[verifiable] = (
        df.loc[verifiable, config.return_col].notna()
        & (
            (df.loc[verifiable, config.return_col] - df.loc[verifiable, "return_computed"]).abs()
            <= config.return_atol
        )
    )
    n_return_bad = int((~return_ok).sum())
    checks.append(ValidationCheck(
        name="return_consistency",
        passed=n_return_bad == 0,
        severity="warning",
        n_issues=n_return_bad,
        message="Provided return is consistent with computed return." if n_return_bad == 0
        else f"Found {n_return_bad} rows where provided return differs from computed.",
        examples=df.loc[~return_ok, config.timestamp_col].head(5).tolist(),
    ))
    df["_ok_return"] = return_ok

    # 9. Provided delta_oi vs diff(open_interest).
    df["delta_oi_computed"] = df[config.open_interest_col].diff()
    delta_ok = pd.Series(True, index=df.index)
    verifiable = df["delta_oi_computed"].notna()
    delta_ok[verifiable] = (
        df.loc[verifiable, config.delta_oi_col].notna()
        & (
            (df.loc[verifiable, config.delta_oi_col] - df.loc[verifiable, "delta_oi_computed"]).abs()
            <= config.delta_oi_atol
        )
    )
    n_delta_bad = int((~delta_ok).sum())
    checks.append(ValidationCheck(
        name="delta_oi_consistency",
        passed=n_delta_bad == 0,
        severity="warning",
        n_issues=n_delta_bad,
        message="Provided delta_oi is consistent with diff(open_interest)." if n_delta_bad == 0
        else f"Found {n_delta_bad} rows where provided delta_oi differs from computed.",
        examples=df.loc[~delta_ok, config.timestamp_col].head(5).tolist(),
    ))
    df["_ok_delta_oi"] = delta_ok

    return df, ValidationReport(checks=checks)


def prepare_validated_data(
    df_checked: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Produce the clean dataset used downstream.

    Drops invalid rows, builds return_clean / delta_oi_clean and recomputes
    bar_index.
    """
    df = df_checked.copy()
    if df.empty:
        return df

    df = df.dropna(subset=[config.timestamp_col])

    numeric_cols = [c for c in config.numeric_columns if c in df.columns]
    df = df[np.isfinite(df[numeric_cols]).all(axis=1)]

    if "_ok_prices_positive" in df.columns:
        df = df[df["_ok_prices_positive"]]
    if "_ok_nonnegative" in df.columns:
        df = df[df["_ok_nonnegative"]]
    if "_ok_ohlc" in df.columns:
        df = df[df["_ok_ohlc"]]

    if config.prefer_computed_return and "return_computed" in df.columns:
        df["return_clean"] = df["return_computed"].fillna(df[config.return_col])
    else:
        df["return_clean"] = df[config.return_col]
        if "_ok_return" in df.columns:
            df = df[df["_ok_return"]]

    if config.prefer_computed_delta_oi and "delta_oi_computed" in df.columns:
        df["delta_oi_clean"] = df["delta_oi_computed"].fillna(df[config.delta_oi_col])
    else:
        df["delta_oi_clean"] = df[config.delta_oi_col]
        if "_ok_delta_oi" in df.columns:
            df = df[df["_ok_delta_oi"]]

    df = df.dropna(subset=["return_clean", "delta_oi_clean"])
    df = df.sort_values(config.timestamp_col, kind="mergesort").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)
    return df


def save_validated_data(df: pd.DataFrame, config: ResearchConfig) -> Path:
    """Persist the validated dataset to parquet."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.validated_path, index=False)
    return config.validated_path


def load_validated_data(config: ResearchConfig) -> pd.DataFrame:
    """Load the previously saved validated dataset."""
    if not config.validated_path.exists():
        raise FileNotFoundError(
            f"Validated dataset not found: {config.validated_path}. "
            "Run data validation first."
        )
    return pd.read_parquet(config.validated_path)