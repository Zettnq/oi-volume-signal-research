# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Central configuration for the OI / volume-delta reversal research."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TIMEFRAME_TO_TIMEDELTA = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "volume_delta",
    "open_interest",
    "return",
    "delta_oi",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "volume_delta",
    "open_interest",
    "return",
    "delta_oi",
]


@dataclass
class ResearchConfig:
    """Paths, columns and tolerances shared by the whole pipeline."""

    project_root: Path
    raw_csv_path: Path
    timeframe: str = "1h"
    asset: str = "btcusdt"

    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 10)
    primary_horizon: int = 3

    entry_mode: str = "next_open"
    timestamp_col: str = "timestamp"

    required_columns: list[str] | None = None
    numeric_columns: list[str] | None = None
    price_columns: list[str] | None = None

    volume_col: str = "volume"
    volume_delta_col: str = "volume_delta"
    open_interest_col: str = "open_interest"
    return_col: str = "return"
    delta_oi_col: str = "delta_oi"

    return_atol: float = 1e-5
    delta_oi_atol: float = 1e-4
    ohlc_atol: float = 1e-8

    min_rows: int = 100

    prefer_computed_return: bool = True
    prefer_computed_delta_oi: bool = True

    plot_theme: str = "light"  # "light" | "ice_nine"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.raw_csv_path = Path(self.raw_csv_path).expanduser().resolve()

        if self.required_columns is None:
            self.required_columns = list(REQUIRED_COLUMNS)
        if self.numeric_columns is None:
            self.numeric_columns = list(NUMERIC_COLUMNS)
        if self.price_columns is None:
            self.price_columns = ["open", "high", "low", "close"]

        if self.timeframe not in TIMEFRAME_TO_TIMEDELTA:
            raise ValueError(
                f"Unsupported timeframe: {self.timeframe}. "
                f"Supported: {sorted(TIMEFRAME_TO_TIMEDELTA.keys())}"
            )
        if self.entry_mode not in {"next_open"}:
            raise ValueError(
                f"Unsupported entry_mode: {self.entry_mode}. "
                "Currently only 'next_open' is supported."
            )
        if len(self.horizons) == 0:
            raise ValueError("horizons cannot be empty.")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons contain duplicates.")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("All horizons must be positive integers.")
        if self.primary_horizon not in self.horizons:
            raise ValueError("primary horizon must be included in horizons.")

        if self.plot_theme not in {"light", "ice_nine"}:
            raise ValueError(
                f"Unsupported plot theme: {self.plot_theme}. "
                "Supported: 'light', 'ice_nine'."
        )

    def display_path(self, p: Path | str) -> str:
        """Path relative to project root — safe for public outputs."""
        try:
            return Path(p).relative_to(self.project_root).as_posix()
        except ValueError:
            return Path(p).name


    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def expected_timedelta(self) -> pd.Timedelta:
        return TIMEFRAME_TO_TIMEDELTA[self.timeframe]

    # ------------------------------------------------------------------ #
    # Directory layout
    # ------------------------------------------------------------------ #
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def tables_dir(self) -> Path:
        return self.reports_dir / "tables" / self.timeframe

    # ------------------------------------------------------------------ #
    # Artifact paths (single source of truth)
    # ------------------------------------------------------------------ #
    @property
    def validated_path(self) -> Path:
        return self.processed_dir / f"{self.asset}_{self.timeframe}_validated.parquet"

    @property
    def signals_path(self) -> Path:
        return self.processed_dir / f"{self.asset}_{self.timeframe}_signals.parquet"

    @property
    def forward_returns_path(self) -> Path:
        return self.processed_dir / f"{self.asset}_{self.timeframe}_forward_returns.parquet"

    @property
    def events_raw_path(self) -> Path:
        return self.processed_dir / "events_raw.parquet"

    @property
    def events_episodes_path(self) -> Path:
        return self.processed_dir / "events_episodes.parquet"

    @property
    def events_confirmation_path(self) -> Path:
        return self.processed_dir / "events_confirmation.parquet"

    @property
    def raw_5m_path(self) -> Path:
        return self.data_dir / "raw" / f"{self.asset}_5m.csv"

    @property
    def data_5m_path(self) -> Path:
        return self.processed_dir / f"{self.asset}_5m.parquet"

    def figures_dir_for(self, spec: str) -> Path:
        """Figures folder for a pipeline stage: 'data', 's1'..'s5'."""
        return self.reports_dir / "figures" / spec


def get_config(
    project_root: Path | str,
    raw_csv_path: Path | str,
    timeframe: str = "1h",
    asset: str = "btcusdt",
) -> ResearchConfig:
    """Convenience factory for a standard config."""
    return ResearchConfig(
        project_root=Path(project_root),
        raw_csv_path=Path(raw_csv_path),
        timeframe=timeframe,
        asset=asset,
    )
