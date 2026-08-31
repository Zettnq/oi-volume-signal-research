# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Raw-signal intrabar decomposition (S5).

Decomposes the 5m path of the signal bar itself and of the following bar to
locate where the reversal lives relative to a raw signal:

- signal bar: by construction moves AGAINST the expected reversal (0% reversed);
- next bar:   the first bar where the reversal may begin;
- (gap bar of terminal signals coincides with the next bar for isolated episodes).

Impulse metrics are reused from intrabar.decomposition to avoid duplication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ResearchConfig
from ..intrabar.decomposition import impulse_metrics

__all__ = ["extract_raw_signals", "collect_signal_paths", "impulse_metrics"]

_SIGNAL_TYPES = [
    ("long", "long_signal", "episode_id_long"),
    ("short", "short_signal", "episode_id_short"),
]


def extract_raw_signals(
    df1h: pd.DataFrame,
    config: ResearchConfig | None = None,
) -> pd.DataFrame:
    """Extract every raw signal bar (long and short separately).

    Uses boolean signal columns when present, otherwise episode membership.
    """
    ts_col = config.timestamp_col if config is not None else "timestamp"
    ts_by_bar = df1h.set_index("bar_index")[ts_col]

    rows = []
    for sig, sig_col, ep_col in _SIGNAL_TYPES:
        if sig_col in df1h.columns:
            bars = df1h.loc[df1h[sig_col].astype(bool), "bar_index"]
        elif ep_col in df1h.columns:
            bars = df1h.loc[df1h[ep_col].notna(), "bar_index"]
        else:
            continue

        sign = 1 if sig == "long" else -1
        for bar in bars:
            if bar not in ts_by_bar.index:
                continue
            rows.append(
                {
                    "signal_type": sig,
                    "sign": sign,
                    "signal_bar": bar,
                    "signal_ts": ts_by_bar.loc[bar],
                }
            )
    return pd.DataFrame(rows)


def collect_signal_paths(
    signals: pd.DataFrame,
    hour_paths: dict,
    which: str = "signal",
) -> np.ndarray:
    """Collect the signed 5m path of the signal bar or the next bar.

    which="signal": path of the signal bar itself;
    which="next":   path of the following bar (t+1).
    Positive values = movement in the expected reversal direction.
    """
    if which not in {"signal", "next"}:
        raise ValueError(f"Unknown which: {which}. Supported: 'signal', 'next'.")

    paths = []
    for r in signals.itertuples(index=False):
        ts = r.signal_ts
        if which == "next":
            ts = ts + pd.Timedelta(hours=1)
        p = hour_paths.get(ts)
        if p is None:
            continue
        paths.append(p * r.sign)
    return np.array(paths)