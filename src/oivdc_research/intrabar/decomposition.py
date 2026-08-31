# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""S1 oracle decomposition: where does the reversal live inside the gap hour?

Test 0 / S5 building blocks: extract S1 events (episode + gap bar), collect the
signed 5m path of the gap hour and summarise its impulse structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ResearchConfig

__all__ = ["extract_s1_events", "collect_signed_paths", "impulse_metrics"]


def extract_s1_events(df1h: pd.DataFrame, config: ResearchConfig | None = None) -> pd.DataFrame:
    """Extract S1 (oracle) events: one row per episode with its gap bar.

    gap bar = first bar after the episode end (t_end + 1). Known only ex-post,
    hence "oracle". sign = +1 for long episodes (expect up), -1 for short.
    """
    ts_col = config.timestamp_col if config is not None else "timestamp"
    ts_by_bar = df1h.set_index("bar_index")[ts_col]

    rows = []
    for sig, col in [("long", "episode_id_long"), ("short", "episode_id_short")]:
        if col not in df1h.columns:
            continue
        sub = df1h[df1h[col].notna()]
        ends = sub.groupby(col)["bar_index"].max()
        for end in ends.values:
            gap = end + 1
            if gap not in ts_by_bar.index:
                continue
            rows.append(
                {
                    "signal_type": sig,
                    "sign": 1 if sig == "long" else -1,
                    "gap_ts": ts_by_bar.loc[gap],
                }
            )
    return pd.DataFrame(rows)


def collect_signed_paths(events: pd.DataFrame, hour_paths: dict) -> np.ndarray:
    """Collect the signed 5m path of each event's gap hour.

    Positive values = movement in the expected reversal direction.
    """
    paths = []
    for r in events.itertuples(index=False):
        p = hour_paths.get(r.gap_ts)
        if p is None:
            continue
        paths.append(p * r.sign)
    return np.array(paths)


def impulse_metrics(paths: np.ndarray) -> dict:
    """Summarise the impulse structure of signed gap-hour paths.

    - idx50/idx80: first 5m bar (0..11) where 50%/80% of the total reversal
      is reached (median over reversed events);
    - single_jump_frac: share of reversed events where a SINGLE 5m increment
      accounts for >=80% of the total reversal (computed on increments, not on
      the cumulative path);
    - max_5m_contribution: largest single 5m increment as a fraction of total.
    """
    if paths.size == 0:
        return {k: np.nan for k in (
            "n_events", "n_reversed", "share_reversed", "idx50_median",
            "idx80_median", "single_jump_frac", "mean_total_reversal",
            "median_total_reversal", "mean_max_5m_contribution",
            "median_max_5m_contribution",
        )} | {"n_events": 0, "n_reversed": 0}

    total = paths[:, -1]
    pos_mask = total > 0
    pos = paths[pos_mask]
    tot = total[pos_mask]

    if len(pos) == 0:
        return {
            "n_events": int(len(paths)),
            "n_reversed": 0,
            "share_reversed": 0.0,
            "idx50_median": np.nan,
            "idx80_median": np.nan,
            "single_jump_frac": np.nan,
            "mean_total_reversal": np.nan,
            "median_total_reversal": np.nan,
            "mean_max_5m_contribution": np.nan,
            "median_max_5m_contribution": np.nan,
        }

    frac = pos / tot[:, None]
    idx50 = np.argmax(frac >= 0.5, axis=1)
    idx80 = np.argmax(frac >= 0.8, axis=1)

    # Single 5m increments from the cumulative path.
    increments = np.diff(
        np.concatenate([np.zeros((pos.shape[0], 1)), pos], axis=1),
        axis=1,
    )
    max_5m_inc = increments.max(axis=1)
    max_5m_contribution = max_5m_inc / tot
    single_jump_frac = np.mean(max_5m_contribution >= 0.8)

    return {
        "n_events": int(len(paths)),
        "n_reversed": int(pos_mask.sum()),
        "share_reversed": float(pos_mask.mean()),
        "idx50_median": float(np.nanmedian(idx50)),
        "idx80_median": float(np.nanmedian(idx80)),
        "single_jump_frac": float(single_jump_frac),
        "mean_total_reversal": float(np.mean(tot)),
        "median_total_reversal": float(np.median(tot)),
        "mean_max_5m_contribution": float(np.mean(max_5m_contribution)),
        "median_max_5m_contribution": float(np.median(max_5m_contribution)),
    }