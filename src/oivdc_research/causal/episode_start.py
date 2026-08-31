# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Causal episode-start events (S5).

An episode start is a signal bar whose previous bar carries no signal of the
same type; it is identifiable at the close of the bar (no look-ahead). Entering
at open[t+1] on every episode start yields a causal strategy whose aggregate
edge is a mixture of:

- isolated (terminal) starts: the episode has length 1 -> the next bar is the
  gap bar and carries the reversal (positive edge);
- multi-bar (continuation) starts: the trend continues -> fading it is adverse.

Because termination is only known ex-post, the causal aggregate nets to ~0;
the isolated/multi split (attach_episode_length) quantifies the asymmetry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ResearchConfig

__all__ = ["build_causal_events", "episode_length_map", "attach_episode_length"]

_SIGNAL_TYPES = [("long", "episode_id_long"), ("short", "episode_id_short")]


def _resolve_horizons(config: ResearchConfig | None, horizons) -> tuple:
    if horizons is not None:
        return tuple(horizons)
    return config.horizons if config is not None else (1, 2, 3, 5, 10)


def build_causal_events(
    df1h: pd.DataFrame,
    config: ResearchConfig | None = None,
    horizons=None,
    mode: str = "starts",
) -> pd.DataFrame:
    """Build causal 1h events with entry at open[t+1].

    mode="starts": only episode starts (signal_t AND NOT signal_{t-1}).
    mode="raw":    every signal bar.

    Forward PnL and MFE/MAE are computed on 1h high/low over bars
    entry..entry+h-1.
    """
    if mode not in {"starts", "raw"}:
        raise ValueError(f"Unknown mode: {mode}. Supported: 'starts', 'raw'.")

    hs = _resolve_horizons(config, horizons)
    n = len(df1h)
    op = df1h["open"].to_numpy()
    hi = df1h["high"].to_numpy()
    lo = df1h["low"].to_numpy()
    cl = df1h["close"].to_numpy()

    rows = []
    for sig, ep_col in _SIGNAL_TYPES:
        if ep_col not in df1h.columns:
            continue
        mask = df1h[ep_col].notna().to_numpy()
        sign = 1 if sig == "long" else -1

        for i in range(n):
            if not mask[i]:
                continue
            if mode == "starts" and i > 0 and mask[i - 1]:
                continue  # continuation bar, not a start

            entry = i + 1
            if entry >= n:
                continue
            entry_price = op[entry]
            if entry_price <= 0 or not np.isfinite(entry_price):
                continue

            rec = {
                "signal_type": sig,
                "sign": sign,
                "start_bar": i,
                "entry_bar": entry,
            }
            for h in hs:
                xb = entry - 1 + h  # bars entry..entry+h-1
                if xb >= n:
                    rec[f"pnl_{h}"] = np.nan
                    rec[f"mfe_{h}"] = np.nan
                    rec[f"mae_{h}"] = np.nan
                    continue
                pr = cl[xb] / entry_price - 1.0
                wh = hi[entry:xb + 1].max()
                wl = lo[entry:xb + 1].min()
                rec[f"pnl_{h}"] = sign * pr
                rec[f"mfe_{h}"] = (wh / entry_price - 1) if sign == 1 else (1 - wl / entry_price)
                rec[f"mae_{h}"] = (wl / entry_price - 1) if sign == 1 else (1 - wh / entry_price)
            rows.append(rec)

    return pd.DataFrame(rows)


def episode_length_map(df1h: pd.DataFrame) -> dict:
    """Map (signal_type, episode_id) -> episode length (in bars)."""
    lengths = {}
    for sig, ep_col in _SIGNAL_TYPES:
        if ep_col not in df1h.columns:
            continue
        sub = df1h[df1h[ep_col].notna()]
        lens = sub.groupby(ep_col)["bar_index"].count()
        for ep, length in lens.items():
            lengths[(sig, ep)] = int(length)
    return lengths


def attach_episode_length(events: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
    """Add episode_length / isolated columns to episode-start events.

    isolated = (episode_length == 1), i.e. the start is also the terminal bar.
    Requires events to contain start_bar and signal_type.
    """
    events = events.copy()
    lens = episode_length_map(df1h)

    bar_to_ep = {}
    for sig, ep_col in _SIGNAL_TYPES:
        if ep_col not in df1h.columns:
            continue
        sub = df1h[df1h[ep_col].notna()]
        bar_to_ep[sig] = sub.set_index("bar_index")[ep_col]

    def _len(row) -> float:
        mapping = bar_to_ep.get(row.signal_type)
        if mapping is None or row.start_bar not in mapping.index:
            return np.nan
        ep = mapping.loc[row.start_bar]
        return lens.get((row.signal_type, ep), np.nan)

    events["episode_length"] = events.apply(_len, axis=1)
    events["isolated"] = events["episode_length"] == 1
    return events