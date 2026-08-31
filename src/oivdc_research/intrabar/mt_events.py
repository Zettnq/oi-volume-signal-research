# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Multi-timeframe (5m) trigger events: S4 oracle/causal variants + controls.

Causality note: S4-A (and controls A/B) condition on the gap hour, which is
only known ex-post — they are ORACLE diagnostics. S4-B is the fully causal,
live-compatible variant that monitors after every 1h signal with reset.
The difference S4-A - S4-B quantifies hindsight selection bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ResearchConfig

__all__ = [
    "build_5m_arrays",
    "build_s4_events",
    "build_s4_causal_events",
    "build_control_a",
    "build_control_b",
    "build_control_c",
]


def _resolve_horizons(config: ResearchConfig | None, horizons) -> tuple:
    if horizons is not None:
        return tuple(horizons)
    return config.horizons if config is not None else (1, 2, 3, 6, 12)


def build_5m_arrays(df5m: pd.DataFrame) -> dict:
    """Flatten 5m data into numpy arrays + hour -> first-index map."""
    df = df5m.sort_values("timestamp").reset_index(drop=True)

    hour = df["timestamp"].dt.floor("h")
    first = df.groupby(hour, sort=False).head(1)
    hour_start = dict(zip(first["timestamp"].dt.floor("h"), first.index))

    return {
        "ts": df["timestamp"].to_numpy(),
        "open": df["open"].to_numpy(),
        "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(),
        "close": df["close"].to_numpy(),
        "ret5": df["close"].pct_change().fillna(0.0).to_numpy(),
        "vd5": df["volume_delta"].fillna(0.0).to_numpy(),
        "hour_start": hour_start,
    }


def _measure_from(a: dict, entry_idx: int, sign: int, horizons) -> dict | None:
    """Forward PnL and MFE/MAE for an entry at open[entry_idx]."""
    n = len(a["close"])
    entry = a["open"][entry_idx]
    if entry <= 0 or not np.isfinite(entry):
        return None

    rec = {"entry_idx": entry_idx}
    for h in horizons:
        xb = entry_idx + h - 1
        if xb >= n:
            rec[f"pnl_{h}"] = np.nan
            rec[f"mfe_{h}"] = np.nan
            rec[f"mae_{h}"] = np.nan
            continue
        pr = a["close"][xb] / entry - 1.0
        wh = a["high"][entry_idx:xb + 1].max()
        wl = a["low"][entry_idx:xb + 1].min()
        rec[f"pnl_{h}"] = sign * pr
        rec[f"mfe_{h}"] = (wh / entry - 1) if sign == 1 else (1 - wl / entry)
        rec[f"mae_{h}"] = (wl / entry - 1) if sign == 1 else (1 - wh / entry)
    return rec


def _scan_directed(a, start, sign, horizons, window, wrong_side=False):
    """Find first directed 5m bar in [start, start+window); enter at next open.

    wrong_side=False: trigger agrees with the expected reversal (right side).
    wrong_side=True:  trigger opposes it (Control B/C).
    PnL is always measured against the ORIGINAL sign.
    """
    n = len(a["close"])
    for c in range(start, min(start + window, n - 1)):
        r, v = a["ret5"][c], a["vd5"][c]
        right = (r > 0 and v > 0) if sign == 1 else (r < 0 and v < 0)
        directed = (not right) if wrong_side else right
        if not directed:
            continue
        return _measure_from(a, c + 1, sign, horizons)
    return None


def build_s4_events(a, s1_events, config=None, horizons=None):
    """S4-A (oracle): right-side 5m trigger inside the gap hour."""
    hs = _resolve_horizons(config, horizons)
    rows = []
    for ev in s1_events.itertuples(index=False):
        start = a["hour_start"].get(ev.gap_ts)
        if start is None:
            continue
        rec = _scan_directed(a, start, ev.sign, hs, window=12)
        if rec is None:
            continue
        rec["signal_type"] = ev.signal_type
        rec["source"] = "s4_a"
        rows.append(rec)
    return pd.DataFrame(rows)


def build_control_a(a, s1_events, config=None, horizons=None, rng_seed=42):
    """Control A: right-side 5m trigger in RANDOM hours (no 1h state)."""
    hs = _resolve_horizons(config, horizons)
    rng = np.random.default_rng(rng_seed)
    hours = list(a["hour_start"].values())
    rows = []
    for ev in s1_events.itertuples(index=False):
        rec = _scan_directed(a, int(rng.choice(hours)), ev.sign, hs, window=12)
        if rec is None:
            continue
        rec["signal_type"] = ev.signal_type
        rec["source"] = "ctrl_a"
        rows.append(rec)
    return pd.DataFrame(rows)


def build_control_b(a, s1_events, config=None, horizons=None):
    """Control B: WRONG-side 5m trigger inside the gap hour."""
    hs = _resolve_horizons(config, horizons)
    rows = []
    for ev in s1_events.itertuples(index=False):
        start = a["hour_start"].get(ev.gap_ts)
        if start is None:
            continue
        rec = _scan_directed(a, start, ev.sign, hs, window=12, wrong_side=True)
        if rec is None:
            continue
        rec["signal_type"] = ev.signal_type
        rec["source"] = "ctrl_b"
        rows.append(rec)
    return pd.DataFrame(rows)


def build_control_c(a, s1_events, config=None, horizons=None, rng_seed=42):
    """Control C: WRONG-side 5m trigger in RANDOM hours (baseline reversion)."""
    hs = _resolve_horizons(config, horizons)
    rng = np.random.default_rng(rng_seed)
    hours = list(a["hour_start"].values())
    rows = []
    for ev in s1_events.itertuples(index=False):
        rec = _scan_directed(a, int(rng.choice(hours)), ev.sign, hs,
                             window=12, wrong_side=True)
        if rec is None:
            continue
        rec["signal_type"] = ev.signal_type
        rec["source"] = "ctrl_c"
        rows.append(rec)
    return pd.DataFrame(rows)


def build_s4_causal_events(a, df1h, config=None, max_window_hours=5, horizons=None):
    """S4-B: fully causal, live-compatible variant.

    Chronological pass: after the close of each 1h signal bar, start monitoring
    the following hours (up to max_window_hours) for the first directed 5m
    confirmation; a new 1h signal resets the window. All decisions use only
    information available at decision time, so continuation-hour entries
    (losers) are included — this is the honest causal estimate.
    """
    hs = _resolve_horizons(config, horizons)
    n = len(a["close"])
    hour_start = a["hour_start"]
    hours = sorted(hour_start.keys())

    ts_col = config.timestamp_col if config is not None else "timestamp"
    ts_h = pd.to_datetime(df1h[ts_col], utc=True).dt.floor("h")
    sig_by_hour = {}
    for t, row in zip(ts_h, df1h.itertuples(index=False)):
        if "episode_id_long" in df1h.columns and pd.notna(getattr(row, "episode_id_long", None)):
            sig_by_hour[t] = "long"
        elif "episode_id_short" in df1h.columns and pd.notna(getattr(row, "episode_id_short", None)):
            sig_by_hour[t] = "short"

    rows = []
    direction = 0
    window_end = -1
    entered = True  # inactive

    for h in hours:
        start = hour_start[h]

        # 1) scan this hour's 5m bars if monitoring is active
        if (not entered) and direction != 0 and start < window_end:
            last = min(start + 12, window_end, n - 1)
            for c in range(start, last):
                r, v = a["ret5"][c], a["vd5"][c]
                directed = (r > 0 and v > 0) if direction == 1 else (r < 0 and v < 0)
                if not directed:
                    continue
                rec = _measure_from(a, c + 1, direction, hs)
                if rec is not None:
                    rec["signal_type"] = "long" if direction == 1 else "short"
                    rec["source"] = "s4_b"
                    rows.append(rec)
                entered = True
                break

        # 2) after the hour closes, evaluate the 1h signal and reset
        st = sig_by_hour.get(h)
        if st is not None:
            direction = 1 if st == "long" else -1
            nxt = hour_start.get(h + pd.Timedelta(hours=1))
            if nxt is None:
                entered = True
            else:
                window_end = min(nxt + max_window_hours * 12, n)
                entered = False

    return pd.DataFrame(rows)