# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Event construction: raw signals, episodes and confirmation events.

An "event" is a tradable unit with an entry price and forward returns. Three
families are supported:

- raw_signal:  every signal bar is an event;
- episode:     one event per cluster (feasible entry, no look-ahead);
- confirmation: episode + directed confirmation bar (mirrors the live logic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig

__all__ = [
    "build_events_raw",
    "build_events_episodes",
    "build_events_confirmation",
    "melt_events_long",
    "get_event_selection_summary",
    "select_nonoverlapping_events",
    "build_events_episodes_oracle",
    "_add_path_metrics",
]

_EVENT_ID_COLS = [
    "event_type",
    "signal_type",
    "signal",
    "trigger_bar_index",
    "trigger_timestamp",
    "entry_bar_index",
    "entry_time",
    "entry_price",
    "length",
]


def build_events_raw(df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """One event per signal bar, reusing precomputed fwd_ret_* / strategy_pnl_*."""
    if "is_signal" not in df.columns:
        raise ValueError("DataFrame must contain 'is_signal' column.")

    events = df.loc[df["is_signal"].astype(bool)].copy()
    if events.empty:
        return pd.DataFrame()

    events = events.rename(
        columns={
            "bar_index": "trigger_bar_index",
            config.timestamp_col: "trigger_timestamp",
        }
    )
    events["event_type"] = "raw_signal"
    events["length"] = 1

    keep_cols = list(_EVENT_ID_COLS)
    for h in config.horizons:
        keep_cols += [f"fwd_ret_{h}", f"strategy_pnl_{h}"]
    keep_cols = [c for c in keep_cols if c in events.columns]

    events = events[keep_cols].reset_index(drop=True)
    # Drop events with no entry (last bar of the dataset).
    events = events.dropna(subset=["entry_price"])
    return events


def build_events_episodes(df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """One event per episode (cluster), feasible entry without look-ahead.

    The episode ends at bar t_end. The gap bar (t_end+1) confirms the end only
    at its close, so entry occurs at open[t_end+2] and exit at close[t_end+1+h].
    """
    df_indexed = df.set_index("bar_index")
    all_events = []

    for sig_type, ep_col in [("long", "episode_id_long"), ("short", "episode_id_short")]:
        if ep_col not in df.columns:
            continue

        ep_data = (
            df[df[ep_col].notna()]
            .groupby(ep_col)
            .agg(
                trigger_bar_index=("bar_index", "max"),
                trigger_timestamp=(config.timestamp_col, "max"),
                length=("bar_index", "count"),
            )
            .reset_index()
        )
        if ep_data.empty:
            continue

        ep_data["signal_type"] = sig_type
        ep_data["signal"] = 1 if sig_type == "long" else -1
        ep_data["event_type"] = "episode"

        ep_data["gap_bar_index"] = ep_data["trigger_bar_index"] + 1
        ep_data["entry_bar_index"] = ep_data["trigger_bar_index"] + 2

        valid_mask = (
            (ep_data["gap_bar_index"] < len(df))
            & (ep_data["entry_bar_index"] < len(df))
        )
        ep_data = ep_data[valid_mask].copy()
        if ep_data.empty:
            continue

        entry_bars = ep_data["entry_bar_index"]
        ep_data["entry_time"] = df_indexed.loc[entry_bars, config.timestamp_col].to_numpy()
        ep_data["entry_price"] = df_indexed.loc[entry_bars, "open"].to_numpy()

        for h in config.horizons:
            exit_bars = ep_data["trigger_bar_index"] + 1 + h
            fwd_ret = pd.Series(np.nan, index=ep_data.index)
            valid_exit = exit_bars < len(df)

            if valid_exit.any():
                exit_prices = df_indexed.loc[exit_bars[valid_exit], "close"].to_numpy()
                entry_prices = ep_data.loc[valid_exit, "entry_price"].to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    fwd_ret[valid_exit] = exit_prices / entry_prices - 1.0

            ep_data[f"fwd_ret_{h}"] = fwd_ret
            ep_data[f"strategy_pnl_{h}"] = ep_data["signal"] * fwd_ret

            ep_data = _add_path_metrics(ep_data, df, config)
            all_events.append(ep_data)

        all_events.append(ep_data)

    if not all_events:
        return pd.DataFrame()

    events = pd.concat(all_events, ignore_index=True)
    return events.sort_values("trigger_timestamp").reset_index(drop=True)


def build_events_confirmation(
    df: pd.DataFrame,
    config: ResearchConfig,
    max_wait: int = 5,
) -> pd.DataFrame:
    """Episode + directed confirmation, mirroring the live strategy.

    After the episode ends, wait up to max_wait bars for the first directed
    confirmation bar c (opposite-direction return + volume delta) and enter at
    open[c+1]. MFE/MAE over the holding window are included to characterise
    path-dependent (trailing) exits.
    """
    n = len(df)
    ts = df[config.timestamp_col].to_numpy()
    ret_col = "return_clean" if "return_clean" in df.columns else config.return_col
    ret = df[ret_col].to_numpy()
    vd = df[config.volume_delta_col].to_numpy()
    op = df["open"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    cl = df["close"].to_numpy()

    rows = []
    for sig_type, ep_col in [("long", "episode_id_long"), ("short", "episode_id_short")]:
        if ep_col not in df.columns:
            continue

        ep_df = (
            df[df[ep_col].notna()]
            .groupby(ep_col)
            .agg(
                start_bar=("bar_index", "min"),
                end_bar=("bar_index", "max"),
                length=("bar_index", "count"),
            )
            .reset_index()
        )
        sign = 1 if sig_type == "long" else -1

        for row in ep_df.itertuples(index=False):
            t_end = row.end_bar

            conf_bar = None
            for c in range(t_end + 1, min(t_end + 1 + max_wait, n)):
                if sig_type == "short":
                    if ret[c] < 0 and vd[c] < 0:
                        conf_bar = c
                        break
                else:
                    if ret[c] > 0 and vd[c] > 0:
                        conf_bar = c
                        break
            if conf_bar is None:
                continue

            entry_bar = conf_bar + 1
            if entry_bar >= n:
                continue
            entry_price = op[entry_bar]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            rec = {
                "signal_type": sig_type,
                "signal": sign,
                "event_type": "confirmation",
                "trigger_bar_index": t_end,
                "episode_length": row.length,
                "conf_bar_index": conf_bar,
                "wait": conf_bar - t_end,
                "entry_bar_index": entry_bar,
                "entry_time": ts[entry_bar],
                "entry_price": entry_price,
            }

            for h in config.horizons:
                exit_bar = entry_bar - 1 + h  # bars entry_bar .. entry_bar+h-1
                if exit_bar >= n:
                    rec[f"fwd_ret_{h}"] = np.nan
                    rec[f"strategy_pnl_{h}"] = np.nan
                    rec[f"mfe_{h}"] = np.nan
                    rec[f"mae_{h}"] = np.nan
                    continue

                price_ret = cl[exit_bar] / entry_price - 1.0
                window_hi = hi[entry_bar:exit_bar + 1].max()
                window_lo = lo[entry_bar:exit_bar + 1].min()

                if sign == 1:
                    mfe = window_hi / entry_price - 1.0
                    mae = window_lo / entry_price - 1.0
                else:
                    mfe = 1.0 - window_lo / entry_price
                    mae = 1.0 - window_hi / entry_price

                rec[f"fwd_ret_{h}"] = price_ret
                rec[f"strategy_pnl_{h}"] = sign * price_ret
                rec[f"mfe_{h}"] = mfe
                rec[f"mae_{h}"] = mae

            rows.append(rec)

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values("entry_time").reset_index(drop=True)


def melt_events_long(events_wide: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Wide → long: one row per (event, horizon).

    Produces universal columns price_return (= fwd_ret_h) and strategy_pnl
    (= strategy_pnl_h) plus a horizon column.
    """
    if events_wide.empty:
        return pd.DataFrame()

    id_vars = [c for c in _EVENT_ID_COLS if c in events_wide.columns]

    frames = []
    for h in config.horizons:
        has_ret = f"fwd_ret_{h}" in events_wide.columns
        has_pnl = f"strategy_pnl_{h}" in events_wide.columns
        if not (has_ret or has_pnl):
            continue

        sub = events_wide[id_vars].copy()
        if has_ret:
            sub["price_return"] = events_wide[f"fwd_ret_{h}"].to_numpy()
        if has_pnl:
            sub["strategy_pnl"] = events_wide[f"strategy_pnl_{h}"].to_numpy()
        sub["horizon"] = h
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_event_selection_summary(
    events_raw: pd.DataFrame,
    events_episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Event counts per methodology and signal type."""
    summary = []
    for name, events in [("raw_signals", events_raw), ("episodes", events_episodes)]:
        if events.empty:
            continue
        for sig_type in ["long", "short"]:
            summary.append(
                {
                    "method": name,
                    "signal_type": sig_type,
                    "n_events": int((events["signal_type"] == sig_type).sum()),
                }
            )
    return pd.DataFrame(summary)


def select_nonoverlapping_events(
    events_wide: pd.DataFrame,
    horizon: int,
    bar_col: str = "trigger_bar_index",
    gap: int | None = None,
) -> pd.DataFrame:
    """Chronological non-overlap filter: next.trigger >= last.selected + gap.

    Defaults to gap = horizon so holding windows do not intersect.
    """
    if events_wide.empty:
        return events_wide
    if bar_col not in events_wide.columns:
        raise ValueError(f"events_wide must contain '{bar_col}' column.")

    if gap is None:
        gap = horizon

    events_sorted = events_wide.sort_values(bar_col).reset_index(drop=True)
    selected = np.zeros(len(events_sorted), dtype=bool)
    last_selected = -np.inf

    for i, bar in enumerate(events_sorted[bar_col].to_numpy()):
        if bar >= last_selected + gap:
            selected[i] = True
            last_selected = bar

    return events_sorted.loc[selected].reset_index(drop=True)


def _add_path_metrics(
    events: pd.DataFrame,
    df: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Attach MFE/MAE over the holding window (bars entry..entry+h-1).

    Assumes bar_index is a contiguous 0..n-1 range (guaranteed by the
    validation stage), so positional indexing equals bar_index.
    """
    n = len(df)
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    events = events.copy()

    eb = events["entry_bar_index"].to_numpy()
    ep = events["entry_price"].to_numpy()
    sg = events["signal"].to_numpy()

    for h in config.horizons:
        mfe = np.full(len(events), np.nan)
        mae = np.full(len(events), np.nan)
        for i in range(len(events)):
            e = int(eb[i])
            px = ep[i]
            s = sg[i]
            xb = e + h - 1
            if xb >= n or not np.isfinite(px) or px <= 0:
                continue
            wh = hi[e:xb + 1].max()
            wl = lo[e:xb + 1].min()
            mfe[i] = (wh / px - 1.0) if s == 1 else (1.0 - wl / px)
            mae[i] = (wl / px - 1.0) if s == 1 else (1.0 - wh / px)
        events[f"mfe_{h}"] = mfe
        events[f"mae_{h}"] = mae
    return events


def build_events_episodes_oracle(
    df: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """S1 oracle: one event per episode with LOOK-AHEAD entry at the gap bar.

    Entry at open[t_end+1] (the gap bar itself), exit at close[t_end+h].
    Non-causal by construction: the gap bar is identifiable only ex-post.
    Serves as the diagnostic upper bound of the effect (notebook 05).
    """
    df_indexed = df.set_index("bar_index")
    n = len(df)
    all_events = []

    for sig_type, ep_col in [("long", "episode_id_long"), ("short", "episode_id_short")]:
        if ep_col not in df.columns:
            continue

        ep_data = (
            df[df[ep_col].notna()]
            .groupby(ep_col)
            .agg(
                trigger_bar_index=("bar_index", "max"),
                trigger_timestamp=(config.timestamp_col, "max"),
                length=("bar_index", "count"),
            )
            .reset_index()
        )
        if ep_data.empty:
            continue

        ep_data["signal_type"] = sig_type
        ep_data["signal"] = 1 if sig_type == "long" else -1
        ep_data["event_type"] = "episode_oracle"

        ep_data["gap_bar_index"] = ep_data["trigger_bar_index"] + 1
        ep_data["entry_bar_index"] = ep_data["trigger_bar_index"] + 1

        ep_data = ep_data[ep_data["entry_bar_index"] < n].copy()
        if ep_data.empty:
            continue

        entry_bars = ep_data["entry_bar_index"].to_numpy()
        ep_data["entry_time"] = df_indexed.loc[entry_bars, config.timestamp_col].to_numpy()
        ep_data["entry_price"] = df["open"].to_numpy()[entry_bars]

        for h in config.horizons:
            exit_bars = ep_data["trigger_bar_index"] + h
            fwd_ret = pd.Series(np.nan, index=ep_data.index)
            valid_exit = exit_bars < n
            if valid_exit.any():
                exit_prices = df["close"].to_numpy()[exit_bars.to_numpy()[valid_exit.to_numpy()]]
                entry_prices = ep_data.loc[valid_exit, "entry_price"].to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    fwd_ret[valid_exit] = exit_prices / entry_prices - 1.0
            ep_data[f"fwd_ret_{h}"] = fwd_ret
            ep_data[f"strategy_pnl_{h}"] = ep_data["signal"] * fwd_ret

        ep_data = _add_path_metrics(ep_data, df, config)
        all_events.append(ep_data)

    if not all_events:
        return pd.DataFrame()

    events = pd.concat(all_events, ignore_index=True)
    return events.sort_values("trigger_timestamp").reset_index(drop=True)