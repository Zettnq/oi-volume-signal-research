# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Causal (no look-ahead) entry analysis: episode starts and raw signals.

This subpackage tests whether the exhaustion edge is extractable without
hindsight: causal entry at episode starts (S5) and intrabar decomposition of
raw signals relative to the signal bar.
"""
from .episode_start import build_causal_events, episode_length_map, attach_episode_length
from .intrabar_signal import extract_raw_signals, collect_signal_paths

__all__ = [
    "build_causal_events",
    "episode_length_map",
    "attach_episode_length",
    "extract_raw_signals",
    "collect_signal_paths",
]