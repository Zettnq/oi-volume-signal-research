# Copyright (c) 2026 Zettnq
# SPDX-License-Identifier: MIT

"""Intrabar (5m) analysis of the exhaustion-reversal phenomenon.

This subpackage decomposes the 1h oracle (S1) reversal into 5m structure
(Test 0 / S5) and builds multi-timeframe trigger events with causal and
oracle variants plus controls (S4).
"""
from .data5m import load_5m, validate_5m, hour_paths
from .decomposition import extract_s1_events, collect_signed_paths, impulse_metrics
from .mt_events import (
    build_5m_arrays,
    build_s4_events,
    build_s4_causal_events,
    build_control_a,
    build_control_b,
    build_control_c,
)

__all__ = [
    "load_5m",
    "validate_5m",
    "hour_paths",
    "extract_s1_events",
    "collect_signed_paths",
    "impulse_metrics",
    "build_5m_arrays",
    "build_s4_events",
    "build_s4_causal_events",
    "build_control_a",
    "build_control_b",
    "build_control_c",
]