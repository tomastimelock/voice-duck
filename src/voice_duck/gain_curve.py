# Filepath: src/voice_duck/gain_curve.py
# Condensed Description: Converts a raw envelope into a smoothed linear gain curve with attack/release/hold/lookahead.
# Architecture Layer: Pipeline
# Environment: Local
# Script Hierarchy: build_gain_curve
# Dependencies: Internal: voice_duck.config, voice_duck.utils / External: numpy
# Exposes: build_gain_curve
# Configuration: none
from __future__ import annotations

import logging

import numpy as np

from voice_duck.config import DuckConfig
from voice_duck.utils import db_to_linear

logger = logging.getLogger(__name__)


def build_gain_curve(
    envelope: np.ndarray,
    sample_rate: int,
    config: DuckConfig,
) -> np.ndarray:
    """Convert a linear amplitude envelope into a smoothed gain curve.

    Applies threshold gating, one-pole IIR attack/release smoothing, an
    optional hold counter, and optional lookahead time-shift to produce a
    per-frame linear gain multiplier ready to be applied to the target audio.

    The IIR loop is intentionally written as a plain Python loop so it can be
    JIT-compiled with Numba in v0.2 without restructuring.

    # Performance note: O(N) Python loop; Numba JIT is planned for v0.2.

    Args:
        envelope: Linear amplitude envelope, shape (frames,), float32 or float64.
        sample_rate: Sample rate in Hz.
        config: DuckConfig instance carrying all smoothing parameters.

    Returns:
        Float32 array of shape (frames,) containing linear gain values in the
        range [db_to_linear(config.reduction_db), 1.0].
    """
    threshold_linear = db_to_linear(config.threshold_db)

    # Step 1 — threshold gate: frames above threshold get reduction_db, others 0 dB.
    target_db: np.ndarray = np.where(
        envelope > threshold_linear,
        float(config.reduction_db),
        0.0,
    )

    # Step 2 — IIR smoothing with hold counter.
    attack_coeff = np.exp(-1.0 / (sample_rate * config.attack))
    release_coeff = np.exp(-1.0 / (sample_rate * config.release))
    hold_frames = int(config.hold * sample_rate)

    smoothed_db = np.zeros(len(target_db), dtype=np.float64)
    current: float = 0.0
    hold_counter: int = 0

    for i, tgt in enumerate(target_db):
        tgt = float(tgt)
        if tgt < current:
            # Ducking engaging — target is lower (more negative) than current.
            coeff = attack_coeff
            hold_counter = hold_frames
        else:
            if hold_counter > 0:
                hold_counter -= 1
                tgt = current  # Stay at current duck level during hold period.
            coeff = release_coeff
        current = coeff * current + (1.0 - coeff) * tgt
        smoothed_db[i] = current

    # Step 3 — lookahead: shift the gain curve earlier by the lookahead amount.
    lookahead_frames = int(config.lookahead * sample_rate)
    if lookahead_frames > 0:
        smoothed_db = np.concatenate([smoothed_db[lookahead_frames:], np.zeros(lookahead_frames)])

    # Step 4 — convert from dB to linear.
    gain_linear = 10.0 ** (smoothed_db / 20.0)

    logger.debug(
        "build_gain_curve: frames=%d, gain_min=%.4f, gain_max=%.4f",
        len(gain_linear),
        gain_linear.min(),
        gain_linear.max(),
    )

    return gain_linear.astype(np.float32)
