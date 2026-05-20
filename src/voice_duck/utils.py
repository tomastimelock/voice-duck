# Filepath: src/voice_duck/utils.py
# Condensed Description: dB/linear conversion helpers and channel-matching validation for sidechain ducking.
# Architecture Layer: Utility
# Environment: Local
# Script Hierarchy: db_to_linear, linear_to_db, to_mono, match_channels
# Dependencies: Internal: none / External: numpy
# Exposes: db_to_linear, linear_to_db, to_mono, match_channels
# Configuration: none
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def db_to_linear(db: float) -> float:
    """Convert a dB value to a linear amplitude multiplier.

    Args:
        db: Level in decibels.

    Returns:
        Linear amplitude equivalent to the given dB value.
    """
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """Convert a linear amplitude value to decibels.

    Clamps the input to a minimum of 1e-9 to avoid log(0).

    Args:
        linear: Linear amplitude value (must be >= 0).

    Returns:
        Level in decibels, with a floor at approximately -180 dB.
    """
    return 20 * math.log10(max(linear, 1e-9))


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Mix a signal down to mono, returning float32.

    If the signal is 2-D with shape (frames, channels), return the mean across
    channels. If it is already 1-D (frames,), return it unchanged as float32.

    Args:
        signal: Audio array of shape (frames,) or (frames, channels).

    Returns:
        Mono float32 array of shape (frames,).
    """
    if signal.ndim == 2:
        return signal.mean(axis=1).astype(np.float32)
    return signal.astype(np.float32)


def match_channels(
    target: np.ndarray,
    trigger: np.ndarray,
    sample_rate_target: int,
    sample_rate_trigger: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate sample-rate parity and return (target, trigger) unchanged.

    This function does not reshape any arrays — it only validates that the
    sample rates match, so the caller can safely proceed to envelope detection.
    Channel layout mismatches (e.g. stereo target + mono trigger) are handled
    downstream by ``to_mono`` inside the envelope follower.

    Args:
        target: Target audio array, shape (frames,) or (frames, 2).
        trigger: Trigger audio array, shape (frames,) or (frames, 2).
        sample_rate_target: Sample rate of the target signal in Hz.
        sample_rate_trigger: Sample rate of the trigger signal in Hz.

    Returns:
        The pair (target, trigger) with no modifications.

    Raises:
        ValueError: If the two sample rates differ, with a message that
            includes both values and a hint to resample before calling duck().
    """
    if sample_rate_target != sample_rate_trigger:
        raise ValueError(
            f"Sample rate mismatch: target={sample_rate_target} Hz, "
            f"trigger={sample_rate_trigger} Hz. "
            "Resample before calling duck()."
        )
    logger.debug(
        "match_channels: sample_rate=%d, target.shape=%s, trigger.shape=%s",
        sample_rate_target,
        target.shape,
        trigger.shape,
    )
    return target, trigger
