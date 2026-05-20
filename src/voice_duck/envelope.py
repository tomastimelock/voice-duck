# Filepath: src/voice_duck/envelope.py
# Condensed Description: Frame-by-frame RMS and peak envelope follower on the trigger signal.
# Architecture Layer: Utility
# Environment: Local
# Script Hierarchy: compute_envelope
# Dependencies: Internal: voice_duck.utils / External: numpy
# Exposes: compute_envelope
# Configuration: none
from __future__ import annotations

import logging

import numpy as np

from voice_duck.utils import to_mono

logger = logging.getLogger(__name__)


def compute_envelope(
    trigger: np.ndarray,
    sample_rate: int,
    detector: str = "rms",
    rms_window: float = 0.01,
) -> np.ndarray:
    """Compute a frame-by-frame amplitude envelope from the trigger signal.

    Mixes the trigger to mono first if it is multi-channel. For RMS detection
    a sliding window is computed via the cumsum-of-squares trick in O(N). For
    peak detection a sliding maximum is applied over the same window length.

    Args:
        trigger: Audio array of shape (frames,) or (frames, channels).
        sample_rate: Sample rate of the trigger signal in Hz.
        detector: ``"rms"`` for root-mean-square or ``"peak"`` for sliding max.
        rms_window: Averaging window length in seconds (used for both RMS and
            peak detectors as the sliding window length).

    Returns:
        Float32 array of shape (frames,) containing linear amplitude values.

    Raises:
        ValueError: If ``detector`` is not ``"rms"`` or ``"peak"``.
    """
    if detector not in ("rms", "peak"):
        raise ValueError(f"Unknown detector: '{detector}'. Must be 'rms' or 'peak'.")

    mono = to_mono(trigger)
    frames = len(mono)
    window = max(1, int(rms_window * sample_rate))

    logger.debug(
        "compute_envelope: detector=%s, frames=%d, window=%d, sample_rate=%d",
        detector,
        frames,
        window,
        sample_rate,
    )

    if detector == "rms":
        return _rms_envelope(mono, window)
    return _peak_envelope(mono, window)


def _rms_envelope(mono: np.ndarray, window: int) -> np.ndarray:
    """Compute a sliding RMS envelope using the cumsum-of-squares trick.

    Args:
        mono: 1-D float array of shape (frames,).
        window: Number of samples in the sliding window.

    Returns:
        Float32 array of shape (frames,) with RMS amplitude values.
    """
    sq = mono.astype(np.float64) ** 2
    cs = np.cumsum(sq)
    cs = np.concatenate([[0.0], cs])
    window_sums = cs[window:] - cs[:-window]  # shape: (frames - window + 1,)
    rms = np.sqrt(window_sums / window)
    # Pad the beginning with the first valid value so the output length == frames.
    pad_value = rms[0] if len(rms) > 0 else 0.0
    pad = np.full(window - 1, pad_value)
    return np.concatenate([pad, rms]).astype(np.float32)


def _peak_envelope(mono: np.ndarray, window: int) -> np.ndarray:
    """Compute a sliding peak (maximum absolute value) envelope.

    Uses a straightforward O(N * window) loop for correctness. For typical
    RMS window sizes of 10-100 ms at 48 kHz this is fast enough.

    Args:
        mono: 1-D float array of shape (frames,).
        window: Number of samples in the sliding window.

    Returns:
        Float32 array of shape (frames,) with peak amplitude values.
    """
    frames = len(mono)
    abs_mono = np.abs(mono).astype(np.float64)
    peak = np.empty(frames, dtype=np.float64)

    # Pad with zeros on the left so each position sees a full window
    padded = np.concatenate([np.zeros(window - 1), abs_mono])

    # Use stride-based approach: for each output sample i, max over padded[i:i+window]
    # Build a 2-D view using stride tricks for vectorized max
    from numpy.lib.stride_tricks import as_strided

    item_size = padded.strides[0]
    shape = (frames, window)
    strides = (item_size, item_size)
    windows_view = as_strided(padded, shape=shape, strides=strides)
    peak = windows_view.max(axis=1)

    return peak.astype(np.float32)
