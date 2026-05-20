# Filepath: tests/test_envelope.py
# Condensed Description: Tests for the RMS and peak envelope follower in voice_duck.envelope.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_rms_envelope_sine, test_peak_envelope_sine, test_envelope_shape_matches_frames
# Dependencies: Internal: voice_duck.envelope / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np
import pytest

from voice_duck.envelope import compute_envelope


def test_rms_envelope_sine(sine_1khz: np.ndarray, sr: int) -> None:
    """RMS envelope of a 1 kHz unity sine should settle to sqrt(0.5) within 0.5%."""
    env = compute_envelope(sine_1khz, sr, detector="rms", rms_window=0.01)

    assert env.shape == (len(sine_1khz),)
    # Skip the first 50 ms transient where the sliding window is still filling.
    steady = env[int(0.05 * sr) :]
    np.testing.assert_allclose(np.mean(steady), np.sqrt(0.5), rtol=0.005)


def test_peak_envelope_sine(sine_1khz: np.ndarray, sr: int) -> None:
    """Peak envelope of a 1 kHz unity sine should have a maximum of 1.0."""
    env = compute_envelope(sine_1khz, sr, detector="peak", rms_window=0.01)

    assert np.max(env) == pytest.approx(1.0, abs=0.01)


def test_envelope_shape_matches_frames_rms(sine_1khz: np.ndarray, sr: int) -> None:
    """RMS envelope output shape must exactly match the number of input frames."""
    env = compute_envelope(sine_1khz, sr, detector="rms", rms_window=0.01)

    assert env.shape == (len(sine_1khz),)
    assert env.dtype == np.float32


def test_envelope_shape_matches_frames_peak(sine_1khz: np.ndarray, sr: int) -> None:
    """Peak envelope output shape must exactly match the number of input frames."""
    env = compute_envelope(sine_1khz, sr, detector="peak", rms_window=0.01)

    assert env.shape == (len(sine_1khz),)
    assert env.dtype == np.float32


def test_envelope_unknown_detector_raises(sine_1khz: np.ndarray, sr: int) -> None:
    """An unknown detector name must raise ValueError with the bad name in the message."""
    with pytest.raises(ValueError, match="unknown_detector"):
        compute_envelope(sine_1khz, sr, detector="unknown_detector")


def test_rms_envelope_silence_is_zero(silence: np.ndarray, sr: int) -> None:
    """RMS envelope of silence must be all zeros."""
    env = compute_envelope(silence, sr, detector="rms", rms_window=0.01)

    np.testing.assert_allclose(env, 0.0, atol=1e-7)


def test_peak_envelope_silence_is_zero(silence: np.ndarray, sr: int) -> None:
    """Peak envelope of silence must be all zeros."""
    env = compute_envelope(silence, sr, detector="peak", rms_window=0.01)

    np.testing.assert_allclose(env, 0.0, atol=1e-7)


def test_rms_envelope_stereo_input(stereo_trigger: np.ndarray, sr: int) -> None:
    """RMS envelope accepts stereo input and returns a 1-D array."""
    env = compute_envelope(stereo_trigger, sr, detector="rms", rms_window=0.01)

    assert env.ndim == 1
    assert env.shape == (len(stereo_trigger),)
