# Filepath: tests/test_threshold.py
# Condensed Description: Tests that below-threshold triggers produce no ducking and above-threshold triggers engage ducking.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_silence_trigger_no_ducking, test_just_below_threshold_no_ducking, test_threshold_boundary
# Dependencies: Internal: voice_duck.config, voice_duck.ducker / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np

from voice_duck.config import DuckConfig
from voice_duck.ducker import Ducker


def test_silence_trigger_no_ducking(silence: np.ndarray, sr: int) -> None:
    """All-silence trigger must produce a gain envelope of 1.0 (no ducking) everywhere."""
    ducker = Ducker(DuckConfig(threshold_db=-40.0, reduction_db=-20.0))
    gain = ducker.envelope(silence, sr)

    np.testing.assert_allclose(gain, 1.0, atol=0.01)


def test_just_below_threshold_no_ducking(sr: int) -> None:
    """A trigger at -50 dBFS (below -40 dB threshold) must produce no meaningful ducking."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # -50 dBFS sine — 10 dB below the threshold
    quiet = (10 ** (-50 / 20) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # Moderate-level target
    loud = np.ones(sr, dtype=np.float32) * 0.4

    ducker = Ducker(DuckConfig(threshold_db=-40.0, reduction_db=-20.0))
    result = ducker.apply(loud, quiet, sr)

    # Output must be nearly identical to input — no ducking applied.
    np.testing.assert_allclose(result.flatten(), loud.flatten(), atol=0.01)


def test_threshold_boundary(sr: int) -> None:
    """A trigger slightly above threshold must drop the gain envelope below 1.0."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # -35 dBFS sine — 5 dB above the -40 dB threshold
    above = (10 ** (-35 / 20) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    ducker = Ducker(DuckConfig(threshold_db=-40.0, reduction_db=-20.0, attack=0.01, release=0.4))
    gain = ducker.envelope(above, sr)

    # After settling there must be frames where the gain dropped below 1.0.
    assert float(np.min(gain)) < 1.0


def test_trigger_well_above_threshold_ducks(sr: int) -> None:
    """A loud trigger (0.5 amplitude) must duck the gain substantially below 1.0."""
    trigger = np.ones(sr, dtype=np.float32) * 0.5  # well above -40 dBFS

    ducker = Ducker(DuckConfig(threshold_db=-40.0, reduction_db=-12.0, attack=0.05, release=0.4))
    gain = ducker.envelope(trigger, sr)

    # After attack settling, gain must be noticeably reduced.
    settle_frame = int(0.5 * sr)
    assert float(gain[settle_frame]) < 0.9


def test_threshold_default_used_when_unspecified(sr: int) -> None:
    """DuckConfig default threshold of -40.0 dB must be respected."""
    cfg = DuckConfig()
    assert cfg.threshold_db == -40.0
