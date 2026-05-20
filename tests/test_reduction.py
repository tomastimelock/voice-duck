# Filepath: tests/test_reduction.py
# Condensed Description: Tests that above-threshold input drives the gain curve to reduction_db after settling.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_full_duck_level_after_settling, test_custom_reduction_db, test_output_amplitude_matches_reduction
# Dependencies: Internal: voice_duck.config, voice_duck.ducker, voice_duck.utils / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np
import pytest

from voice_duck.config import DuckConfig
from voice_duck.ducker import Ducker
from voice_duck.utils import db_to_linear, linear_to_db


def test_full_duck_level_after_settling(sr: int) -> None:
    """After 5x attack time, the gain must have settled within 0.5 dB of reduction_db."""
    attack = 0.05
    reduction_db = -12.0
    cfg = DuckConfig(
        reduction_db=reduction_db,
        attack=attack,
        release=0.4,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)
    trigger = np.ones(sr, dtype=np.float32) * 0.5  # loud trigger — well above threshold

    gain = ducker.envelope(trigger, sr)

    settle_frame = int(5 * attack * sr)
    settled_gain_db = linear_to_db(float(gain[settle_frame]))
    assert abs(settled_gain_db - reduction_db) < 0.5


def test_custom_reduction_db_minus20(sr: int) -> None:
    """With reduction_db=-20, settled gain must be within 0.5 dB of -20 dBFS."""
    reduction_db = -20.0
    cfg = DuckConfig(
        reduction_db=reduction_db,
        attack=0.01,
        release=0.4,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)
    # Long trigger so IIR has time to settle
    trigger = np.ones(2 * sr, dtype=np.float32) * 0.5

    gain = ducker.envelope(trigger, sr)

    settled_gain_db = linear_to_db(float(gain[-1]))
    assert abs(settled_gain_db - reduction_db) < 0.5


def test_output_amplitude_matches_reduction(sr: int) -> None:
    """After settling, output RMS must approximately equal input RMS x db_to_linear(reduction_db)."""
    reduction_db = -12.0
    attack = 0.01
    cfg = DuckConfig(
        reduction_db=reduction_db,
        attack=attack,
        release=0.4,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)

    # DC target so we can measure level precisely
    target = np.ones(2 * sr, dtype=np.float32) * 0.5
    trigger = np.ones(2 * sr, dtype=np.float32) * 0.5

    result = ducker.apply(target, trigger, sr)

    # Use the settled second half
    settled_output = result[sr:]
    settled_target = target[sr:]

    input_rms = float(np.sqrt(np.mean(settled_target**2)))
    output_rms = float(np.sqrt(np.mean(settled_output**2)))
    expected_rms = input_rms * db_to_linear(reduction_db)

    np.testing.assert_allclose(output_rms, expected_rms, rtol=0.05)


def test_reduction_db_zero_passes_through(sr: int) -> None:
    """reduction_db=0 should leave the signal essentially unchanged."""
    cfg = DuckConfig(
        reduction_db=0.0,
        attack=0.01,
        release=0.4,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)

    target = np.ones(sr, dtype=np.float32) * 0.4
    trigger = np.ones(sr, dtype=np.float32) * 0.5

    result = ducker.apply(target, trigger, sr)

    np.testing.assert_allclose(result.flatten(), target.flatten(), atol=0.01)


def test_reduction_db_must_be_nonpositive() -> None:
    """DuckConfig must reject positive reduction_db values."""
    with pytest.raises(Exception, match="reduction_db"):
        DuckConfig(reduction_db=3.0)
