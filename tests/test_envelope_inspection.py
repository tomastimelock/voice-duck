# Filepath: tests/test_envelope_inspection.py
# Condensed Description: Tests that Ducker.envelope() returns correct shape and bounded values.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_envelope_shape, test_envelope_values_bounded, test_envelope_unity_when_silent, test_envelope_minimum_at_full_duck
# Dependencies: Internal: voice_duck.config, voice_duck.ducker, voice_duck.utils / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np
import pytest

from voice_duck.config import DuckConfig
from voice_duck.ducker import Ducker
from voice_duck.utils import db_to_linear


def test_envelope_shape(sr: int) -> None:
    """Ducker.envelope() must return a 1-D array with length equal to the trigger length."""
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    d = Ducker(DuckConfig(reduction_db=-12.0))
    env = d.envelope(trigger, sr)

    assert env.ndim == 1
    assert env.shape == (sr,)


def test_envelope_dtype_is_float32(sr: int) -> None:
    """Ducker.envelope() must return a float32 array."""
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    d = Ducker(DuckConfig(reduction_db=-12.0))
    env = d.envelope(trigger, sr)

    assert env.dtype == np.float32


def test_envelope_values_bounded(sr: int) -> None:
    """All gain values must lie in [db_to_linear(reduction_db), 1.0]."""
    reduction_db = -20.0
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    d = Ducker(DuckConfig(reduction_db=reduction_db, attack=0.01, release=0.1))
    env = d.envelope(trigger, sr)

    lower_bound = db_to_linear(reduction_db)
    assert float(np.min(env)) >= lower_bound - 1e-4
    assert float(np.max(env)) <= 1.0 + 1e-4


def test_envelope_unity_when_silent(sr: int) -> None:
    """All-silence trigger must produce a unity (1.0) gain envelope."""
    trigger = np.zeros(sr, dtype=np.float32)
    d = Ducker(DuckConfig(reduction_db=-20.0, threshold_db=-40.0))
    env = d.envelope(trigger, sr)

    np.testing.assert_allclose(env, 1.0, atol=0.01)


def test_envelope_minimum_at_full_duck(sr: int) -> None:
    """After settling with a loud trigger, minimum gain must approach db_to_linear(reduction_db)."""
    reduction_db = -20.0
    # Use 3 seconds so the IIR has time to fully settle
    n = 3 * sr
    trigger = np.ones(n, dtype=np.float32) * 0.5
    d = Ducker(DuckConfig(reduction_db=reduction_db, attack=0.01, release=0.1, threshold_db=-40.0))
    env = d.envelope(trigger, sr)

    # Examine the settled second half
    settled = env[sr // 2 :]
    min_gain = float(np.min(settled))
    expected = db_to_linear(reduction_db)

    assert abs(min_gain - expected) < 0.05  # within 5% of expected linear gain


def test_envelope_starts_at_unity(sr: int) -> None:
    """The first sample of the gain envelope must be 1.0 when the trigger starts silent."""
    trigger = np.zeros(sr, dtype=np.float32)
    # Trigger fires halfway through
    trigger[sr // 2 :] = 0.5

    d = Ducker(DuckConfig(reduction_db=-20.0, attack=0.01, release=0.4, threshold_db=-40.0))
    env = d.envelope(trigger, sr)

    assert float(env[0]) == pytest.approx(1.0, abs=0.001)


def test_envelope_never_exceeds_unity(sr: int) -> None:
    """Gain must never exceed 1.0 — voice-duck is attenuation only, never boost."""
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    d = Ducker(DuckConfig(reduction_db=-6.0, attack=0.01, release=0.1))
    env = d.envelope(trigger, sr)

    assert float(np.max(env)) <= 1.0 + 1e-5


def test_envelope_long_trigger_shape(sr: int) -> None:
    """envelope() must return array with length matching the trigger, not just sr frames."""
    n = 3 * sr
    trigger = np.ones(n, dtype=np.float32) * 0.5
    d = Ducker(DuckConfig(reduction_db=-12.0))
    env = d.envelope(trigger, sr)

    assert env.shape == (n,)
