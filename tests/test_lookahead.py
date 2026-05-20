# Filepath: tests/test_lookahead.py
# Condensed Description: Tests that the lookahead parameter shifts the gain envelope earlier in time.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_lookahead_shifts_envelope_earlier, test_zero_lookahead_no_preroll, test_lookahead_output_length_unchanged
# Dependencies: Internal: voice_duck.config, voice_duck.ducker / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np
import pytest

from voice_duck.config import DuckConfig
from voice_duck.ducker import Ducker


def test_lookahead_shifts_envelope_earlier(sr: int) -> None:
    """With lookahead=0.1 s and trigger starting at t=0.5 s, gain must drop before t=0.5 s."""
    n = 2 * sr
    trigger = np.zeros(n, dtype=np.float32)
    trigger[sr // 2 :] = 0.5  # trigger starts at t=0.5 s

    lookahead = 0.1
    cfg = DuckConfig(
        lookahead=lookahead,
        attack=0.01,
        release=0.4,
        reduction_db=-20.0,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)
    gain = ducker.envelope(trigger, sr)

    assert len(gain) == n

    # At t = 0.5 s - lookahead/2, the gain should have already started dropping.
    early_frame = int((0.5 - lookahead * 0.5) * sr)
    assert float(gain[early_frame]) < 0.95


def test_zero_lookahead_no_preroll(sr: int) -> None:
    """With lookahead=0 the gain must be unity before the trigger fires."""
    n = 2 * sr
    trigger = np.zeros(n, dtype=np.float32)
    trigger[sr // 2 :] = 0.5  # trigger starts at t=0.5 s

    cfg = DuckConfig(
        lookahead=0.0,
        attack=0.01,
        release=0.4,
        reduction_db=-20.0,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)
    gain = ducker.envelope(trigger, sr)

    # At t=0.45 s (50 ms before trigger fires) gain must still be unity.
    pre_frame = int(0.45 * sr)
    np.testing.assert_allclose(gain[:pre_frame], 1.0, atol=0.01)


def test_lookahead_output_length_unchanged(sr: int) -> None:
    """Output array length must equal input length regardless of lookahead setting."""
    n = 2 * sr
    trigger = np.zeros(n, dtype=np.float32)
    trigger[sr // 2 :] = 0.5

    for la in (0.0, 0.05, 0.1, 0.2):
        cfg = DuckConfig(
            lookahead=la, attack=0.01, release=0.4, reduction_db=-20.0, threshold_db=-40.0
        )
        gain = Ducker(cfg).envelope(trigger, sr)
        assert len(gain) == n, f"Length mismatch for lookahead={la}"


def test_lookahead_nonnegative_required() -> None:
    """DuckConfig must reject negative lookahead values."""
    with pytest.raises(Exception, match="lookahead"):
        DuckConfig(lookahead=-0.05)
