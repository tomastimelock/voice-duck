# Filepath: tests/test_hold.py
# Condensed Description: Tests that the hold parameter prevents premature release between short voice bursts.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_hold_prevents_release_between_bursts, test_hold_zero_allows_release, test_hold_expires_after_silence
# Dependencies: Internal: voice_duck.config, voice_duck.ducker / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np
import pytest

from voice_duck.config import DuckConfig
from voice_duck.ducker import Ducker


def _two_burst_trigger(sr: int) -> np.ndarray:
    """Build a 3-second trigger with two 100 ms bursts and a 200 ms gap between them."""
    n = 3 * sr
    trigger = np.zeros(n, dtype=np.float32)
    # First burst: 0.0-0.1 s
    trigger[: int(0.1 * sr)] = 0.5
    # Gap: 0.1-0.3 s (silence - shorter than a 0.5 s hold)
    # Second burst: 0.3-0.4 s
    trigger[int(0.3 * sr) : int(0.4 * sr)] = 0.5
    # Rest: silence
    return trigger


def test_hold_prevents_release_between_bursts(sr: int) -> None:
    """With hold=0.5 s the gain must remain ducked during the 200 ms gap between bursts."""
    trigger = _two_burst_trigger(sr)
    cfg = DuckConfig(
        hold=0.5,
        attack=0.01,
        release=0.05,
        reduction_db=-20.0,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)
    gain = ducker.envelope(trigger, sr)

    # At t=0.2 s (midpoint of the gap), hold is still active — gain must be ducked.
    gap_frame = int(0.2 * sr)
    assert float(gain[gap_frame]) < 0.9


def test_hold_zero_allows_release(sr: int) -> None:
    """With hold=0 the gain should start recovering during the gap between bursts."""
    trigger = _two_burst_trigger(sr)
    cfg = DuckConfig(
        hold=0.0,
        attack=0.01,
        release=0.05,  # fast release so it can recover in 200 ms
        reduction_db=-20.0,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)
    gain = ducker.envelope(trigger, sr)

    # At t=0.25 s (well into the gap and 5x release after first burst ends),
    # gain should have returned significantly toward 1.0.
    late_gap_frame = int(0.25 * sr)
    assert float(gain[late_gap_frame]) > 0.5  # meaningfully recovered


def test_hold_expires_after_silence(sr: int) -> None:
    """After the hold period expires and trigger is silent, gain must return toward 1.0."""
    n = 3 * sr
    trigger = np.zeros(n, dtype=np.float32)
    # Single burst at the very start
    trigger[: int(0.05 * sr)] = 0.5
    hold = 0.2
    cfg = DuckConfig(
        hold=hold,
        attack=0.01,
        release=0.05,  # fast release — recovers quickly after hold
        reduction_db=-20.0,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)
    gain = ducker.envelope(trigger, sr)

    # After hold expires and enough release time, gain must be close to 1.0.
    # Check at t = hold + 5 * release = 0.2 + 0.25 = 0.45 s
    recovered_frame = int((hold + 5 * 0.05) * sr)
    assert float(gain[recovered_frame]) > 0.9


def test_hold_nonnegative_required() -> None:
    """DuckConfig must reject negative hold values."""
    with pytest.raises(Exception, match="hold"):
        DuckConfig(hold=-0.1)
