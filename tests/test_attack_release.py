# Filepath: tests/test_attack_release.py
# Condensed Description: Tests that the IIR smoother engages within attack time and recovers within release time.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_attack_time_approx, test_release_time_approx, test_attack_faster_than_release
# Dependencies: Internal: voice_duck.config, voice_duck.ducker / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np
import pytest

from voice_duck.config import DuckConfig
from voice_duck.ducker import Ducker


def test_attack_time_approx(sr: int) -> None:
    """Gain must start dropping meaningfully within the configured attack time."""
    attack = 0.1
    cfg = DuckConfig(
        attack=attack,
        release=2.0,
        reduction_db=-20.0,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)

    # Step trigger: silence for 0.1 s, then loud.
    n = 2 * sr
    trigger = np.zeros(n, dtype=np.float32)
    trigger[sr // 10 :] = 0.5  # loud from t=0.1 s onward

    gain = ducker.envelope(trigger, sr)

    # At t = 0.1 s + attack, the gain should have begun dropping.
    target_frame = sr // 10 + int(attack * sr)
    gain_db_at_target = 20 * np.log10(max(float(gain[target_frame]), 1e-9))
    # Must be meaningfully ducked — more than 1 dB below unity.
    assert gain_db_at_target < -1.0


def test_release_time_approx(sr: int) -> None:
    """After the trigger goes silent, gain must start recovering within the release time."""
    release = 0.3
    cfg = DuckConfig(
        attack=0.01,
        release=release,
        reduction_db=-20.0,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)

    # Loud trigger for 0.5 s then silence for 1.5 s.
    n = 2 * sr
    trigger = np.zeros(n, dtype=np.float32)
    trigger[: sr // 2] = 0.5  # loud for first 0.5 s

    gain = ducker.envelope(trigger, sr)

    # At the moment the trigger goes silent the gain should be near reduction_db.
    off_frame = sr // 2
    gain_at_off = float(gain[off_frame])

    # After one full release period, gain must have recovered significantly.
    check_frame = off_frame + int(release * sr)
    if check_frame < n:
        gain_after_release = float(gain[check_frame])
        # Should be noticeably higher than when the trigger first went off.
        assert gain_after_release > gain_at_off + 0.05  # at least 5% recovered


def test_attack_faster_than_release(sr: int) -> None:
    """With attack=0.02 and release=0.5, duck must engage before it fully releases."""
    attack = 0.02
    release = 0.5
    cfg = DuckConfig(
        attack=attack,
        release=release,
        reduction_db=-20.0,
        threshold_db=-40.0,
    )
    ducker = Ducker(cfg)

    # Step on: loud from t=0 to t=0.5s, then silent for 1s
    n = int(1.5 * sr)
    trigger = np.zeros(n, dtype=np.float32)
    trigger[: sr // 2] = 0.5

    gain = ducker.envelope(trigger, sr)

    # Attack frame: gain at t = 5 * attack should be close to full duck.
    attack_check = int(5 * attack * sr)
    gain_db_ducked = 20 * np.log10(max(float(gain[attack_check]), 1e-9))

    # Release frame: gain at t = 0.5s + 5 * release would be needed to be fully
    # released — but that's 3 s total which is beyond our array; instead
    # just verify gain at t=0.5s + release is still clearly below unity.
    release_check = sr // 2 + int(release * sr)
    if release_check < n:
        gain_db_releasing = 20 * np.log10(max(float(gain[release_check]), 1e-9))
        # Duck should have reached deeper than the partial release reading
        assert gain_db_ducked < gain_db_releasing


def test_attack_positive_required() -> None:
    """DuckConfig must reject attack <= 0."""
    with pytest.raises(Exception, match="attack"):
        DuckConfig(attack=0.0, release=0.4)


def test_release_positive_required() -> None:
    """DuckConfig must reject release <= 0."""
    with pytest.raises(Exception, match="release"):
        DuckConfig(attack=0.05, release=0.0)
