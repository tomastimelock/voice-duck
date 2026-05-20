# Filepath: tests/test_channels.py
# Condensed Description: Tests channel-handling logic — mono/stereo combinations and envelope broadcast.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_mono_trigger_stereo_target_both_channels_equal, test_stereo_trigger_mixdown_for_detection, test_mono_target_stereo_trigger, test_output_shape_matches_target
# Dependencies: Internal: voice_duck.config, voice_duck.ducker / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np

from voice_duck.config import DuckConfig
from voice_duck.ducker import Ducker


def _default_cfg() -> DuckConfig:
    return DuckConfig(reduction_db=-12.0, attack=0.01, release=0.1, threshold_db=-40.0)


def test_mono_trigger_stereo_target_both_channels_equal(
    stereo_target: np.ndarray,
    loud_trigger: np.ndarray,
    sr: int,
) -> None:
    """Mono trigger + stereo target: both output channels must be identical."""
    assert loud_trigger.ndim == 1, "loud_trigger fixture must be mono"
    assert stereo_target.ndim == 2 and stereo_target.shape[1] == 2

    # Pad trigger to match target length
    trig = np.zeros(len(stereo_target), dtype=np.float32)
    trig[: len(loud_trigger)] = loud_trigger

    ducker = Ducker(_default_cfg())
    result = ducker.apply(stereo_target, trig, sr)

    assert result.shape == stereo_target.shape
    np.testing.assert_allclose(result[:, 0], result[:, 1], atol=1e-6)


def test_stereo_trigger_mixdown_for_detection(
    stereo_target: np.ndarray,
    stereo_trigger: np.ndarray,
    sr: int,
) -> None:
    """Stereo trigger must be mixed to mono for detection; result shape matches target."""
    ducker = Ducker(_default_cfg())
    result = ducker.apply(stereo_target, stereo_trigger, sr)

    assert result.shape == stereo_target.shape


def test_stereo_trigger_same_as_mono_mixdown(
    stereo_target: np.ndarray,
    stereo_trigger: np.ndarray,
    sr: int,
) -> None:
    """Stereo trigger should produce same gain curve as explicit mono mixdown of that trigger."""
    cfg = _default_cfg()
    ducker = Ducker(cfg)

    # Result from stereo trigger
    result_stereo_trig = ducker.apply(stereo_target, stereo_trigger, sr)

    # Manually mix trigger to mono and apply
    mono_trig = stereo_trigger.mean(axis=1).astype(np.float32)
    result_mono_trig = ducker.apply(stereo_target, mono_trig, sr)

    np.testing.assert_allclose(result_stereo_trig, result_mono_trig, atol=1e-5)


def test_mono_target_stereo_trigger(sr: int) -> None:
    """Mono target + stereo trigger must work without error and return a 1-D array."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mono_target = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    stereo_trig = np.stack([0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)] * 2, axis=1)

    ducker = Ducker(_default_cfg())
    result = ducker.apply(mono_target, stereo_trig, sr)

    # Mono input → mono output
    assert result.ndim == 1
    assert len(result) == sr


def test_output_shape_matches_target(
    stereo_target: np.ndarray,
    loud_trigger: np.ndarray,
    sr: int,
) -> None:
    """Output array shape must exactly match target shape for all channel combinations."""
    trig = np.zeros(len(stereo_target), dtype=np.float32)
    trig[: len(loud_trigger)] = loud_trigger

    ducker = Ducker(_default_cfg())
    result = ducker.apply(stereo_target, trig, sr)

    assert result.shape == stereo_target.shape


def test_both_channels_ducked_equally_with_stereo_trigger(sr: int) -> None:
    """Stereo trigger's mono envelope must duck both target channels by the same amount."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Asymmetric stereo trigger: L is loud, R is quieter
    left_loud = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right_quiet = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo_trig = np.stack([left_loud, right_quiet], axis=1)

    # Stereo target with identical L and R
    mono = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    target = np.stack([mono, mono], axis=1)

    ducker = Ducker(DuckConfig(reduction_db=-12.0, attack=0.01, release=0.1, threshold_db=-40.0))
    result = ducker.apply(target, stereo_trig, sr)

    # Both channels must receive identical gain
    np.testing.assert_allclose(result[:, 0], result[:, 1], atol=1e-6)
