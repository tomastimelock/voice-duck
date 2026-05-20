# Filepath: tests/test_clear_errors.py
# Condensed Description: Tests that sample-rate mismatches raise ValueError with both rates in the message.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_sample_rate_mismatch_raises, test_error_message_contains_both_rates, test_matching_rates_no_error
# Dependencies: Internal: voice_duck.utils, voice_duck.ducker / External: pytest, numpy
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

import numpy as np
import pytest

from voice_duck.utils import match_channels


def test_sample_rate_mismatch_raises(sr: int) -> None:
    """match_channels must raise ValueError when sample rates differ."""
    target = np.ones((sr, 2), dtype=np.float32) * 0.3
    trigger = np.ones(sr, dtype=np.float32) * 0.5

    with pytest.raises(ValueError):
        match_channels(target, trigger, sample_rate_target=48000, sample_rate_trigger=44100)


def test_error_message_contains_both_rates(sr: int) -> None:
    """The ValueError message from a sample-rate mismatch must include both rate values."""
    target = np.ones((sr, 2), dtype=np.float32)
    trigger = np.ones(sr, dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        match_channels(target, trigger, sample_rate_target=48000, sample_rate_trigger=22050)

    message = str(exc_info.value)
    assert "48000" in message
    assert "22050" in message


def test_matching_rates_no_error(sr: int) -> None:
    """match_channels must return the arrays unchanged when sample rates are equal."""
    target = np.ones((sr, 2), dtype=np.float32)
    trigger = np.ones(sr, dtype=np.float32)

    t_out, tr_out = match_channels(target, trigger, sample_rate_target=sr, sample_rate_trigger=sr)

    assert t_out is target
    assert tr_out is trigger


def test_error_message_contains_resample_hint(sr: int) -> None:
    """The ValueError message should mention resampling as the remedy."""
    target = np.ones((sr, 2), dtype=np.float32)
    trigger = np.ones(sr, dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        match_channels(target, trigger, sample_rate_target=48000, sample_rate_trigger=44100)

    assert "resample" in str(exc_info.value).lower()


def test_duck_requires_sample_rate_for_arrays(sr: int) -> None:
    """duck() called with numpy arrays but no sample_rate must raise ValueError."""
    from voice_duck.ducker import duck

    target = np.ones((sr, 2), dtype=np.float32) * 0.4
    trigger = np.ones(sr, dtype=np.float32) * 0.5

    with pytest.raises(ValueError, match="sample_rate"):
        duck(target, trigger)  # no sample_rate
