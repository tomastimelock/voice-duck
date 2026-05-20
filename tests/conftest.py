# Filepath: tests/conftest.py
# Condensed Description: Shared pytest fixtures providing synthetic audio signals for voice-duck tests.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: fixtures: sr, sine_1khz, silence, loud_trigger, stereo_target, stereo_trigger, tmp_wav
# Dependencies: Internal: none / External: pytest, numpy, soundfile
# Exposes: sr, sine_1khz, silence, loud_trigger, stereo_target, stereo_trigger, tmp_wav
# Configuration: none
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sr() -> int:
    """Default sample rate used across all tests."""
    return 48000


@pytest.fixture
def sine_1khz(sr: int) -> np.ndarray:
    """1-second mono float32 1 kHz sine at near-unity amplitude."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2 * np.pi * 1000 * t).astype(np.float32)


@pytest.fixture
def silence(sr: int) -> np.ndarray:
    """1-second mono float32 silence."""
    return np.zeros(sr, dtype=np.float32)


@pytest.fixture
def loud_trigger(sr: int) -> np.ndarray:
    """0.5-second mono float32 signal well above -40 dBFS threshold."""
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def stereo_target(sr: int) -> np.ndarray:
    """1-second stereo float32 signal at moderate level, shape (sr, 2)."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mono = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


@pytest.fixture
def stereo_trigger(sr: int) -> np.ndarray:
    """1-second stereo float32 trigger signal above threshold, shape (sr, 2)."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.stack([mono, mono * 0.8], axis=1)


@pytest.fixture
def tmp_wav(tmp_path: Path, sr: int) -> Callable[[np.ndarray, str, int | None], Path]:
    """Factory: write an ndarray to a temp WAV file and return its Path."""

    def _make(audio: np.ndarray, name: str = "test.wav", sample_rate: int | None = None) -> Path:
        p = tmp_path / name
        sf.write(str(p), audio, sample_rate or sr)
        return p

    return _make
