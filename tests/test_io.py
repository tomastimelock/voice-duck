# Filepath: tests/test_io.py
# Condensed Description: Tests for voice_duck.io load() and save() wrappers.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_load_roundtrip, test_save_unsupported_extension_raises, test_load_nonexistent_raises
# Dependencies: Internal: voice_duck.io / External: pytest, numpy, soundfile
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_duck.io import load, save


def test_load_roundtrip(tmp_path: Path, sr: int, sine_1khz: np.ndarray) -> None:
    """load() must return a float32 2-D array with the correct sample rate."""
    p = tmp_path / "test.wav"
    # Write a mono signal; soundfile with always_2d should give (frames, 1)
    sf.write(str(p), sine_1khz, sr)

    audio, loaded_sr = load(p)

    assert loaded_sr == sr
    assert audio.dtype == np.float32
    assert audio.ndim == 2
    assert audio.shape[0] == len(sine_1khz)


def test_save_roundtrip(tmp_path: Path, sr: int, stereo_target: np.ndarray) -> None:
    """save() followed by load() must reproduce the audio array within float32 tolerance."""
    p = tmp_path / "saved.wav"
    save(p, stereo_target, sr)

    audio, loaded_sr = load(p)

    assert loaded_sr == sr
    np.testing.assert_allclose(audio, stereo_target, atol=1e-5)


def test_save_returns_resolved_path(tmp_path: Path, sr: int) -> None:
    """save() must return the resolved Path of the written file."""
    p = tmp_path / "out.wav"
    audio = np.zeros((sr, 2), dtype=np.float32)
    result = save(p, audio, sr)

    assert isinstance(result, Path)
    assert result.exists()


def test_save_unsupported_extension_raises(tmp_path: Path, sr: int) -> None:
    """save() must raise ValueError for an extension libsndfile does not support."""
    p = tmp_path / "audio.xyz_unsupported"
    audio = np.zeros((sr, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported"):
        save(p, audio, sr)


def test_load_nonexistent_raises(tmp_path: Path) -> None:
    """load() must raise when the file does not exist."""
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        load(tmp_path / "no_such_file.wav")


def test_save_converts_to_float32(tmp_path: Path, sr: int) -> None:
    """save() must accept int16-typed input and write it as float32."""
    p = tmp_path / "int16.wav"
    audio_int16 = (np.ones((sr, 2)) * 0.4).astype(np.float64)  # float64 input

    save(p, audio_int16, sr)

    loaded, _ = load(p)
    assert loaded.dtype == np.float32
