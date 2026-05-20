# Filepath: src/voice_duck/io.py
# Condensed Description: soundfile-backed load() and save() wrappers normalised to float32 (frames, channels).
# Architecture Layer: Adapter
# Environment: Local
# Script Hierarchy: load, save
# Dependencies: Internal: none / External: soundfile, numpy
# Exposes: load, save
# Configuration: none
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file and return a float32 array with shape (frames, channels).

    Args:
        path: Path to the audio file (any format supported by libsndfile).

    Returns:
        A tuple of (audio, sample_rate) where audio is a float32 ndarray of
        shape (frames, channels) and sample_rate is an integer.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If libsndfile cannot open or decode the file.
    """
    resolved = Path(path).expanduser().resolve()
    logger.info("Loading audio: %s", resolved)
    audio, sample_rate = sf.read(str(resolved), always_2d=True, dtype="float32")
    logger.debug("Loaded: shape=%s, sample_rate=%d", audio.shape, sample_rate)
    return audio, sample_rate


def save(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write audio to a file as float32, inferring format from the extension.

    Args:
        path: Destination file path. The format is inferred from the extension
            (e.g. ``.wav``, ``.flac``, ``.ogg``).
        audio: Audio array of shape (frames,) or (frames, channels), any
            numeric dtype — will be converted to float32 before writing.
        sample_rate: Sample rate in Hz.

    Returns:
        The resolved Path that was written.

    Raises:
        ValueError: If the file extension is not supported by libsndfile.
    """
    resolved = Path(path).expanduser().resolve()
    ext = resolved.suffix.lstrip(".").upper()

    # soundfile.available_formats() returns a dict keyed by uppercase extension strings.
    supported = sf.available_formats()
    if ext not in supported:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported extensions: {sorted(supported.keys())}"
        )

    audio_f32 = audio.astype(np.float32)
    logger.info(
        "Writing audio: %s (shape=%s, sample_rate=%d)", resolved, audio_f32.shape, sample_rate
    )
    sf.write(str(resolved), audio_f32, sample_rate, subtype="FLOAT")
    return resolved
