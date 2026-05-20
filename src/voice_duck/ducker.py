# Filepath: src/voice_duck/ducker.py
# Condensed Description: duck() one-shot function and Ducker class — public-facing sidechain processing entry points.
# Architecture Layer: Pipeline
# Environment: Local
# Script Hierarchy: duck, Ducker.apply, Ducker.envelope
# Dependencies: Internal: voice_duck.config, voice_duck.envelope, voice_duck.gain_curve, voice_duck.utils, voice_duck.io / External: numpy
# Exposes: duck, Ducker
# Configuration: none
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from voice_duck.config import DuckConfig
from voice_duck.envelope import compute_envelope
from voice_duck.gain_curve import build_gain_curve
from voice_duck.io import load
from voice_duck.utils import match_channels

logger = logging.getLogger(__name__)


def duck(
    target: np.ndarray | str | Path,
    trigger: np.ndarray | str | Path,
    sample_rate: int | None = None,
    reduction_db: float = -12.0,
    attack: float = 0.05,
    release: float = 0.4,
    threshold_db: float = -40.0,
    lookahead: float = 0.0,
    hold: float = 0.0,
    config: DuckConfig | None = None,
) -> np.ndarray:
    """Apply sidechain ducking: attenuate the target whenever the trigger is loud.

    Accepts paths or numpy arrays for both inputs. When a path is given it is
    loaded via ``io.load``. When a numpy array is given ``sample_rate`` must be
    provided. The ``config`` argument, if given, overrides all individual
    parameter arguments.

    Args:
        target: Audio to be ducked. Path string/Path or float32 ndarray of
            shape (frames,) or (frames, channels).
        trigger: Audio whose loudness controls the ducking. Same types as
            target.
        sample_rate: Sample rate in Hz. Required when either input is an array.
        reduction_db: Target gain in dB when ducking is active. Must be <= 0.
        attack: Ducking engagement time constant in seconds.
        release: Ducking release time constant in seconds.
        threshold_db: Trigger threshold in dBFS; below this is silence.
        lookahead: Pre-roll in seconds; duck begins this early.
        hold: Minimum hold time in seconds once full duck is reached.
        config: If provided, overrides all individual parameter arguments.

    Returns:
        Float32 array of the ducked target, same shape as the input target.

    Raises:
        ValueError: If sample_rate is not provided when an array is passed,
            or if the sample rates of the two inputs differ.
    """
    # Resolve config
    if config is None:
        config = DuckConfig(
            reduction_db=reduction_db,
            attack=attack,
            release=release,
            threshold_db=threshold_db,
            lookahead=lookahead,
            hold=hold,
        )

    # Load from file paths or validate sample_rate for arrays
    target_sr: int
    trigger_sr: int

    if isinstance(target, (str, Path)):
        target_audio, target_sr = load(target)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate must be provided when target is a numpy array.")
        target_audio = np.asarray(target, dtype=np.float32)
        target_sr = sample_rate

    if isinstance(trigger, (str, Path)):
        trigger_audio, trigger_sr = load(trigger)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate must be provided when trigger is a numpy array.")
        trigger_audio = np.asarray(trigger, dtype=np.float32)
        trigger_sr = sample_rate

    # Remember whether the original target was 1-D so we can restore that shape.
    target_was_1d = target_audio.ndim == 1

    # Ensure both are 2-D (frames, channels) for uniform processing.
    if target_audio.ndim == 1:
        target_audio = target_audio[:, np.newaxis]
    if trigger_audio.ndim == 1:
        trigger_audio = trigger_audio[:, np.newaxis]

    # Validate sample rates match.
    match_channels(target_audio, trigger_audio, target_sr, trigger_sr)

    # Compute trigger envelope (mono mixdown happens inside compute_envelope).
    env = compute_envelope(
        trigger_audio,
        target_sr,
        config.detector,
        config.rms_window,
    )

    # Build gain curve.
    gain = build_gain_curve(env, target_sr, config)

    # Apply: broadcast gain across all channels.
    ducked = target_audio * gain[:, np.newaxis]
    ducked = ducked.astype(np.float32)

    logger.info(
        "duck: applied gain curve, target.shape=%s, gain min=%.4f max=%.4f",
        ducked.shape,
        gain.min(),
        gain.max(),
    )

    if target_was_1d:
        return ducked[:, 0]
    return ducked


class Ducker:
    """Reusable sidechain ducker with persistent configuration.

    Accepts the same parameters as ``duck()``. Once constructed the same
    ``DuckConfig`` is reused across all ``apply()`` calls, which avoids
    repeated validation overhead when processing many segments.
    """

    def __init__(self, config: DuckConfig | None = None, **kwargs: object) -> None:
        """Initialise the Ducker with a DuckConfig or keyword arguments.

        Args:
            config: Pre-built DuckConfig. If provided, kwargs are ignored.
            **kwargs: Keyword arguments forwarded to DuckConfig if config is
                not given (e.g. reduction_db=-18, attack=0.02).
        """
        if config is not None:
            self._config = config
        else:
            self._config = DuckConfig(**kwargs)  # type: ignore[arg-type]

    def apply(
        self,
        target: np.ndarray,
        trigger: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Duck the target using the trigger and the Ducker's config.

        Args:
            target: Audio to be ducked, shape (frames,) or (frames, channels).
            trigger: Audio that controls ducking, same shape options.
            sample_rate: Shared sample rate in Hz.

        Returns:
            Float32 ducked audio, same shape as target.
        """
        return duck(target, trigger, sample_rate=sample_rate, config=self._config)

    def envelope(self, trigger: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the gain envelope without applying it to any target audio.

        Useful for visualisation, debugging, and testing. The returned values
        are linear gain multipliers (1.0 = unity, lower = more ducking).

        Args:
            trigger: Trigger audio array, shape (frames,) or (frames, channels).
            sample_rate: Sample rate in Hz.

        Returns:
            Float32 array of shape (frames,) with linear gain values in
            [db_to_linear(reduction_db), 1.0].
        """
        env = compute_envelope(
            trigger,
            sample_rate,
            self._config.detector,
            self._config.rms_window,
        )
        return build_gain_curve(env, sample_rate, self._config)
