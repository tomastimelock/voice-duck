# Filepath: src/voice_duck/config.py
# Condensed Description: Pydantic v2 DuckConfig model with field validation for all ducking parameters.
# Architecture Layer: Utility
# Environment: Local
# Script Hierarchy: DuckConfig
# Dependencies: Internal: none / External: pydantic
# Exposes: DuckConfig
# Configuration: none
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DuckConfig(BaseModel):
    """Configuration for the voice-duck sidechain compressor.

    All time values are in seconds. ``reduction_db`` must be <= 0 (it is a
    gain reduction, not a boost). ``attack`` and ``release`` must be > 0.
    ``lookahead`` and ``hold`` must be >= 0.
    """

    model_config = ConfigDict(frozen=True)

    reduction_db: float = Field(
        default=-12.0,
        description="Target gain reduction in dB when ducking is active. Must be <= 0.",
    )
    attack: float = Field(
        default=0.05, description="Time constant in seconds for the ducking to engage."
    )
    release: float = Field(
        default=0.4, description="Time constant in seconds for the ducking to release."
    )
    threshold_db: float = Field(
        default=-40.0,
        description="Trigger level threshold in dBFS; below this is treated as silence.",
    )
    lookahead: float = Field(
        default=0.0, description="Seconds of pre-roll ducking before the trigger fires."
    )
    hold: float = Field(
        default=0.0, description="Minimum seconds at full duck after the trigger falls silent."
    )
    detector: Literal["rms", "peak"] = Field(
        default="rms", description="Envelope detector algorithm."
    )
    rms_window: float = Field(default=0.01, description="RMS averaging window length in seconds.")

    @field_validator("reduction_db")
    @classmethod
    def reduction_must_be_nonpositive(cls, v: float) -> float:
        """Validate that reduction_db is <= 0."""
        if v > 0:
            raise ValueError("reduction_db must be <= 0")
        return v

    @field_validator("attack", "release")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        """Validate that attack and release are strictly positive."""
        if v <= 0:
            raise ValueError("attack and release must be > 0")
        return v

    @field_validator("lookahead", "hold")
    @classmethod
    def must_be_nonnegative(cls, v: float) -> float:
        """Validate that lookahead and hold are non-negative."""
        if v < 0:
            raise ValueError("lookahead and hold must be >= 0")
        return v
