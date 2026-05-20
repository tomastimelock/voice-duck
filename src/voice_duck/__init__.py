# Filepath: src/voice_duck/__init__.py
# Condensed Description: Package entry point — version, public API re-exports for voice_duck.
# Architecture Layer: Utility
# Environment: Local
# Script Hierarchy: none
# Dependencies: Internal: voice_duck.ducker, voice_duck.config, voice_duck.io / External: none
# Exposes: duck, Ducker, DuckConfig, load, save, __version__
# Configuration: none
from __future__ import annotations

__version__ = "0.1.0"

from voice_duck.config import DuckConfig
from voice_duck.ducker import Ducker, duck
from voice_duck.io import load, save

__all__ = ["DuckConfig", "Ducker", "duck", "load", "save"]
