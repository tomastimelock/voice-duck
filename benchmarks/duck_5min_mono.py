# Filepath: benchmarks/duck_5min_mono.py
# Condensed Description: Benchmark Ducker.apply() on 5 minutes of mono float32 audio at 48 kHz
# Architecture Layer: Benchmark
# Environment: Local
# Script Hierarchy: Standalone
# Dependencies: Internal: voice_duck / External: numpy, time
# Exposes: Prints duck_time_s result with PASS/FAIL against 2.0s target
# Configuration: SAMPLE_RATE=48000, DURATION_S=300, TARGET_S=2.0

"""Benchmark: duck a 5-minute mono audio signal.

Synthesises a white-noise target and a sine-burst trigger (simulating speech),
then times Ducker.apply() using time.perf_counter().

Run from the repo root:
    python benchmarks/duck_5min_mono.py

Always exits 0 so it never breaks CI.
"""

import sys
import time

import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000
DURATION_S = 5 * 60  # 5 minutes
N_FRAMES = SAMPLE_RATE * DURATION_S
TARGET_S = 2.0  # acceptable wall-clock ceiling
RNG_SEED = 42

# ── synthesise test signals ────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)

# Target: white noise (represents music bed)
target = rng.standard_normal(N_FRAMES).astype(np.float32) * 0.3

# Trigger: 1 kHz sine bursts every 2 seconds for 1 second (simulates speech)
trigger = np.zeros(N_FRAMES, dtype=np.float32)
t = np.arange(N_FRAMES, dtype=np.float32) / SAMPLE_RATE
for burst_start in range(0, DURATION_S, 2):
    i0 = burst_start * SAMPLE_RATE
    i1 = i0 + SAMPLE_RATE
    trigger[i0:i1] = np.sin(2.0 * np.pi * 1000.0 * t[i0:i1]) * 0.8

# ── import package ─────────────────────────────────────────────────────────────
try:
    from voice_duck import DuckConfig, Ducker
except ImportError as exc:
    print(f"SKIP  voice_duck not importable: {exc}")
    sys.exit(0)

# ── baseline: pure numpy ────────────────────────────────────────────────────────
config = DuckConfig(
    reduction_db=-12.0,
    attack=0.05,
    release=0.4,
    threshold_db=-40.0,
    lookahead=0.0,
    hold=0.0,
)
ducker = Ducker(config=config)

# Warm up (avoid cold-start JIT or import penalty in the timed section)
_warmup_target = target[:SAMPLE_RATE].copy()
_warmup_trigger = trigger[:SAMPLE_RATE].copy()
_ = ducker.apply(_warmup_target, _warmup_trigger, SAMPLE_RATE)

t0 = time.perf_counter()
result = ducker.apply(target.copy(), trigger, SAMPLE_RATE)
t1 = time.perf_counter()
duck_time = t1 - t0

status = "PASS" if duck_time <= TARGET_S else "FAIL"
print(f"duck_time_s: {duck_time:.3f}s  (5 min mono, 48kHz)  {status}  [target <= {TARGET_S}s]")

# ── optional: numba path ────────────────────────────────────────────────────────
# In v0.2, the one-pole IIR smoother in envelope.py will be JIT-compiled with
# Numba. The expected speedup is >= 2x on the attack/release loop, which is the
# only stateful (non-vectorised) step in the pipeline.
try:
    import numba as nb  # noqa: F401

    print(
        "numba available — "
        "IIR smoother JIT path not yet implemented (planned for v0.2). "
        "Re-run after upgrading to voice-duck[fast] v0.2+ to see the speedup."
    )
except ImportError:
    print(
        "numba not installed — install voice-duck[fast] for the planned v0.2 IIR JIT acceleration."
    )

# Always exit 0 so this script never blocks CI.
sys.exit(0)
