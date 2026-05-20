# voice-duck

Sidechain auto-ducking for voice-over music beds — one function, three numbers.

Built for narration-over-music workflows in CineForge and DocFlow at Trollfabriken AITrix AB after `pedalboard`'s missing sidechain forced an ffmpeg-shell-out for every dialogue track.

---

## What it solves

| Previous problem | Solution |
| --- | --- |
| `pedalboard.Compressor` has no sidechain input | Dedicated sidechain ducker; trigger and target are separate inputs |
| `ffmpeg sidechaincompress` requires opaque filter graph strings | `duck(target, trigger, reduction_db=-12)` — one function, three numbers |
| Naive pydub volume automation clicks on transitions | One-pole IIR smoother with configurable attack/release |
| Music ducks after the first syllable of speech | `lookahead` parameter shifts envelope earlier |
| Music briefly un-ducks between words | `hold` parameter keeps the duck engaged |
| Ducking computed on stereo trigger applies different curves per channel | Mono mixdown for detection; same envelope applied to both target channels |

---

## Installation

```bash
pip install voice-duck
```

With the Numba-accelerated IIR smoother (planned for v0.2, ~3x speedup on long files):

```bash
pip install "voice-duck[fast]"
```

Development install:

```bash
pip install "voice-duck[dev]"
```

---

## Quick start

```python
import soundfile as sf
from voice_duck import duck

# Load the music bed and the narration track
target, sr = sf.read("music_bed.wav", always_2d=True)   # shape: (frames, channels)
trigger, sr2 = sf.read("narration.wav", always_2d=True)

assert sr == sr2, "Sample rates must match — resample first"

# Duck the music by 12 dB whenever narration is above -40 dBFS.
# attack=0.05s so the duck engages fast; release=0.4s for a smooth fade-back.
# lookahead=0.05s prevents the first syllable from slipping through unduck.
ducked = duck(
    target,
    trigger,
    sample_rate=sr,
    reduction_db=-12.0,
    attack=0.05,
    release=0.4,
    threshold_db=-40.0,
    lookahead=0.05,
    hold=0.2,
)

# Write result — same shape and dtype as input
sf.write("ducked_output.wav", ducked, sr)
```

---

## How it works

```
  trigger signal (voice)
  ┌─────────────────┐
  │  raw audio      │
  └────────┬────────┘
           │
           ▼
  ① Envelope detection
     RMS window: 10 ms (default)
     or peak; configurable via DuckConfig.detector
           │
           ▼
  ② Threshold gate
     below threshold_db → gain target = 0 dB (no duck)
     above threshold_db → gain target = reduction_db
           │
           ▼
  ③ Lookahead shift
     zero-pad trigger by lookahead seconds;
     trim target end by same amount.
     Envelope moves ahead of the audio it controls.
           │
           ▼
  ④ Attack / release smoother
     one-pole IIR per frame
     attack coeff = exp(-1 / (sr * attack))
     release coeff = exp(-1 / (sr * release))
           │
           ▼
  ⑤ Hold counter
     once at full duck, do not release for hold seconds
     even if trigger goes silent
           │
           ▼
  ⑥ Channel broadcast
     trigger → mono mixdown for envelope
     envelope applied identically to all target channels
           │
           ▼
  ⑦ Apply gain
     output[n] = target[n] × 10^(envelope_db[n] / 20)
  ┌─────────────────┐
  │  ducked target  │
  └─────────────────┘
```

---

## Configuration

```python
from voice_duck import DuckConfig

config = DuckConfig(
    reduction_db=-12.0,    # dB to reduce music by when voice is active; must be <= 0
    attack=0.05,           # seconds for duck to engage; smaller = more aggressive
    release=0.4,           # seconds for music to fade back; larger = smoother
    threshold_db=-40.0,    # trigger sensitivity; frames quieter than this = silence
    lookahead=0.0,         # seconds to shift envelope earlier; eliminates first-syllable slip
    hold=0.0,              # seconds to stay ducked after trigger goes silent
    detector="rms",        # "rms" (default) or "peak"; rms is smoother on sibilants
    rms_window=0.01,       # RMS averaging window in seconds; default 10 ms
)
```

Field summary:

| Field | Default | Range | Notes |
| --- | --- | --- | --- |
| `reduction_db` | `-12.0` | `<= 0` | 0 = no duck, -inf = silence |
| `attack` | `0.05` | `> 0` | 50 ms is a good starting point for speech |
| `release` | `0.4` | `> 0` | 400 ms avoids the pumping artefact |
| `threshold_db` | `-40.0` | any | Raise to -30 in noisy rooms |
| `lookahead` | `0.0` | `>= 0` | 50–100 ms recommended when combined with hold |
| `hold` | `0.0` | `>= 0` | 200 ms prevents inter-word un-ducks |
| `detector` | `"rms"` | `"rms"`, `"peak"` | Peak responds faster; RMS is smoother |
| `rms_window` | `0.01` | `> 0` | 10 ms balances time resolution and noise floor |

---

## Testing without files

No audio files needed. Pass numpy arrays directly:

```python
import numpy as np
from voice_duck import Ducker, DuckConfig

sr = 48_000
duration = 3.0  # seconds
n = int(sr * duration)

# Target: constant tone
target = np.sin(2 * np.pi * 440 * np.arange(n) / sr).astype(np.float32)
target = target[:, np.newaxis]  # shape: (frames, 1)

# Trigger: 1 kHz sine at -6 dBFS for the middle second
trigger = np.zeros((n, 1), dtype=np.float32)
mid = slice(sr, 2 * sr)
trigger[mid, 0] = np.sin(2 * np.pi * 1000 * np.arange(sr) / sr) * 0.5

config = DuckConfig(reduction_db=-12.0, attack=0.05, release=0.4)
ducker = Ducker(config=config)

ducked = ducker.apply(target, trigger, sr)
assert ducked.shape == target.shape

# Inspect the gain envelope without applying it
envelope = ducker.envelope(trigger, sr)
assert envelope.shape == (n,)            # one value per frame
assert envelope.min() >= 10 ** (-12 / 20) - 1e-4  # floor at reduction_db
assert envelope.max() <= 1.0 + 1e-6     # ceiling at unity
```

---

## CLI

```bash
# Most common: duck music.mp3 by voice.wav, write out.wav
voice-duck --target music.mp3 --trigger voice.wav --out ducked.wav

# Tuned for podcast / aggressive ducking
voice-duck --target bed.wav --trigger narration.wav \
    --reduction -18 --attack 0.02 --release 0.6 --out ep01.wav

# Inspect the envelope without rendering (writes a CSV of gain values)
voice-duck --target bed.wav --trigger narration.wav \
    --inspect envelope.csv
```

---

## Package structure

```
voice-duck/
├── src/
│   └── voice_duck/
│       ├── __init__.py        ← version + public re-exports (duck, Ducker, DuckConfig, load, save)
│       ├── ducker.py          ← duck() one-shot function and Ducker class
│       ├── config.py          ← DuckConfig pydantic v2 model; field validation
│       ├── envelope.py        ← RMS / peak envelope follower
│       ├── gain_curve.py      ← attack/release/hold/lookahead curve generator
│       ├── io.py              ← load() / save() — soundfile wrappers
│       ├── cli.py             ← argparse CLI entry point
│       └── utils.py           ← dB↔linear helpers, channel matching
├── tests/
│   ├── test_envelope.py       ← RMS matches sqrt(0.5); peak == 1.0
│   ├── test_threshold.py      ← below-threshold input produces no ducking
│   ├── test_reduction.py      ← above-threshold drives envelope to reduction_db
│   ├── test_attack_release.py ← step response within 10% of attack/release time
│   ├── test_hold.py           ← two bursts within hold window stay continuously ducked
│   ├── test_lookahead.py      ← duck starts lookahead seconds before trigger fires
│   ├── test_channels.py       ← mono trigger + stereo target; stereo trigger mixdown
│   ├── test_clear_errors.py   ← sample-rate mismatch raises ValueError with both rates
│   ├── test_cli.py            ← CLI roundtrip; output exists and differs from input
│   └── test_envelope_inspection.py ← Ducker.envelope() shape and value range
├── benchmarks/
│   └── duck_5min_mono.py      ← times Ducker.apply() on 5 min mono 48 kHz; target <= 2.0s
├── pyproject.toml             ← hatchling build; ruff; pytest; coverage config
├── README.md
└── LICENSE                    ← MIT
```

---

© Trollfabriken AITrix AB — MIT licensed
