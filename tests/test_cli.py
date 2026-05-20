# Filepath: tests/test_cli.py
# Condensed Description: Tests for the voice-duck CLI — roundtrip ducking, inspect mode, and error paths.
# Architecture Layer: Test
# Environment: Local
# Script Hierarchy: test_cli_help_exits_zero, test_cli_basic_roundtrip, test_cli_output_differs_from_input, test_cli_inspect_writes_csv, test_cli_missing_trigger_exits
# Dependencies: Internal: voice_duck.cli / External: pytest, numpy, soundfile
# Exposes: pytest test functions
# Configuration: none
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_duck.cli import main


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, sr)


def test_cli_help_exits_zero() -> None:
    """--help must exit with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_basic_roundtrip(tmp_path: Path, sr: int) -> None:
    """Normal duck mode must write an output file that exists."""
    audio = np.ones((sr, 2), dtype=np.float32) * 0.4
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    target_f = tmp_path / "target.wav"
    trigger_f = tmp_path / "trigger.wav"
    out_f = tmp_path / "out.wav"
    _write_wav(target_f, audio, sr)
    _write_wav(trigger_f, trigger, sr)

    code = main(
        [
            "--target",
            str(target_f),
            "--trigger",
            str(trigger_f),
            "--out",
            str(out_f),
        ]
    )

    assert code == 0
    assert out_f.exists()


def test_cli_output_differs_from_input(tmp_path: Path, sr: int) -> None:
    """The ducked output must not be identical to the input target."""
    audio = np.ones((sr, 2), dtype=np.float32) * 0.4
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    target_f = tmp_path / "target.wav"
    trigger_f = tmp_path / "trigger.wav"
    out_f = tmp_path / "out.wav"
    _write_wav(target_f, audio, sr)
    _write_wav(trigger_f, trigger, sr)

    main(
        [
            "--target",
            str(target_f),
            "--trigger",
            str(trigger_f),
            "--out",
            str(out_f),
        ]
    )

    result, _ = sf.read(str(out_f))
    # The ducked result must be different from the original audio.
    assert not np.allclose(result, audio[: len(result)], atol=0.001)


def test_cli_output_is_quieter_than_input(tmp_path: Path, sr: int) -> None:
    """Ducked output must be quieter than the input target when the trigger is loud."""
    audio = np.ones((sr, 2), dtype=np.float32) * 0.4
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    target_f = tmp_path / "target.wav"
    trigger_f = tmp_path / "trigger.wav"
    out_f = tmp_path / "out.wav"
    _write_wav(target_f, audio, sr)
    _write_wav(trigger_f, trigger, sr)

    main(
        [
            "--target",
            str(target_f),
            "--trigger",
            str(trigger_f),
            "--out",
            str(out_f),
            "--reduction",
            "-20",
        ]
    )

    result, _ = sf.read(str(out_f))
    input_rms = float(np.sqrt(np.mean(audio**2)))
    output_rms = float(np.sqrt(np.mean(result**2)))
    assert output_rms < input_rms


def test_cli_inspect_writes_csv(tmp_path: Path, sr: int) -> None:
    """--inspect mode must write a CSV file with header row plus data rows."""
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    target_f = tmp_path / "target.wav"
    trigger_f = tmp_path / "trigger.wav"
    csv_f = tmp_path / "envelope.csv"
    _write_wav(target_f, np.ones((sr, 2), dtype=np.float32) * 0.4, sr)
    _write_wav(trigger_f, trigger, sr)

    code = main(
        [
            "--target",
            str(target_f),
            "--trigger",
            str(trigger_f),
            "--out",
            str(tmp_path / "out.wav"),
            "--inspect",
            str(csv_f),
        ]
    )

    assert code == 0
    assert csv_f.exists()
    lines = csv_f.read_text().strip().splitlines()
    # Must have a header + at least one data row
    assert len(lines) > 1
    # Header must contain expected column names
    header = lines[0]
    assert "frame" in header
    assert "gain_linear" in header
    assert "gain_db" in header


def test_cli_inspect_csv_row_count_matches_frames(tmp_path: Path, sr: int) -> None:
    """CSV must have exactly sr+1 lines (1 header + 1 per frame) for a 1-second trigger."""
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    trigger_f = tmp_path / "trigger.wav"
    csv_f = tmp_path / "envelope.csv"
    _write_wav(trigger_f, trigger, sr)

    main(
        [
            "--trigger",
            str(trigger_f),
            "--inspect",
            str(csv_f),
        ]
    )

    lines = csv_f.read_text().strip().splitlines()
    # 1 header + sr data rows
    assert len(lines) == sr + 1


def test_cli_missing_trigger_exits(tmp_path: Path, sr: int) -> None:
    """Omitting --trigger must cause a non-zero exit."""
    target_f = tmp_path / "target.wav"
    _write_wav(target_f, np.ones((sr, 2), dtype=np.float32) * 0.4, sr)

    with pytest.raises(SystemExit) as exc_info:
        main(["--target", str(target_f), "--out", str(tmp_path / "out.wav")])

    assert exc_info.value.code != 0


def test_cli_missing_out_returns_error(tmp_path: Path, sr: int) -> None:
    """Omitting --out in normal duck mode must return exit code 1."""
    audio = np.ones((sr, 2), dtype=np.float32) * 0.4
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    target_f = tmp_path / "target.wav"
    trigger_f = tmp_path / "trigger.wav"
    _write_wav(target_f, audio, sr)
    _write_wav(trigger_f, trigger, sr)

    code = main(
        [
            "--target",
            str(target_f),
            "--trigger",
            str(trigger_f),
            # no --out
        ]
    )

    assert code == 1


def test_cli_invalid_reduction_returns_error(tmp_path: Path, sr: int) -> None:
    """A positive reduction_db value must make main() return 1."""
    audio = np.ones((sr, 2), dtype=np.float32) * 0.4
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    target_f = tmp_path / "target.wav"
    trigger_f = tmp_path / "trigger.wav"
    out_f = tmp_path / "out.wav"
    _write_wav(target_f, audio, sr)
    _write_wav(trigger_f, trigger, sr)

    code = main(
        [
            "--target",
            str(target_f),
            "--trigger",
            str(trigger_f),
            "--out",
            str(out_f),
            "--reduction",
            "5",  # positive — invalid
        ]
    )

    assert code == 1


def test_cli_missing_target_file_returns_error(tmp_path: Path, sr: int) -> None:
    """A non-existent target file must make main() return 1."""
    trigger = np.ones(sr, dtype=np.float32) * 0.5
    trigger_f = tmp_path / "trigger.wav"
    out_f = tmp_path / "out.wav"
    _write_wav(trigger_f, trigger, sr)

    code = main(
        [
            "--target",
            str(tmp_path / "does_not_exist.wav"),
            "--trigger",
            str(trigger_f),
            "--out",
            str(out_f),
        ]
    )

    assert code == 1


def test_cli_missing_trigger_file_returns_error(tmp_path: Path, sr: int) -> None:
    """A non-existent trigger file must make main() return 1."""
    audio = np.ones((sr, 2), dtype=np.float32) * 0.4
    target_f = tmp_path / "target.wav"
    out_f = tmp_path / "out.wav"
    _write_wav(target_f, audio, sr)

    code = main(
        [
            "--target",
            str(target_f),
            "--trigger",
            str(tmp_path / "does_not_exist.wav"),
            "--out",
            str(out_f),
        ]
    )

    assert code == 1


def test_cli_inspect_missing_trigger_file_returns_error(tmp_path: Path) -> None:
    """--inspect with a non-existent trigger file must return 1."""
    csv_f = tmp_path / "envelope.csv"

    code = main(
        [
            "--trigger",
            str(tmp_path / "no_such_file.wav"),
            "--inspect",
            str(csv_f),
        ]
    )

    assert code == 1


def test_cli_sample_rate_mismatch_returns_error(tmp_path: Path) -> None:
    """Target and trigger at different sample rates must make main() return 1."""
    audio_48k = np.ones((48000, 2), dtype=np.float32) * 0.4
    trigger_44k = np.ones(44100, dtype=np.float32) * 0.5
    target_f = tmp_path / "target48k.wav"
    trigger_f = tmp_path / "trigger44k.wav"
    out_f = tmp_path / "out.wav"
    sf.write(str(target_f), audio_48k, 48000)
    sf.write(str(trigger_f), trigger_44k, 44100)

    code = main(
        [
            "--target",
            str(target_f),
            "--trigger",
            str(trigger_f),
            "--out",
            str(out_f),
        ]
    )

    assert code == 1
