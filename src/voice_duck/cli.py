# Filepath: src/voice_duck/cli.py
# Condensed Description: argparse CLI entry point for voice-duck with duck and inspect modes.
# Architecture Layer: CLI
# Environment: Local
# Script Hierarchy: main
# Dependencies: Internal: voice_duck.ducker, voice_duck.config, voice_duck.io / External: argparse, csv
# Exposes: main
# Configuration: none
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the voice-duck CLI."""
    p = argparse.ArgumentParser(
        prog="voice-duck",
        description="Sidechain auto-ducking for voice-over music beds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  voice-duck --target music.mp3 --trigger voice.wav --out ducked.wav\n"
            "  voice-duck --target bed.wav --trigger narr.wav --reduction -18 --out ep01.wav\n"
            "  voice-duck --target bed.wav --trigger narr.wav --inspect envelope.csv"
        ),
    )
    p.add_argument("--target", metavar="FILE", help="Load the audio file to duck (e.g. music bed).")
    p.add_argument(
        "--trigger",
        metavar="FILE",
        required=True,
        help="Load the audio file that triggers ducking (e.g. voice-over).",
    )
    p.add_argument("--out", metavar="FILE", help="Write the ducked audio to FILE.")
    p.add_argument(
        "--reduction",
        metavar="N",
        type=float,
        default=-12.0,
        help="Set the gain reduction in dB when ducking is active (default: -12.0).",
    )
    p.add_argument(
        "--attack",
        metavar="N",
        type=float,
        default=0.05,
        help="Set the ducking attack time in seconds (default: 0.05).",
    )
    p.add_argument(
        "--release",
        metavar="N",
        type=float,
        default=0.4,
        help="Set the ducking release time in seconds (default: 0.4).",
    )
    p.add_argument(
        "--threshold",
        metavar="N",
        type=float,
        default=-40.0,
        help="Set the trigger threshold in dBFS; quieter than this is silence (default: -40.0).",
    )
    p.add_argument(
        "--lookahead",
        metavar="N",
        type=float,
        default=0.0,
        help="Set pre-roll ducking in seconds before the trigger fires (default: 0.0).",
    )
    p.add_argument(
        "--hold",
        metavar="N",
        type=float,
        default=0.0,
        help="Set minimum hold time in seconds at full duck (default: 0.0).",
    )
    p.add_argument(
        "--inspect",
        metavar="FILE",
        help="Write the gain envelope as CSV to FILE instead of rendering audio.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the voice-duck CLI.

    Two operating modes:

    - Normal: load --target and --trigger, apply ducking, write --out.
    - Inspect: compute the gain envelope only, write a CSV to --inspect.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        0 on success, 1 on any error.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Late import so the module is usable even if pydantic/numpy not installed at
    # import time in other contexts (though in practice they always will be).
    from voice_duck.config import DuckConfig
    from voice_duck.ducker import Ducker
    from voice_duck.io import load, save

    try:
        config = DuckConfig(
            reduction_db=args.reduction,
            attack=args.attack,
            release=args.release,
            threshold_db=args.threshold,
            lookahead=args.lookahead,
            hold=args.hold,
        )
    except Exception as exc:
        print(f"Error: invalid parameter — {exc}", file=sys.stderr)
        return 1

    ducker = Ducker(config=config)

    # ---- Inspect mode ----
    if args.inspect:
        if args.trigger is None:
            print("Error: --trigger is required.", file=sys.stderr)
            return 1

        try:
            trigger_audio, trigger_sr = load(args.trigger)
        except Exception as exc:
            print(f"Error loading trigger: {exc}", file=sys.stderr)
            return 1

        try:
            gain_linear = ducker.envelope(trigger_audio, trigger_sr)
        except Exception as exc:
            print(f"Error computing envelope: {exc}", file=sys.stderr)
            return 1

        import math

        inspect_path = Path(args.inspect).expanduser().resolve()
        try:
            with inspect_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["frame", "gain_linear", "gain_db"])
                for i, g in enumerate(gain_linear):
                    g_float = float(g)
                    gain_db = 20.0 * math.log10(max(g_float, 1e-9))
                    writer.writerow([i, f"{g_float:.8f}", f"{gain_db:.4f}"])
        except OSError as exc:
            print(f"Error writing inspect CSV: {exc}", file=sys.stderr)
            return 1

        logger.info("Envelope written to %s (%d frames)", inspect_path, len(gain_linear))
        return 0

    # ---- Normal duck mode ----
    if args.target is None:
        print("Error: --target is required in normal mode.", file=sys.stderr)
        return 1
    if args.out is None:
        print("Error: --out is required in normal mode.", file=sys.stderr)
        return 1

    try:
        target_audio, target_sr = load(args.target)
    except Exception as exc:
        print(f"Error loading target: {exc}", file=sys.stderr)
        return 1

    try:
        trigger_audio, trigger_sr = load(args.trigger)
    except Exception as exc:
        print(f"Error loading trigger: {exc}", file=sys.stderr)
        return 1

    try:
        ducked = ducker.apply(target_audio, trigger_audio, target_sr)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error during ducking: {exc}", file=sys.stderr)
        return 1

    try:
        out_path = save(args.out, ducked, target_sr)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1

    logger.info("Ducked audio written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
