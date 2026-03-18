#!/usr/bin/env python3
"""
Find closest camwfs dark files in time.

Usage examples:
  python find_closest_camwfs_dark.py /path/to/lab_or_sky_file1.fits /path/to/file2.xrif
  python find_closest_camwfs_dark.py --dt 2025-04-12T00:45:00Z
  python find_closest_camwfs_dark.py --dark-dir /opt/MagAOX/calib/camwfs-dark --dt 20250412004500
"""

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from sparkles.camwfs_dark import DEFAULT_DARK_DIR, find_closest_camwfs_dark


def format_dt_utc(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def iter_targets(files: Iterable[str], dts: Iterable[str]) -> List[str]:
    targets = list(files) + list(dts)
    if not targets:
        raise ValueError("Provide at least one file path or --dt timestamp.")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find closest camwfs dark file(s) in time."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Lab/sky files whose names include __TYYYYmmddHHMMSS...",
    )
    parser.add_argument(
        "--dt",
        action="append",
        default=[],
        help="Explicit datetime (ISO like 2025-04-12T00:45:00Z or compact 20250412004500). Repeatable.",
    )
    parser.add_argument(
        "--dark-dir",
        type=Path,
        default=DEFAULT_DARK_DIR,
        help=f"camwfs dark directory (default: {DEFAULT_DARK_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        targets = iter_targets(args.files, args.dt)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for item in targets:
        try:
            match = find_closest_camwfs_dark(item, dark_dir=args.dark_dir)
        except Exception as exc:
            print(f"{item}\n  ERROR: {exc}")
            continue

        sign = "+" if match.delta_seconds >= 0 else "-"
        print(item)
        print(f"  target_dt_utc : {format_dt_utc(match.target_dt_utc)}")
        print(f"  closest_dark  : {match.dark_path}")
        print(f"  dark_dt_utc   : {format_dt_utc(match.dark_dt_utc)}")
        print(f"  delta_seconds : {sign}{abs(match.delta_seconds):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
