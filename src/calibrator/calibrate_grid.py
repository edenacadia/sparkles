#!/usr/bin/env python3
"""
Run spark calibration across a fixed parameter grid.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import List, Sequence, Tuple

from calibrate import calibrate


DEFAULT_ANGLES = [0, 15, 30, 45, 60, 75, 90]
DEFAULT_SEPARATIONS = [10, 14, 18, 22]
DEFAULT_AMPLITUDES = [0.02, 0.03, 0.04]


def build_grid(
    separations: Sequence[float],
    angles: Sequence[float],
    amplitudes: Sequence[float],
) -> List[Tuple[float, float, float]]:
    return [(sep, ang, amp) for sep, ang, amp in product(separations, angles, amplitudes)]


def calib_folder_name(sep: float, ang: float, amp: float, freq: float) -> str:
    return f"sep{int(sep):02d}_ang{int(ang):02d}_amp{amp:01.3f}_freq{int(freq):02d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterate through sparkle parameter combinations and save/process."
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Limit CPU cores/threads for processing runs.",
    )
    parser.add_argument(
        "--calib-dir",
        default="/home/eden/data/spark_calib/",
        help="Calibration output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all combinations without executing runs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one run fails (default: continue).",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=2000.0,
        help="Frequency used for folder naming/lookups (default: 2000).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Run even if outputs for that combination already exist.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Run only the data-saving stage for each combo.",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Run only the processing stage for each combo.",
    )
    parser.add_argument(
        "--dual-save",
        action="store_true",
        help="Use dual stream save mode (camwfs-sw + aol1_imWFS2-sw).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.save_only and args.process_only:
        raise SystemExit("Use only one of --save-only or --process-only.")

    do_save = not args.process_only
    do_process = not args.save_only

    combos = build_grid(
        separations=DEFAULT_SEPARATIONS,
        angles=DEFAULT_ANGLES,
        amplitudes=DEFAULT_AMPLITUDES,
    )
    total = len(combos)
    print(f"Total calibration runs: {total}")

    calib_root = Path(args.calib_dir)

    if args.dry_run:
        for i, (sep, ang, amp) in enumerate(combos, start=1):
            folder = calib_root / calib_folder_name(sep, ang, amp, args.freq)
            exists = folder.exists()
            status = "SKIP (exists)" if exists and not args.force_rerun else "RUN"
            print(f"[{i:03d}/{total:03d}] {status} sep={sep}, ang={ang}, amp={amp}")
        return 0

    failures = []
    skipped = []

    spark_saver = None
    if do_save and not args.dry_run:
        if args.dual_save:
            import spark_save_dual as ss
            saver_cls = ss.SparkSaveDual
        else:
            import spark_save as ss
            saver_cls = ss.SparkSave

        spark_saver = saver_cls()
        spark_saver.setup(args.calib_dir)

    for i, (sep, ang, amp) in enumerate(combos, start=1):
        folder = calib_root / calib_folder_name(sep, ang, amp, args.freq)
        if not args.force_rerun:
            savedata_exists = (folder / "savedata.txt").exists()
            metadata_exists = (folder / "metadata.txt").exists()

            if do_save and do_process and savedata_exists and metadata_exists:
                print(
                    f"\n[{i:03d}/{total:03d}] Skipping existing save+process outputs: {folder}"
                )
                skipped.append((sep, ang, amp, str(folder)))
                continue
            if do_save and not do_process and savedata_exists:
                print(f"\n[{i:03d}/{total:03d}] Skipping existing saved run: {folder}")
                skipped.append((sep, ang, amp, str(folder)))
                continue
            if do_process and not do_save and metadata_exists:
                print(
                    f"\n[{i:03d}/{total:03d}] Skipping existing calibration run: {folder}"
                )
                skipped.append((sep, ang, amp, str(folder)))
                continue

        print(f"\n[{i:03d}/{total:03d}] Running sep={sep}, ang={ang}, amp={amp}")
        try:
            calibrate(
                sparkle_params=[sep, ang, amp],
                n_cores=args.cores,
                calibration_folder=args.calib_dir,
                do_save=do_save,
                do_process=do_process,
                spark_saver=spark_saver,
                process_freq=args.freq,
                dual_save=args.dual_save,
            )
            print(f"[{i:03d}/{total:03d}] OK")
        except Exception as exc:
            print(f"[{i:03d}/{total:03d}] FAILED: {exc}")
            failures.append((sep, ang, amp, str(exc)))
            if args.stop_on_error:
                break

    print("\nBatch complete.")
    print(f"Successful: {total - len(failures) - len(skipped)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failures)}")
    if skipped:
        print("Skipped details:")
        for sep, ang, amp, folder in skipped:
            print(f"  sep={sep}, ang={ang}, amp={amp} -> {folder}")
    if failures:
        print("Failure details:")
        for sep, ang, amp, err in failures:
            print(f"  sep={sep}, ang={ang}, amp={amp} -> {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
