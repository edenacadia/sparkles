#!/usr/bin/env python3
"""
Batch recalibration for flat-layout sparkle calibration folders.

Expected folder shape under --calib-root:
    <group>/<sepXX_angYY_ampA_freqF>/savedata.txt
or
    <sepXX_angYY_ampA_freqF>/savedata.txt

This mode does not require or use streamwriter subfolders.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CALIBRATION_FOLDER = "/home/eden/data/spark_calib/"


def set_thread_limit(n_cores: int) -> None:
    """Limit thread usage for common NumPy/SciPy backends."""
    if n_cores < 1:
        raise ValueError("n_cores must be >= 1")
    value = str(n_cores)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value
    os.environ["VECLIB_MAXIMUM_THREADS"] = value
    os.environ["BLIS_NUM_THREADS"] = value


def _parse_calib_folder_name(folder_name: str) -> tuple[float, float, float, float]:
    match = re.fullmatch(
        r"sep(?P<sep>\d+)_ang(?P<ang>\d+)_amp(?P<amp>\d+(?:\.\d+)?)_freq(?P<freq>\d+(?:\.\d+)?)",
        folder_name,
    )
    if match is None:
        raise ValueError(f"Unrecognized calibration folder format: {folder_name}")
    return (
        float(match.group("sep")),
        float(match.group("ang")),
        float(match.group("amp")),
        float(match.group("freq")),
    )


def discover_flat_calib_dirs(calib_root: str) -> list[Path]:
    root = Path(calib_root)
    if not root.exists():
        raise FileNotFoundError(f"calib_root does not exist: {calib_root}")

    savedata_files = sorted(root.glob("**/savedata.txt"))
    target_dirs = []
    for savedata in savedata_files:
        calib_dir = savedata.parent
        try:
            _parse_calib_folder_name(calib_dir.name)
        except ValueError:
            continue
        target_dirs.append(calib_dir)
    return sorted(set(target_dirs))


def batch_recalibrate_flat(
    calib_root: str = CALIBRATION_FOLDER,
    n_cores: int | None = None,
    n_pca_max: int = 4000,
) -> dict[str, object]:
    import spark_calib as sc

    if n_cores is not None:
        set_thread_limit(n_cores)

    target_dirs = discover_flat_calib_dirs(calib_root)
    total_targets = len(target_dirs)
    processed = []
    failures = []

    for idx, calib_dir in enumerate(target_dirs, start=1):
        print(f"[{idx}/{total_targets}] Processing folder: {calib_dir}")
        try:
            sep, ang, amp, freq = _parse_calib_folder_name(calib_dir.name)
            calib_parent = str(calib_dir.parent)
            cspk = sc.SparkCalib()
            if not cspk.setup(calib_parent, sep, ang, amp, freq, n_pca_max=n_pca_max):
                raise RuntimeError("SparkCalib setup failed for folder.")
            cspk.calibrate()
            processed.append(str(calib_dir))
            print(f"[{idx}/{total_targets}] Completed folder: {calib_dir}")
        except Exception as exc:  # noqa: BLE001 - keep batch running
            failures.append({"folder": str(calib_dir), "error": str(exc)})
            print(f"[{idx}/{total_targets}] Failed folder: {calib_dir} ({exc})")

    return {
        "root": str(Path(calib_root)),
        "discovered": total_targets,
        "processed": len(processed),
        "failed": len(failures),
        "processed_folders": processed,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch recalibrate flat-layout sparkle calibration folders."
    )
    parser.add_argument(
        "--calib-root",
        default=CALIBRATION_FOLDER,
        help=f"Root folder to scan (default: {CALIBRATION_FOLDER})",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=5,
        help="Limit CPU cores/threads used by numeric libraries.",
    )
    parser.add_argument(
        "--n-pca-max",
        type=int,
        default=4000,
        help="Maximum PCA frames per folder (default: 4000).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = batch_recalibrate_flat(
        calib_root=args.calib_root,
        n_cores=args.cores,
        n_pca_max=args.n_pca_max,
    )
    print(
        "flat batch recalibration complete: "
        f"{result['processed']}/{result['discovered']} processed, "
        f"{result['failed']} failed"
    )
    if result["failures"]:
        print("Failures:")
        for item in result["failures"]:
            print(f" - {item['folder']}: {item['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
