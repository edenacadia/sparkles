#!/usr/bin/env python3
"""
Calibrate sparkle settings with optional CPU core/thread limits.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence


CALIBRATION_FOLDER = "/home/eden/data/spark_calib/"


def set_thread_limit(n_cores: int) -> None:
    """
    Limit thread usage for common NumPy/SciPy backends.
    Must run before importing numpy/scipy-dependent modules.
    """
    if n_cores < 1:
        raise ValueError("n_cores must be >= 1")

    value = str(n_cores)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value
    os.environ["VECLIB_MAXIMUM_THREADS"] = value
    os.environ["BLIS_NUM_THREADS"] = value


def calibrate(
    sparkle_params: Sequence[float],
    n_cores: int | None = None,
    calibration_folder: str = CALIBRATION_FOLDER,
) -> bool:
    if n_cores is not None:
        set_thread_limit(n_cores)

    # Import after setting env vars so thread limits are respected.
    import spark_calib as sc

    sep, ang, amp = sparkle_params
    cspk = sc.SparkCalib()
    cspk.setup(calibration_folder)
    cspk.setParams(sep, ang, amp)
    cspk.calibrate()
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run spark calibration.")
    parser.add_argument("--sep", type=float, required=True, help="Sparkle separation")
    parser.add_argument("--ang", type=float, required=True, help="Sparkle angle")
    parser.add_argument("--amp", type=float, required=True, help="Sparkle amplitude")
    parser.add_argument(
        "--cores",
        type=int,
        default=5,
        help="Limit CPU cores/threads used by numeric libraries.",
    )
    parser.add_argument(
        "--calib-dir",
        default=CALIBRATION_FOLDER,
        help=f"Calibration output directory (default: {CALIBRATION_FOLDER})",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    calibrate(
        sparkle_params=[args.sep, args.ang, args.amp],
        n_cores=args.cores,
        calibration_folder=args.calib_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())