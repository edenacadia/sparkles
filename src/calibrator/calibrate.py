#!/usr/bin/env python3
"""
Save and/or process sparkle calibration data.
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
    do_save: bool = True,
    do_process: bool = True,
    spark_saver=None,
    process_freq: float | None = 2000.0,
    dual_save: bool = False,
) -> dict[str, object]:
    if not do_save and not do_process:
        raise ValueError("At least one stage must be enabled: save and/or process.")

    if len(sparkle_params) != 3:
        raise ValueError("sparkle_params must be [sep, ang, amp].")

    sep, ang, amp = sparkle_params
    freq_used: float | None = process_freq
    saver = spark_saver

    if do_save:
        if dual_save:
            import spark_save_dual as ss
            saver_cls = ss.SparkSaveDual
        else:
            import spark_save as ss
            saver_cls = ss.SparkSave

        if saver is None:
            saver = saver_cls()
            saver.setup(calibration_folder)
        elif not hasattr(saver, "client"):
            # If caller passes a SparkSave that is not initialized yet.
            saver.setup(calibration_folder)

        saver.setParams(sep, ang, amp)
        saver.take_sparkle_data()
        freq_used = float(saver.freq)

    if do_process:
        if n_cores is not None:
            set_thread_limit(n_cores)

        # Import after setting env vars so thread limits are respected.
        if dual_save:
            import spark_calib_dual as sc
            calib_cls = sc.SparkCalibDual
        else:
            import spark_calib as sc
            calib_cls = sc.SparkCalib

        if freq_used is None:
            raise ValueError(
                "process_freq is required when running process-only mode."
            )

        cspk = calib_cls()
        if not cspk.setup(calibration_folder, sep, ang, amp, freq_used):
            raise RuntimeError(
                "SparkCalib setup failed. Confirm saved data exists for these params."
            )
        cspk.calibrate()

    return {
        "sep": sep,
        "ang": ang,
        "amp": amp,
        "freq": freq_used,
        "saved": do_save,
        "processed": do_process,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save sparkle data and/or process calibration products."
    )
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
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Run only data capture stage.",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Run only processing stage.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=2000.0,
        help="Sparkle frequency used for process-only mode folder lookup (default: 2000).",
    )
    parser.add_argument(
        "--dual-save",
        action="store_true",
        help="Use dual stream save mode (camwfs-sw + aol1_imWFS2-sw).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.save_only and args.process_only:
        parser.error("Use only one of --save-only or --process-only.")

    do_save = not args.process_only
    do_process = not args.save_only

    calibrate(
        sparkle_params=[args.sep, args.ang, args.amp],
        n_cores=args.cores,
        calibration_folder=args.calib_dir,
        do_save=do_save,
        do_process=do_process,
        process_freq=args.freq,
        dual_save=args.dual_save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())