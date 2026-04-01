#!/usr/bin/env python3
"""
Save and/or process sparkle calibration data.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Sequence

CALIBRATION_FOLDER = "/home/eden/data/spark_calib/"
SPARK_AO_FOLDER = "/home/eden/data/spark_AO"

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


def recalibrate_all_camwfs(
    spark_ao_folder: str = SPARK_AO_FOLDER,
    n_cores: int | None = None,
    n_pca_max: int = 4000,
) -> dict[str, object]:
    """
    Re-run processing for all camwfs-sw calibration folders under spark_ao_folder.
    """
    if n_cores is not None:
        set_thread_limit(n_cores)

    return recalibrate_all_streamwriter(
        streamwriter="camwfs-sw",
        spark_ao_folder=spark_ao_folder,
        n_cores=n_cores,
        n_pca_max=n_pca_max,
    )


def recalibrate_all_streamwriter(
    streamwriter: str,
    spark_ao_folder: str = SPARK_AO_FOLDER,
    n_cores: int | None = None,
    n_pca_max: int = 4000,
) -> dict[str, object]:
    """
    Re-run calibration processing for one streamwriter across all folders under spark_ao_folder.
    """
    if not streamwriter:
        raise ValueError("streamwriter must be a non-empty string.")
    if n_cores is not None:
        set_thread_limit(n_cores)

    import spark_calib_dual as sc

    root = Path(spark_ao_folder)
    if not root.exists():
        raise FileNotFoundError(f"spark_ao_folder does not exist: {spark_ao_folder}")

    savedata_files = sorted(root.glob("**/savedata.txt"))
    target_calib_dirs = [
        p.parent for p in savedata_files if (p.parent / streamwriter).exists()
    ]
    target_calib_dirs = sorted(set(target_calib_dirs))
    total_targets = len(target_calib_dirs)

    processed = []
    failures = []

    for idx, calib_dir in enumerate(target_calib_dirs, start=1):
        print(f"[{idx}/{total_targets}] Processing folder: {calib_dir}")
        try:
            sep, ang, amp, freq = _parse_calib_folder_name(calib_dir.name)
            calib_parent = str(calib_dir.parent)
            cspk = sc.SparkCalibDual()
            if not cspk.setup(calib_parent, sep, ang, amp, freq, n_pca_max=n_pca_max):
                raise RuntimeError("SparkCalibDual setup failed for folder.")
            cspk.calibrate(streamwriters=[streamwriter])
            processed.append(str(calib_dir))
            print(f"[{idx}/{total_targets}] Completed folder: {calib_dir}")
        except Exception as exc:  # noqa: BLE001 - keep batch running
            failures.append({"folder": str(calib_dir), "error": str(exc)})
            print(f"[{idx}/{total_targets}] Failed folder: {calib_dir} ({exc})")

    return {
        "root": str(root),
        "streamwriter": streamwriter,
        "discovered": len(target_calib_dirs),
        "processed": len(processed),
        "failed": len(failures),
        "processed_folders": processed,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save sparkle data and/or process calibration products."
    )
    parser.add_argument("--sep", type=float, help="Sparkle separation")
    parser.add_argument("--ang", type=float, help="Sparkle angle")
    parser.add_argument("--amp", type=float, help="Sparkle amplitude")
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
    parser.add_argument(
        "--recalibrate-camwfs-all",
        action="store_true",
        help=(
            "Re-run calibration processing for all camwfs-sw folders found under "
            "--spark-ao-dir."
        ),
    )
    parser.add_argument(
        "--recalibrate-streamwriter-all",
        action="store_true",
        help=(
            "Re-run calibration processing for one streamwriter in every folder under "
            "--spark-ao-dir that contains that streamwriter subfolder."
        ),
    )
    parser.add_argument(
        "--streamwriter",
        default="camwfs-sw",
        help=(
            "Streamwriter subfolder name used by --recalibrate-streamwriter-all "
            "(default: camwfs-sw)."
        ),
    )
    parser.add_argument(
        "--spark-ao-dir",
        default=SPARK_AO_FOLDER,
        help=f"Root folder to scan in batch camwfs mode (default: {SPARK_AO_FOLDER})",
    )
    parser.add_argument(
        "--n-pca-max",
        type=int,
        default=4000,
        help="Maximum PCA frames for batch camwfs mode (default: 4000).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.save_only and args.process_only:
        parser.error("Use only one of --save-only or --process-only.")

    if args.recalibrate_camwfs_all and args.recalibrate_streamwriter_all:
        parser.error("Use only one of --recalibrate-camwfs-all or --recalibrate-streamwriter-all.")

    if args.recalibrate_camwfs_all:
        result = recalibrate_all_camwfs(
            spark_ao_folder=args.spark_ao_dir,
            n_cores=args.cores,
            n_pca_max=args.n_pca_max,
        )
        print(
            f"camwfs batch recalibration complete: "
            f"{result['processed']}/{result['discovered']} processed, "
            f"{result['failed']} failed"
        )
        if result["failures"]:
            print("Failures:")
            for item in result["failures"]:
                print(f" - {item['folder']}: {item['error']}")
        return 0

    if args.recalibrate_streamwriter_all:
        result = recalibrate_all_streamwriter(
            streamwriter=args.streamwriter,
            spark_ao_folder=args.spark_ao_dir,
            n_cores=args.cores,
            n_pca_max=args.n_pca_max,
        )
        print(
            f"{result['streamwriter']} batch recalibration complete: "
            f"{result['processed']}/{result['discovered']} processed, "
            f"{result['failed']} failed"
        )
        if result["failures"]:
            print("Failures:")
            for item in result["failures"]:
                print(f" - {item['folder']}: {item['error']}")
        return 0

    missing = [name for name in ("sep", "ang", "amp") if getattr(args, name) is None]
    if missing:
        parser.error(
            "Missing required arguments for single-run mode: "
            + ", ".join(f"--{name}" for name in missing)
        )

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