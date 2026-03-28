import shutil
from pathlib import Path

CALIBRATION_FOLDER = "/home/eden/data/spark_calib/"

def copy_metadata_to_savedata(
    calibration_folder: str = CALIBRATION_FOLDER,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """
    Copy metadata.txt -> savedata.txt in each direct subdirectory.

    Returns:
        (n_folders_scanned, n_files_copied, n_skipped)
    """
    root = Path(calibration_folder)
    if not root.exists():
        raise FileNotFoundError(f"Calibration folder does not exist: {calibration_folder}")
    if not root.is_dir():
        raise NotADirectoryError(f"Calibration folder is not a directory: {calibration_folder}")

    n_folders_scanned = 0
    n_files_copied = 0
    n_skipped = 0

    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        n_folders_scanned += 1

        metadata_path = folder / "metadata.txt"
        savedata_path = folder / "savedata.txt"

        if not metadata_path.exists():
            n_skipped += 1
            continue
        if savedata_path.exists() and not overwrite:
            n_skipped += 1
            continue

        shutil.copy2(metadata_path, savedata_path)
        n_files_copied += 1

    return n_folders_scanned, n_files_copied, n_skipped

def main() -> int:
    copy_metadata_to_savedata(
        calibration_folder=CALIBRATION_FOLDER,
        overwrite=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())