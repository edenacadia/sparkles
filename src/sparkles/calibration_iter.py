from astropy.io import fits
import numpy as np
import pandas as pd
import pathlib
import re
import warnings


_CALIB_DIR_PATTERN = re.compile(
    r"^sep(?P<sep>\d+)_ang(?P<ang>\d+)_amp(?P<amp>\d+(?:\.\d+)?)_freq(?P<freq>\d+(?:\.\d+)?)$"
)


def parse_calibration_folder_name(folder_name):
    """
    Parse sparkle parameters from a calibration folder name.
    Expected format: sep##_ang##_amp#.#_freq####
    """
    match = _CALIB_DIR_PATTERN.fullmatch(folder_name)
    if match is None:
        return None
    return {
        "spark_sep": float(match.group("sep")),
        "spark_ang": float(match.group("ang")),
        "spark_amp": float(match.group("amp")),
        "wfs_hz": float(match.group("freq")),
    }


def calibration_rms_df(
    calib_directory,
    rms_filename="ref_rms.fits",
    pca_filename="ref_pca.fits",
    warn_missing=True,
):
    """
    Scan a calibration root directory and return RMS per PCA mode as a DataFrame.

    Each row corresponds to a single PCA mode from a single calibration folder.
    Output columns:
      - calib_folder
      - calib_path
      - spark_sep
      - spark_ang
      - spark_amp
      - wfs_hz
      - pca_mode
      - rms
    """

    calib_root = pathlib.Path(calib_directory).expanduser()
    if not calib_root.exists():
        raise FileNotFoundError(f"Calibration directory does not exist: {calib_root}")
    if not calib_root.is_dir():
        raise NotADirectoryError(f"Expected a directory path, got: {calib_root}")

    rows = []
    for calib_path in sorted(p for p in calib_root.iterdir() if p.is_dir()):
        params = parse_calibration_folder_name(calib_path.name)
        if params is None:
            continue

        try:
            # getting the RMS values
            rms_vals = get_rms(calib_path, rms_filename=rms_filename)
            # calc the pca image stats
            pca_vals = get_pca_rms(calib_path, pca_filename=pca_filename)
        except FileNotFoundError as exc:
            if warn_missing:
                warnings.warn(f"Skipping {calib_path.name}: {exc}", stacklevel=2)
            continue

        if len(rms_vals) != len(pca_vals):
            n_modes = min(len(rms_vals), len(pca_vals))
            if warn_missing:
                warnings.warn(
                    (
                        f"Length mismatch in {calib_path.name}: "
                        f"{len(rms_vals)} RMS modes vs {len(pca_vals)} PCA modes. "
                        f"Using first {n_modes} modes."
                    ),
                    stacklevel=2,
                )
        else:
            n_modes = len(rms_vals)


        for mode_idx in range(n_modes):
            rows.append(
                {
                    "calib_folder": calib_path.name,
                    "calib_path": str(calib_path),
                    "spark_sep": params["spark_sep"],
                    "spark_ang": params["spark_ang"],
                    "spark_amp": params["spark_amp"],
                    "wfs_hz": params["wfs_hz"],
                    "pca_mode": int(mode_idx),
                    "rms": float(rms_vals[mode_idx]),
                    "pca_rms": float(pca_vals[mode_idx]),
                }
            )

    columns = [
        "calib_folder",
        "calib_path",
        "spark_sep",
        "spark_ang",
        "spark_amp",
        "wfs_hz",
        "pca_mode",
        "rms",
        "pca_rms"
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(
            ["spark_sep", "spark_ang", "spark_amp", "wfs_hz", "pca_mode"]
        ).reset_index(drop=True)
    return df

def get_rms(calib_path, rms_filename="ref_rms.fits"):
    rms_path = calib_path / rms_filename
    if not rms_path.exists():
        raise FileNotFoundError(f"missing {rms_filename}")

    with fits.open(rms_path) as hdul:
        rms_data = np.asarray(hdul[0].data)
    rms_vals = np.ravel(rms_data).astype(float)

    return rms_vals

def get_pca_rms(calib_path, pca_filename="ref_pca.fits"):
    pca_path = calib_path / pca_filename
    if not pca_path.exists():
        raise FileNotFoundError(f"missing {pca_filename}")

    with fits.open(pca_path) as hdul:
        pca_data = np.asarray(hdul[0].data)
    pca_data = pca_data.T
    pca_vals = []

    # iter through the pca images, calc the RMS
    for image in pca_data:
        pca_vals.append(np.std(image))

    return pca_vals


def _coerce_basis_matrix(basis):
    """
    Ensure PCA/modal basis is shaped (n_pixels, n_modes).
    """
    arr = np.asarray(basis, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D basis array, got shape {arr.shape}")
    if arr.shape[0] < arr.shape[1]:
        # Common alternate convention is (n_modes, n_pixels).
        arr = arr.T
    return arr


def project_all_calibrations_to_modal_basis(
    calib_directory,
    modal_basis,
    pca_filename="ref_pca.fits",
    warn_missing=True,
):
    """
    Iterate over all sparkle calibration folders and project each stored PCA basis
    onto a provided modal basis.

    Returns
    -------
    projections : list[np.ndarray]
        One entry per calibration folder, in sorted folder order. Each entry has
        shape (n_modal_modes, n_calibration_modes).
    params_by_index : dict[int, dict]
        Metadata keyed by list index with sparkle parameters and folder info.
    """
    calib_root = pathlib.Path(calib_directory).expanduser()
    if not calib_root.exists():
        raise FileNotFoundError(f"Calibration directory does not exist: {calib_root}")
    if not calib_root.is_dir():
        raise NotADirectoryError(f"Expected a directory path, got: {calib_root}")

    modal = _coerce_basis_matrix(modal_basis)
    n_modal_pixels = modal.shape[0]

    projections = []
    params_by_index = {}

    for calib_path in sorted(p for p in calib_root.iterdir() if p.is_dir()):
        params = parse_calibration_folder_name(calib_path.name)
        if params is None:
            continue

        pca_path = calib_path / pca_filename
        if not pca_path.exists():
            if warn_missing:
                warnings.warn(f"Skipping {calib_path.name}: missing {pca_filename}", stacklevel=2)
            continue

        with fits.open(pca_path) as hdul:
            calib_basis = _coerce_basis_matrix(hdul[0].data)

        if calib_basis.shape[0] != n_modal_pixels:
            if warn_missing:
                warnings.warn(
                    (
                        f"Skipping {calib_path.name}: pixel-size mismatch "
                        f"(calib={calib_basis.shape[0]}, modal={n_modal_pixels})"
                    ),
                    stacklevel=2,
                )
            continue

        # Project calibration basis modes onto the modal basis.
        # modal.T: (n_modal_modes, n_pixels)
        # calib_basis: (n_pixels, n_calibration_modes)
        projection = modal.T @ calib_basis

        idx = len(projections)
        projections.append(projection)
        params_by_index[idx] = {
            "calib_folder": calib_path.name,
            "calib_path": str(calib_path),
            "spark_sep": params["spark_sep"],
            "spark_ang": params["spark_ang"],
            "spark_amp": params["spark_amp"],
            "wfs_hz": params["wfs_hz"],
            "n_modal_modes": int(modal.shape[1]),
            "n_calibration_modes": int(calib_basis.shape[1]),
        }

    return projections, params_by_index


def sort_projection_params(
    projection_list,
    param_map,
    sep_order=None,
    ang_order=None,
    amp_order=None,
):
    """
    Reorder projection/metadata pairs by explicit sep/ang/amp order.

    Parameters
    ----------
    projection_list : sequence
        Projections aligned to `param_map` indices.
    param_map : dict[int, dict]
        Index -> metadata dict with keys `spark_sep`, `spark_ang`, `spark_amp`.
    sep_order, ang_order, amp_order : sequence[float] | None
        Desired ordering values for each parameter. If None, the parameter is
        sorted ascending numerically.

    Returns
    -------
    sorted_projection_list : list
        Reordered projections.
    sorted_param_map : dict[int, dict]
        Reindexed metadata map where keys align with sorted_projection_list.
    """
    if projection_list is None:
        raise ValueError("projection_list cannot be None.")
    if param_map is None:
        raise ValueError("param_map cannot be None.")

    n = len(projection_list)
    missing = [i for i in range(n) if i not in param_map]
    if missing:
        raise ValueError(f"param_map is missing indices for projection_list: {missing[:10]}")

    def _ranker(order_values):
        if order_values is None:
            return None
        order_map = {float(v): i for i, v in enumerate(order_values)}

        def _rank(v):
            fv = float(v)
            # Unknown values go to the end, preserving numeric tie-break.
            return (order_map.get(fv, len(order_map)), fv)

        return _rank

    sep_rank = _ranker(sep_order)
    ang_rank = _ranker(ang_order)
    amp_rank = _ranker(amp_order)

    def _key(i):
        meta = param_map[i]
        sep = float(meta["spark_sep"])
        ang = float(meta["spark_ang"])
        amp = float(meta["spark_amp"])
        sep_key = sep_rank(sep) if sep_rank is not None else (0, sep)
        ang_key = ang_rank(ang) if ang_rank is not None else (0, ang)
        amp_key = amp_rank(amp) if amp_rank is not None else (0, amp)
        return (sep_key, ang_key, amp_key, i)

    sorted_old_indices = sorted(range(n), key=_key)

    sorted_projection_list = [projection_list[i] for i in sorted_old_indices]
    sorted_param_map = {
        new_i: dict(param_map[old_i]) for new_i, old_i in enumerate(sorted_old_indices)
    }
    for new_i, old_i in enumerate(sorted_old_indices):
        sorted_param_map[new_i]["old_idx"] = int(old_i)

    return sorted_projection_list, sorted_param_map

