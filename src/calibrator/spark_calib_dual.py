#!/usr/bin/env python3
"""
Process dual-stream sparkle calibration data products.

This follows the same outputs as spark_calib.py, but processes one streamwriter
at a time into per-stream subfolders under the sparkle parameter directory.

For camwfs-sw data, dark subtraction + mask normalization is applied before PCA.
"""

from __future__ import annotations

import datetime
import pathlib
from pathlib import Path

import fixr
import numpy as np
from astropy.io import fits

from sparkles import pca
from sparkles.camwfs_dark import find_closest_camwfs_dark
from sparkles.constants import DEFAULT_CALIB_DIR, GLOB_MASK

OBS_FOLDER_FORMTAT = "%Y_%m_%d"
FILE_FORMAT = "%Y%m%d%H%M%S%f000"
RAW_IMAGE_ROOT = pathlib.Path("/opt/MagAOX/rawimages")


class SparkCalibDual:
    def setup(self, dir_spark_calib, sep, ang, amp, freq, n_pca_max=4000):
        self.dir_spark_calib = dir_spark_calib
        self.n_pca_max = n_pca_max
        self.sep = sep
        self.ang = ang
        self.amp = amp
        self.freq = freq

        self.calib_folder = (
            f"sep{int(self.sep):02d}_ang{int(self.ang):02d}_"
            f"amp{self.amp:01.3f}_freq{int(self.freq):02d}"
        )
        self.calib_path = Path(self.dir_spark_calib) / self.calib_folder
        self.savedata_path = self.calib_path / "savedata.txt"

        if not self.calib_path.exists():
            print(f"Directory does not exist, EXIT: {self.calib_path}")
            return False
        if not self.savedata_path.exists():
            print(f"Savedata file does not exist, EXIT: {self.savedata_path}")
            return False

        self.mask_data = fits.getdata(Path(DEFAULT_CALIB_DIR) / GLOB_MASK)
        self.streamwriters = []
        return True

    def calibrate(self):
        self.read_savedata()
        for streamwriter in self.streamwriters:
            print(f"\nProcessing streamwriter: {streamwriter}")
            data = self.load_streamwriter_data(streamwriter)
            self.n_pca = int(np.min([self.n_frames_total, self.n_pca_max]))
            self.gen_lab_ref(data, n_pca=self.n_pca)
            self.save_reference(streamwriter)
        return

    def load_streamwriter_data(self, streamwriter: str):
        data = self.grab_cube(streamwriter)
        if data.size == 0:
            raise RuntimeError(f"No frames found for streamwriter: {streamwriter}")
        if streamwriter == "camwfs-sw":
            data = self.apply_camwfs_clean(data)
        return data

    def read_savedata(self):
        self.savedata = {}
        with open(self.savedata_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split(":", 1)
                self.savedata[key.strip()] = value.strip()

        self.ts_start = datetime.datetime.strptime(self.savedata["ts_start"], FILE_FORMAT)
        self.ts_end = datetime.datetime.strptime(self.savedata["ts_end"], FILE_FORMAT)
        self.dwell = self.savedata.get("SPK_dwell", 0)
        self.delay = self.savedata.get("SPK_delay", 0)

        stream_keys = sorted(k for k in self.savedata if k.startswith("stream_"))
        streamwriters = [self.savedata[k] for k in stream_keys if self.savedata[k]]

        if not streamwriters and "streamwriters" in self.savedata:
            streamwriters = [
                token.strip() for token in self.savedata["streamwriters"].split(",") if token.strip()
            ]

        if not streamwriters:
            # Backward-compatible default (single-stream behavior).
            streamwriters = ["aol1_imWFS2-sw"]

        self.streamwriters = streamwriters
        return

    def _stream_device_name(self, streamwriter: str) -> str:
        return streamwriter[:-3] if streamwriter.endswith("-sw") else streamwriter

    def file_list(self, streamwriter: str):
        device_name = self._stream_device_name(streamwriter)
        date_folders = {
            self.ts_start.strftime(OBS_FOLDER_FORMTAT),
            self.ts_end.strftime(OBS_FOLDER_FORMTAT),
        }
        all_matching_files = []
        for date_fldr in sorted(date_folders):
            folder_path = RAW_IMAGE_ROOT / device_name / date_fldr
            files_here = sorted(folder_path.glob(f"{device_name}_*.xrif"))
            print(f"Found {len(files_here)} files in {folder_path}")
            all_matching_files.extend(files_here)

        xrif_start = f"{device_name}_{self.ts_start.strftime(FILE_FORMAT)}.xrif"
        xrif_end = f"{device_name}_{self.ts_end.strftime(FILE_FORMAT)}.xrif"

        span_files = [fp for fp in all_matching_files if xrif_start < fp.name < xrif_end]
        if span_files:
            print(f"Found {len(span_files)} files between {xrif_start} and {xrif_end}")
            return span_files

        # Fallback parser path for any non-standard filename edge cases.
        parse_failures = 0
        parsed_span = []
        for file_path in all_matching_files:
            ts_token = file_path.name.replace(f"{device_name}_", "").replace(".xrif", "")
            try:
                ts = datetime.datetime.strptime(ts_token, FILE_FORMAT)
            except ValueError:
                parse_failures += 1
                continue
            if self.ts_start < ts < self.ts_end:
                parsed_span.append(file_path)

        print(
            f"Found {len(parsed_span)} files in parsed fallback "
            f"(parse failures: {parse_failures}) between {self.ts_start} and {self.ts_end}"
        )
        return parsed_span

    def pull_file_xrif(self, xfile):
        with open(xfile, "rb") as fh:
            data = fixr.xrif2numpy(fh)
            timing = fixr.xrif2numpy(fh)
        n_files = int(data.size / 120 / 120)
        return np.reshape(data, (n_files, 120, 120)), np.reshape(timing, (n_files, 5))

    def grab_cube(self, streamwriter: str):
        xrif_files = self.file_list(streamwriter)
        self.n_files = len(xrif_files)
        data_conglom = [self.pull_file_xrif(file)[0] for file in xrif_files]
        if not data_conglom:
            self.n_frames_total = 0
            return np.array([])
        data_stack = np.vstack(data_conglom)
        self.n_frames_total = data_stack.shape[0]
        return data_stack

    def apply_camwfs_clean(self, data):
        dark_match = find_closest_camwfs_dark(self.ts_start)
        self.camwfs_dark_path = str(dark_match.dark_path)
        dark_data = fits.getdata(dark_match.dark_path)
        # dark subtraction
        data_dark_sub = data - dark_data
        # subtract off the mean per frame, over mask
        # doing that here made things weird, had more scatter
        # norm in the mask
        data_sub_mask = data_dark_sub * self.mask_data
        frame_sums = np.sum(data_sub_mask, axis=(1, 2), keepdims=True)
        frame_sums = np.where(frame_sums == 0, 1.0, frame_sums)
        normed_data = data_sub_mask / frame_sums
        # now subtract the mean frame
        data_mean_sub = normed_data - np.mean(normed_data, axis=0, keepdims=True)
        
        return data_mean_sub

    def gen_lab_ref(self, data, n_pca=1000, klip=3):
        print("GENERATING REFERENCE PCA basis")
        print(f"Using {n_pca} frames out of {data.shape[0]}")
        Z_KL, Z_KL_img = pca.pca_basis(data[:n_pca, :, :], klip=klip)
        self.ref_pca = Z_KL
        self.ref_pca_img = Z_KL_img

        lab_proj = pca.pca_projection(data, self.ref_pca)
        self.ref_proj = np.array([np.mean(lab_proj[i::4, :], axis=0) for i in range(4)])
        self.ref_rms = np.std(lab_proj, axis=0)
        return

    def _stream_output_dir(self, streamwriter: str) -> Path:
        out_dir = self.calib_path / streamwriter
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _get_save_metadata(self, streamwriter: str):
        metadata = {
            "SPK_sep": self.sep,
            "SPK_ang": self.ang,
            "SPK_amp": self.amp,
            "WFS_hz": self.freq,
            "SPK_dwell": self.dwell,
            "SPK_delay": self.delay,
            "n_files": self.n_files,
            "n_frames_total": self.n_frames_total,
            "n_pca": self.n_pca,
            "ts_start": self.ts_start.strftime(FILE_FORMAT),
            "ts_end": self.ts_end.strftime(FILE_FORMAT),
            "streamwriter": streamwriter,
        }
        if streamwriter == "camwfs-sw":
            metadata["camwfs_dark"] = getattr(self, "camwfs_dark_path", "")
        return metadata

    def save_reference(self, streamwriter: str):
        out_dir = self._stream_output_dir(streamwriter)
        self.metadata = self._get_save_metadata(streamwriter)
        self.save_fits(self.ref_pca, out_dir / "ref_pca.fits")
        self.save_fits(self.ref_proj, out_dir / "ref_proj.fits")
        self.save_fits(self.ref_rms, out_dir / "ref_rms.fits")
        self.save_metadata_txt(out_dir / "metadata.txt")
        return

    def save_metadata_txt(self, filename: Path):
        with open(filename, "w") as fh:
            fh.write("# SparkCalib dual stream metadata\n")
            for key, value in self.metadata.items():
                fh.write(f"{key}: {value}\n")
        return

    def save_fits(self, data, filename: Path):
        hdu = fits.PrimaryHDU()
        hdu.data = data
        metadata = self.metadata
        hdu.header["SPK_sep"] = metadata["SPK_sep"]
        hdu.header["SPK_ang"] = metadata["SPK_ang"]
        hdu.header["SPK_amp"] = metadata["SPK_amp"]
        hdu.header["WFS_hz"] = metadata["WFS_hz"]
        hdu.header["SPK_dwll"] = metadata["SPK_dwell"]
        hdu.header["SPK_dlay"] = metadata["SPK_delay"]
        hdu.header["n_files"] = metadata["n_files"]
        hdu.header["n_frames"] = metadata["n_frames_total"]
        hdu.header["n_pca"] = metadata["n_pca"]
        hdu.header["ts_start"] = metadata["ts_start"]
        hdu.header["ts_end"] = metadata["ts_end"]
        hdu.header["streamwr"] = metadata["streamwriter"]
        if "camwfs_dark" in metadata and metadata["camwfs_dark"]:
            hdu.header["cmwfsdrk"] = metadata["camwfs_dark"]
        hdu.writeto(filename, overwrite=True)
        return

