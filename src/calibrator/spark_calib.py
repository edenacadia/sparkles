import sys
import logging
from enum import Enum
import time
import numpy as np

import xconf
import fixr

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.constants import StateCodes

import purepyindi2 as indi
from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText

import os
import datetime
from zoneinfo import ZoneInfo
import hcipy as hp
from astropy.io import fits
from scipy.optimize import minimize
import pathlib

import sys
from pathlib import Path

from sparkles import pca

# TODO: Need to switch prints to logs

# If this isn't the aol1_imWFS2 streamwriter, than we need to dark subtract
STREAM_WRITER_NAME = 'aol1_imWFS2'
OBS_FOLDER_FORMTAT = "%Y_%m_%d"
MODIFIED_TIME_FORMAT = f"{STREAM_WRITER_NAME}_%Y%m%d%H%M%S%f000.xrif"
FILE_FORMAT = "%Y%m%d%H%M%S%f000"
DATA_PATH = pathlib.Path(f"/opt/MagAOX/rawimages/{STREAM_WRITER_NAME}/")
CHILE_TZ = ZoneInfo("America/Santiago")


class SparkCalib(): 

    def setup(self, dir_spark_calib, sep, ang, amp, freq, n_pca_max=4000):
        #TODO: make a more permanent location
        self.dir_spark_calib  = dir_spark_calib 
        # These are the number of frames wewant to take when starting the program
        self.n_pca_max = n_pca_max
        # these are the sparkle parameters
        self.sep = sep
        self.ang = ang
        self.amp = amp
        self.freq = freq
        # make sure we have a directory for the parameters, 
        # without that we don't have save data to pull
        if not self.check_save_dir():
            return False
        return True

    def calibrate(self):
        """ Main function
        1. Check the sparkle params, update save directory
        2. Grab a stack of frames, with the dark subtracted
        3. Generate the PCA basis, and the self norm
        4. Save the PCA basis and self norm to the save directory
        """
        # Once calibrating, read in the metadata to get 
        self.read_savedata()
        # grab that stack of frames, with the dark subtracted
        data = self.grab_cube()
        # generate the PCA basis, and the self reference
        self.n_pca = np.min([self.n_frames_total, self.n_frames])
        self.gen_lab_ref(data, n_pca=self.n_pca)
        # save the PCA basis and self reference to the save directory
        self.save_reference()
        return

    def check_save_dir(self):
        self.calib_folder = f"sep{int(self.sep):02d}_ang{int(self.ang):02d}_amp{self.amp:01.3f}_freq{int(self.freq):02d}"
        self.calib_path = f"{self.dir_spark_calib}/{self.calib_folder}"
        self.savedata_path = f"{self.calib_path}/savedata.txt"
        if pathlib.Path(self.calib_path).exists():
            print(f"Path exists: {self.calib_path}")
            if not pathlib.Path(self.savedata_path).exists():
                print(f"Savedata file does not exist, EXIT")
                return False
            return True
        else:
            print(f"Directory does not exist, EXIT")
            return False
        
    def read_savedata(self):
        with open(self.savedata_path, "r") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                key, value = line.strip().split(":")
                self.metadata[key] = value
        self.ts_start = datetime.datetime.strptime(self.metadata['ts_start'], FILE_FORMAT)
        self.ts_end = datetime.datetime.strptime(self.metadata['ts_end'], FILE_FORMAT)
        return

    def grab_cube(self):
        # you have to have taken the data first
        xrif_files = self.file_list()
        self.n_files = len(xrif_files)
        # send those xrif files to the pull
        data_conglom = [self.pull_file_xrif(file)[0] for file in xrif_files]
        data_stack = np.vstack(data_conglom)
        self.n_frames_total = data_stack.shape[0]
        # stack into one cube
        return data_stack

    def file_list(self):
        # use the start and end of obs to search for all xrif files
        date_fldr = self.ts_start.strftime(OBS_FOLDER_FORMTAT)
        folder_path = DATA_PATH / date_fldr
        # get all files
        all_matching_files = list(
            sorted(folder_path.glob(f"{STREAM_WRITER_NAME}_*.xrif"))
        )
        print(f"Found {len(all_matching_files)} files in the folder {folder_path}")
        # using the timestamps to make a sort function
        xrif_start = self.ts_start.strftime(MODIFIED_TIME_FORMAT)
        xrif_end = self.ts_end.strftime(MODIFIED_TIME_FORMAT)
        # then make sure 
        span_files = []
        for file_path in all_matching_files:
            if file_path.name > xrif_start:
                if file_path.name < xrif_end:
                    span_files.append(file_path)
        # sort before sending back
        print(f"Found {len(span_files)} files between {xrif_start} and {xrif_end}")
        return sorted(span_files)

    def pull_file_xrif(self, xfile, fsize=512):
        '''
        Take a path, return the data and the timing
        '''
        with open(xfile, 'rb') as fh:
            data = fixr.xrif2numpy(fh)
            timing = fixr.xrif2numpy(fh)
        #TODO: check if this is the right approach
        n_files = int(data.size / 120 / 120)
        return np.reshape(data, (n_files,120,120)), np.reshape(timing, (n_files,5))

    def gen_lab_ref(self, data, n_pca=1000, klip=3):
        """ 
        Pull a bunch of files, take an average
        """
        print("GENERATING REFERENCE PCA basis")
        print(f"Using {n_pca} frames out of {data.shape[0]}")
        # creating the PCA basis
        Z_KL, Z_KL_img = pca.pca_basis(data[:n_pca,:,:], klip=klip)
        self.ref_pca = Z_KL
        self.ref_pca_img = Z_KL_img
        # Then, we're going to create a self reference, for each roll. 
        lab_proj = pca.pca_projection(data, self.ref_pca) # this is for all frames
        # shape: [N, klip]
        # averages by the sparkle pattern
        lab_avgs = np.array([np.mean(lab_proj[i::4,:], axis=0) for i in range(4)])
        self.ref_proj = lab_avgs
        # shape: [spark, klip]
        # we will usually use these as a stdv across the time axis
        lab_rms = np.std(lab_proj, axis=0)
        self.ref_rms = lab_rms
        # shape: [klip]
        return Z_KL, Z_KL_img, lab_avgs
    
    def save_reference(self):
        self.check_calib_dir()
        self.metadata = self._get_save_metadata()
        # Using the save function to have all these saved
        self.save_fits(self.ref_pca, f"{self.calib_path}/ref_pca.fits")
        self.save_fits(self.ref_proj, f"{self.calib_path}/ref_proj.fits")
        self.save_fits(self.ref_rms, f"{self.calib_path}/ref_rms.fits")
        self.save_metadata_txt(f"{self.calib_path}/metadata.txt")
        return

    def _get_save_metadata(self):
        """Build shared metadata used for FITS headers and text saves."""
        return {
            'SPK_sep': self.sep,
            'SPK_ang': self.ang,
            'SPK_amp': self.amp,
            'WFS_hz': self.freq,
            'SPK_dwell': self.dwell,
            'SPK_delay': self.delay,
            'n_files': self.n_files,
            'n_frames_total': self.n_frames_total,
            'n_pca': self.n_pca,
            'ts_start': self.ts_start.strftime(FILE_FORMAT),
            'ts_end': self.ts_end.strftime(FILE_FORMAT),
        }

    def save_metadata_txt(self, filename):
        """Save cube timestamps and calibration parameters to a text file."""
        metadata = self.metadata
        with open(filename, "w") as fh:
            fh.write("# SparkCalib cube save metadata\n")
            for key, value in metadata.items():
                fh.write(f"{key}: {value}\n")
        return

    def save_fits(self, data, filename):
        # generic fits saving function bc I'm putting all the same info in each header 
        # saving the PCA outputs with relavant info in the headers
        hdu = fits.PrimaryHDU()
        hdu.data = data
        metadata = self.metadata
        # saving sparkle params (will keep later)
        hdu.header['SPK_sep'] = metadata['SPK_sep']
        hdu.header['SPK_ang'] = metadata['SPK_ang']
        hdu.header['SPK_amp'] = metadata['SPK_amp']
        hdu.header['WFS_hz'] = metadata['WFS_hz']
        hdu.header['SPK_dwll'] = metadata['SPK_dwell']
        hdu.header['SPK_dlay'] = metadata['SPK_delay']
        hdu.header['n_files'] = metadata['n_files']
        hdu.header['n_frames'] = metadata['n_frames_total']
        hdu.header['n_pca'] = metadata['n_pca']
        hdu.header['ts_start'] = metadata['ts_start']
        hdu.header['ts_end'] = metadata['ts_end']
        # saving and overwriting if needed
        hdu.writeto(filename, overwrite=True)
        return