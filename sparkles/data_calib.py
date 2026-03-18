# spark_calibratio.py
# 2.27.2026
# this is the first prototype of saving the lab calibration outproducts for the sparkle pipeline 
# goal here is to just get the lab calibration objects saving in an inteligent way
# i think I should use the 

import numpy as np
from astropy.io import fits
import lookyloo
import pathlib
import datetime
import fixr
import re
import sparkles.display as spdisp


glob_data_p = pathlib.Path('/data/rawimages/')
glob_telem_p = pathlib.Path('/srv/aoc/opt/MagAOX')

glob_dir_calib = '/home/eden/data/calib/'
glob_mask = 'aol1_wfsmask.fits'

import sparkles.file_read as fr 
import sparkles.pca as pca 

class SparkCalibrate(object):

    def __init__(self, dir_spark_calib, dark_path, spark_sep, spark_ang, spark_amp, wfs_hz=2000):
        # where to store calibrations
        self.dir_spark_calib  = dir_spark_calib
        # TODO: check that this exists 
        # TODO: Update later to query the database 
        self.spark_sep = spark_sep
        self.spark_ang = spark_ang
        self.spark_amp = spark_amp
        self.wfs_hz = wfs_hz
        # create a subfolder based on sparkle params
        self.calib_folder = f"sep{spark_sep}_ang{spark_ang}_amp{spark_amp}_freq{wfs_hz}"
        self.calib_path = f"{self.dir_spark_calib}/{self.calib_folder}"
        self.check_calib_dir()
        # check if this folder exists, if not, make it
        self.dark_path = dark_path
        self.dark_data = fits.open(dark_path)[0].data
        # TODO: make this more scalable? CACAO probably has a pointer for this
        self.mask_data = fits.open(glob_dir_calib + glob_mask)[0].data
        self.mask_nan = self.mask_data.copy()
        self.mask_nan[self.mask_nan == 0] = np.nan
        # The next steps are scripts, seperated in functions
        # set_data()
        # gen_lab_ref() 
        # save_reference()
        # there is still a lot of reliance of xrif pull methods

    def check_calib_dir(self):
        if pathlib.Path(self.calib_path).exists():
            print(f"Path exists: {self.calib_path}")
        else:
            print(f"making directory for this calibration: {self.calib_path}")
            pathlib.Path(f"{self.calib_path}").mkdir(parents=True, exist_ok=True)
        return 

    def set_data(self, lab_obs, lab_dt, lab_obs_n=0):
        # Obs name, the datetime string we need for lookyloo
        self.lab_obs = lab_obs
        # setting a specific datetime better than setting an n
        self.lab_dt = lab_dt
        self.lab_obs_n = lab_obs_n
        self.lab_obs_span = fr.verify_obs(lab_obs, lab_dt, n=lab_obs_n)
        return

    def gen_lab_ref(self, n_lab = 1024, n_start = 0, klip=3, plt_img=False):
        """ 
        Pull a bunch of files, take an average
        """
        print("GENERATING REFERENCE PCA basis")
        # TODO: Do I want to enforce number of files?
        file_lists = fr.gen_file_list(self.lab_obs_span)
        print(f"Found {len(file_lists)} xrifs, total {len(file_lists)*512} files")
        # LAB+TRUE for file sampling, so using the Lab dark!
        data_ar, timing_ar = self.file_sample_n_clean(file_lists, n = n_lab, n_start = n_start)
        self.n_files = data_ar.shape[0]
        # creating the PCA basis
        Z_KL, Z_KL_img = pca.pca_basis(data_ar, klip=klip)
        self.ref_pca = Z_KL
        self.ref_pca_img = Z_KL_img
        # TODO: save a plot of these images

        # Then, we're going to create a self reference, for each roll. 
        lab_proj = pca.pca_projection(data_ar, self.ref_pca) # this is for all frames
        # shape: [N, klip]

        # averages by the sparkle pattern
        lab_avgs = np.array([np.mean(lab_proj[i::4,:], axis=0) for i in range(4)])
        self.ref_proj = lab_avgs
        # shape: [spark, klip]
        
        # we will usually use these as a stdv across the time axis
        lab_rms = np.std(lab_proj, axis=0)
        self.ref_rms = lab_rms
        # shape: [klip]

        return Z_KL, Z_KL_img, lab_avgs, timing_ar

    def save_reference(self):
        # Using the save function to have all these saved
        self.save_fits(self.ref_pca, f"{self.dir_spark_calib}/{self.calib_folder}/ref_pca.fits")
        self.save_fits(self.ref_proj, f"{self.dir_spark_calib}/{self.calib_folder}/ref_proj.fits")
        self.save_fits(self.ref_rms, f"{self.dir_spark_calib}/{self.calib_folder}/ref_rms.fits")
        return

    def save_fits(self, data, filename):
        # generic fits saving function bc I'm putting all the same info in each header 
        # saving the PCA outputs with relavant info in the headers
        hdu = fits.PrimaryHDU()
        hdu.data = data
        # saving sparkle params (will keep later)
        hdu.header['SPK_sep'] = self.spark_sep
        hdu.header['SPK_ang'] = self.spark_ang
        hdu.header['SPK_amp'] = self.spark_amp
        hdu.header['WFS_hz'] = self.wfs_hz
        # saving observation information (will need to change later)
        hdu.header['LAB_obs'] = self.lab_obs
        hdu.header['LAB_date'] = self.lab_dt.strftime("%Y_%m_%d")
        hdu.header['LAB_time'] = self.lab_dt.strftime("%H:%M:%S")
        hdu.header['n_files'] = self.n_files
        # saving and overwriting if needed
        hdu.writeto(filename, overwrite=True)
        return

    def file_sample_n_clean(self, file_lists, n, n_start=0, norm=True):
        data, timing = fr.pull_n_files(file_lists, n, n_start=n_start)
        data_clean = self.file_clean(data, norm=norm)
        return data_clean, timing

    def file_clean(self, data, norm=True):
        """ Clean a single file 
        recently removed the ref
        TODO: subtract the average?
        """
        # TODO: make this respond to a loaded dark 
        data_dark_sub = data - self.dark_data
        # Normalization standard proc.
        if norm:
            data_dark_sub_mask = data_dark_sub * self.mask_data
            data_normed = np.divide(data_dark_sub_mask, np.sum(data_dark_sub_mask)) # this sum should be one 
            data_return = data_normed
        else: 
            data_return = data
        return data_return

    ############## Plots for convenience ##############
    # TODO: make these their own python file

    def show_pca_basis(self):
        # need to generate lab basis first
        spdisp.plot_pca_basis(self.ref_pca_img, self.mask_nan, self.calib_folder)
        return 

