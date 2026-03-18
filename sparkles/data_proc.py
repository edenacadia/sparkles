
# spark_xrif_pca.py
# Eden McEwen 
# January 28th 2026
# this is remade to deal with sparkles in a new way 
# making sparkles compatible with xrif functions
# this also does some optimization 

from astropy.io import fits
from matplotlib import pyplot as plt
from multiprocessing import Pool
from functools import partial
from scipy import signal
from math import ceil
import matplotlib.pylab as pl
import multiprocessing as mp
import numpy as np
import pathlib
# Special joseph additions
from lookyloo.core import get_matching_paths

# sparkle package files
import sparkles.file_read as fr 
import sparkles.pca as pca 

glob_dir_calib = '/home/eden/data/calib/'
glob_dark = 'camwfs-dark_bin2_2000.000000_600.000000_-45.000000__T20220422005142597501326.fits'
glob_mask = 'aol1_wfsmask.fits'
glob_ref = 'aol1_wfsref.fits'
# to be sent to lookyloo scrips
glob_data_p = pathlib.Path('/data/rawimages/')
glob_telem_p = pathlib.Path('/srv/aoc/opt/MagAOX')

class SparkXrif(object):
    Hz = 2000
    n_pool = 4

    def __init__(self, obs_string, target_name, dir_calib, dir_spark_calib, dark_path,
                 spark_sep, spark_ang, spark_amp, wfs_hz=2000, n_avg=2000):
        # set up the calibration file:
        self.dir_calib  = dir_calib
        self.dir_spark_calib = dir_spark_calib
        # TODO: Update later to query the database 
        self.spark_sep = spark_sep
        self.spark_ang = spark_ang
        self.spark_amp = spark_amp
        self.wfs_hz = wfs_hz
        # Define the calib directories, then look to see if it exits:
        self.calib_folder = f"sep{spark_sep}_ang{spark_ang}_amp{spark_amp}_freq{wfs_hz}"
        self.calib_path = f"{self.dir_spark_calib}/{self.calib_folder}"
        self.ref_pca, self.ref_rms = self.load_calib(self.calib_path)
        # TODO: one day, automate grabbing the right dark 
        self.dark_path = dark_path
        self.dark_data = fits.open(dark_path)[0].data
        # TODO: Make the mask load the cacao mask path
        self.mask_data = fits.open(glob_dir_calib + glob_mask)[0].data
        self.mask_nan = self.mask_data.copy()
        self.mask_nan[self.mask_nan == 0] = np.nan
        # Value for the RMS rolling stdv
        self.n_avg = n_avg
        # Setting up data here bc I'm lazy
        self.obs_string = obs_string
        self.target_name = target_name
        sky_dt, sky_obs = fr.dir_to_lookyloo(obs_string, target_name)
        self.set_data(sky_obs, sky_dt)
        return 

    def load_calib(self, calib_path):
        print(f"Calib folder: {calib_path}")
        path_pca = f"{calib_path}/ref_pca.fits"
        path_rms = f"{calib_path}/ref_rms.fits"
        # check if the calibration directory exits
        if pathlib.Path(calib_path).exists():
            data_pca = fits.open(path_pca)[0].data
            data_rms = fits.open(path_rms)[0].data
            print("Calibration files found, loading PCA basis and RMS")
            return data_pca, data_rms
        else: 
            print("Calibration files not found, GENERATE BEFORE CONTINUING")
            return None, None

    def set_data(self, sky_obs_name, sky_dt_start, sky_obs_n=0):
        """
        Changing the behavior so that 
        """
        self.sky_obs_name = sky_obs_name
        self.sky_dt_start = sky_dt_start
        self.sky_obs_n = sky_obs_n
        self.sky_obs_span = fr.verify_obs(sky_obs_name, sky_dt_start, n=sky_obs_n) 
        self.check_datasets(self.sky_obs_span)
        return
    
    def check_datasets(self, obs_span, fsize=512):
        file_list = fr.gen_file_list(obs_span)
        print("Number of files: ", len(file_list)*fsize)
        return
    
    ################### file sampling ###################

    def file_sample_n_clean(self, file_lists, n, n_start=0, norm=True, lab=False):
        data, timing = fr.pull_n_files(file_lists, n, n_start=n_start)
        data_clean = self.file_clean(data, norm=norm, lab=lab)
        return data_clean, timing

    def xfile_sample_clean(self, p, norm=True, lab=False):
        # collecting many xrif files
        data, t = fr.pull_file_xrif(p) # this is gonna be 512
        # cleaning those files with lab data
        data_clean = self.file_clean(data, lab=lab)
        return data_clean, t
    
    def file_clean(self, data, norm=True, lab=False):
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
            data_return = data_dark_sub
        return data_return

    ################### process data ###################

    def proj_xfile(self, p, norm=True, lab=False):
        """
        Docstring for proj_xfile
        """
        # pull data. lab indicates what dark to use
        data_arr, t = self.xfile_sample_clean(p, norm=norm, lab=lab)
        # now each frame should be projected onto the PCA basis
        data_proj = pca.pca_projection(data_arr, self.ref_pca) 
        # first axis file number, second axis PCA return
        return data_proj

    def proj_pool(self, n_start=0, n=100, n_workers=4):
        """
        Uses the list of files the object was generated with. 
        """
        f_list = fr.gen_file_list(self.sky_obs_span) # number of xrif files
        n_end = n_start + n
        # changing so these 
        n_start_x, n_end_x = fr.xrif_list_check(n_start, n_end, len(f_list), n_workers)
        # number of tasks to execute
        n_tasks =  n_end_x - n_start_x
        # create the multiprocessing pool
        with Pool(processes=n_workers) as pool:
            # chunksize to use
            n_tasks_per_chunk = ceil(n_tasks / len(pool._pool))
            # report details for reference
            print(f'chunksize={n_tasks_per_chunk}, n_workers={len(pool._pool)}')
            results = pool.map(self.proj_xfile, f_list[n_start_x:n_end_x], chunksize=n_tasks_per_chunk)
        # return the dot products:
        dot_results = np.vstack(results)
        return dot_results

    ####################### normalize data by lab returns #######################

    def og_rms(self, data_proj, n_avg=100):
        # this is the original RMS calculation, which is just the stdv across the time axis
        klip = data_proj.shape[1]
        og_all = []
        for i in range(klip):
            sky_rms = moving_stdv(data_proj[:,i], n=n_avg)
            og = sky_rms/self.ref_rms[i] #TODO: check this divides
            og_all.append(og)
        return np.array(og_all)


########## CALC HELPER FUNCTIONS ##########


# rolling calculations
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_stdv(arr, n=3):
    a = n//2
    ret = np.array([np.std(arr[i-a:i+a]) for i in np.arange(a, len(arr)-a)])
    return ret

def return_rolling(dot_mat, n=100):
    dot_avgs = moving_average(dot_mat, n=n)
    dot_stds = moving_stdv(dot_mat, n=n)
    return dot_avgs, dot_stds