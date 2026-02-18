# spark_xrif.py
# Eden McEwen 
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
import os
import pathlib
# Special joseph additions
import datetime
import fixr
import lookyloo
from lookyloo.core import get_matching_paths
from sparkles.xrif_pull import *

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

    def __init__(self, sky_obs_name, lab_obs_name, p_dark_s, 
                 p_dark_l, dir_calib, file_mask, file_ref, 
                 dt_start=datetime.datetime(2025, 4, 1, tzinfo=datetime.timezone.utc), sky_obs_n = 0,  n_avg=10000, ref_norm = False ):
        # setting up file references
        self.sky_obs_name = sky_obs_name
        self.lab_obs_name = lab_obs_name
        self.dir_calib  = dir_calib
        self.dt_start = dt_start
        #saving the file names that correspond to this 
        self.p_dark_s = p_dark_s
        self.p_dark_s = p_dark_l
        self.f_mask = file_mask
        self.f_ref = file_ref
        self.n_avg = n_avg
        # Saving the data to reference later
        self.dark_data_sky = fits.open(p_dark_s)[0].data
        self.dark_data_lab = fits.open(p_dark_l)[0].data
        self.mask_data = fits.open(dir_calib + file_mask)[0].data
        self.ref_data = fits.open(dir_calib + file_ref)[0].data
        #decide how to norm data, with or without mask
        self.ref_norm = ref_norm
        if self.ref_norm:
            self.ref_normed = np.divide(self.ref_data * self.mask_data, np.sum(self.ref_data * self.mask_data))
        else:
            self.ref_normed = np.divide(self.mask_data, np.sum(self.mask_data))
        # NAN mat - used later
        self.mask_nan = self.mask_data.copy()
        self.mask_nan[self.mask_nan == 0] = np.nan
        # Using the observation names to get an observation span that can be used in lookyloo
        self.sky_obs_span = verify_obs(sky_obs_name, self.dt_start, n=sky_obs_n) 
        self.lab_obs_span = verify_obs(lab_obs_name, self.dt_start)
        self.check_datasets()
        # setting None for specific check
        self.labroll=1
        self.labref, self.labref_norm, self.labref_rollsub, __ = self.gen_lab_ref()
        return 
    
    def check_datasets(self, fsize=512):
        sky_list = gen_file_list(self.sky_obs_span)
        lab_list = gen_file_list(self.lab_obs_span)
        print("Number of DATA files: ", len(sky_list)*fsize)
        print("Number of REF files: ", len(lab_list)*fsize)
        return
    
    def gen_lab_ref(self, n_lab = 1024, n_start = 0):
        """ 
        Pull a bunch of files, take an average
        """
        print("GENERATING REFERENCE")
        file_lists =  gen_file_list(self.lab_obs_span)
        # LAB+TRUE for file sampling, so using the Lab dark!
        data_ar, timing_ar = self.file_sample_n_clean(file_lists, n = n_lab, n_start = n_start, lab=True)
        # generate a mean for each split
        data_splt = split_data(data_ar)
        mean_splts_arr = np.array([np.nanmean(d_splt, axis=0) for d_splt in data_splt])
        mean_all = np.average(mean_splts_arr, axis=0)
        # Gen labref
        mean_splts_sub = [mean - mean_all for mean in mean_splts_arr]
        # Gen labref_norm
        mean_splts_arr_norm = np.array([avg_split/np.linalg.norm(avg_split.flatten())for avg_split in mean_splts_sub])
        # Gen rollsub
        mean_rollsub = mean_splts_arr - np.roll(mean_splts_arr, 1, axis=0)
        ## dot product lab with its own mean to get worse case amplitude
        self.ref_avg = self.gen_dot_lab_avgs(data_ar, mean_splts_arr_norm)
        self.ref_rollsub = self.gen_dot_lab_rollsub(data_ar, mean_rollsub)
        #TODO: Check to make sure this is continuos 
        return mean_splts_sub, mean_splts_arr_norm, mean_rollsub, timing_ar
    
    def gen_dot_lab_avgs(self, data_ar, lab_ref):
        ref_self_dot = [self.dot_data_wo_ref(data, lab_ref) for data in data_ar]
        ref_derolled = split_data_roll(np.array(ref_self_dot))
        lab_avgs = np.average(ref_derolled, axis=1)
        return lab_avgs
    
    def gen_dot_lab_rollsub(self, data_ar, lab_rollsub):
        # TODO: check the averging here
        data_rollsub = data_ar - np.roll(np.array(data_ar), self.labroll, axis=0)
        ref_self_dot = [self.dot_data_wo_ref(d_rs, lab_rollsub) for d_rs in data_rollsub]
        ref_derolled = split_data_roll(np.array(ref_self_dot))
        lab_avgs = np.average(ref_derolled, axis=1)
        return lab_avgs
    
    def file_sample_n_clean(self, file_lists, n, n_start=0, norm=True, lab=False):
        data, timing = pull_n_files(file_lists, n, n_start=n_start)
        data_clean = self.file_clean(data, norm=norm, lab=lab)
        return data_clean, timing

    def xfile_sample_clean(self, p, norm=True, lab = False):
        data, t = pull_file_xrif(p) # this is gonna be 512
        data_clean = self.file_clean(data, lab=lab)
        return data_clean, t

    # cleaning frames before using them
    def clean_frames(data_cube, dark, mask):
        # Dark subtract 
        data_dark_sub = data_cube - dark
        # normalize in the mask 
        data_masked = data_dark_sub * mask
        # make friendly with a data stack
        data_normed = data_masked / np.sum(data_masked, axis=(1,2))[:,np.newaxis,np.newaxis]
        # then, we can do the mean subtraction
        data_subed = data_normed - np.mean(data_normed, axis=0)
        return data_subed 
    
    def file_clean(self, data, norm = True, lab = False):
        """ Clean a single file """
        if lab:
            data_dark_sub = data - self.dark_data_lab
        else:
            data_dark_sub = data - self.dark_data_sky
        # Normalization standard proc.
        if norm:
            data_dark_sub_mask = data_dark_sub * self.mask_data
            data_normed = np.divide(data_dark_sub_mask, np.sum(data_dark_sub_mask)) # this sum should be one 
            data_ref_sub = data_normed - self.ref_normed 
            data_return = data_ref_sub
        else: 
            data_return = data_dark_sub
        return data_return
    
    def dot_data(self, data):
        """ Automatically takes the dot product with the saved ref norm """
        dot_4way = np.array([np.nansum(ref_split.flatten()*data.flatten()) for ref_split in self.labref_norm])
        return dot_4way
    
    def dot_data_rollsub(self, data, roll=1):
        """ Automatically takes the dot product with the saved ref norm """
        data_rollsub = data - np.roll(np.array(data), roll, axis=0)
        dot_4way = np.array([np.nansum(ref_split.flatten()*data_rollsub.flatten()) for ref_split in self.labref_rollsub])
        return dot_4way
    
    def dot_data_wo_ref(self, data, ref):
        dot_4way = np.array([np.nansum(ref_split.flatten()*data.flatten()) for ref_split in ref])
        return dot_4way
    
    def dot_xfile(self, p, norm=True, lab=False):
        """ Return dot product of single file """
        data, t = self.xfile_sample_clean(p, norm=norm, lab=lab)
        dot_4way = np.array([self.dot_data(d) for d in data])
        return dot_4way, t
    
    def dot_xfile_avg(self, p, n_avg = 512, norm=True, lab=False):
        """ Return dot product of an xrif file avg
         this is a quick and dirty average for speed """
        data, t = self.xfile_sample_clean(p, norm=norm, lab=lab)
        data_split = np.array(split_data(data)) 
        data_avg = np.average(data_split, axis=1)
        # data_avg_all = np.average(data_split, axis=(0,1)) # Don't use this, it's ill advised
        dot_4way = np.array([self.dot_data(d) for d in data_avg])
        return dot_4way, t
    
    def xrif_list_check(n_start, n_end, l_len, n_work, fsize=512):
        ''' Converting files to xrif indexes '''
        n_start_x = n_start // fsize # how many xrif files over to shift
        n_end_x = (n_end // fsize) + 1
        # check the end number 
        if n_start_x > l_len:
            print("ERROR: n_start isn't usable")
            n_start = 0
        if n_end_x == -1:
            n_end_x = (l_len//n_work)*n_work
        elif n_end_x < n_start_x:
            print("ERROR: n_end isn't usable")
            n_end_x = n_start_x + 10 # arbitrary 10 file add
        elif n_end_x > l_len:
            print("ERROR: changing n_end to file count")
            n_end_x = (l_len//n_work)*n_work
        return n_start_x, n_end_x

    def dot_list_pool(self, n_start = 0, n=100, n_workers=4, avg=False):
        """
        return a list of dot products
        """
        f_list = gen_file_list(self.sky_obs_span) # number of xrif files
        n_end = n_start + n
        # changing so these 
        n_start_x, n_end_x = self.xrif_list_check(n_start, n_end, len(f_list), n_workers)
        # number of tasks to execute
        n_tasks =  n_end_x - n_start_x
        # create the multiprocessing pool
        with Pool(processes=n_workers) as pool:
            # chunksize to use
            n_tasks_per_chunk = ceil(n_tasks / len(pool._pool))
            # report details for reference
            print(f'chunksize={n_tasks_per_chunk}, n_workers={len(pool._pool)}')
            # issue tasks and process results
            #TODO: WHAT THE FUCK AM I DOING HERE AHHHHH
            if avg:
                results = pool.map(self.dot_xfile_avg, f_list[n_start_x: n_end_x], chunksize=n_tasks_per_chunk)
            else:
                results = pool.map(self.dot_xfile, f_list[n_start_x: n_end_x], chunksize=n_tasks_per_chunk)
        # return the dot products:
        dot_results = np.vstack([r[0] for r in results])
        return dot_results
    
    def dot_xfile_n_avg(self, n_start, f_list, n, norm=True, lab=False):
        data, t = self.file_sample_n_clean(f_list, n, n_start=n_start, norm=norm, lab=lab)
        print("data: ", np.array(data).shape)
        data_split = np.array(split_data(data)) 
        data_avg = np.average(data_split, axis=1)
        dot_4way = np.array([self.dot_data(d) for d in data_avg])
        print("dot: ", dot_4way.shape)
        return dot_4way
    
    def dot_xfile_n_avg_rollsub(self, n_start, f_list, n, norm=True, roll=1, lab=False):
        data, t = self.file_sample_n_clean(f_list, n, n_start=n_start, norm=norm, lab=lab)
        print("data: ", np.array(data).shape)
        data_split = np.array(split_data(data)) 
        data_avg = np.average(data_split, axis=1)
        # now using a new dot product
        dot_4way = np.array([self.dot_data_rollsub(d, roll=roll) for d in data_avg])
        print("dot: ", dot_4way.shape)
        return dot_4way
    
    def file_list_check(n_start, n_end, l_len, n_work):
        # check the end number 
        if n_start > l_len:
            print("ERROR: n_start isn't usable")
            n_start = 0
        if n_end == -1:
            n_end = (l_len//n_work)*n_work
        elif n_end < n_start:
            print("ERROR: n_end isn't usable")
            n_end = n_start + 100
        elif n_end > l_len:
            print("ERROR: changing n_end to file count")
            n_end = (l_len//n_work)*n_work

    def dot_list_pool_avg(self, n_start=0, n=10000, avg=100, n_workers=4, rollsub=False, roll=1):
        ''' 
        This fucntion takes in arbitrary averaging values
        This is annoyin gbc xrif comes in packages    
        '''
        f_list = gen_file_list(self.sky_obs_span) # number of xrif files
        # split the iter by no .of averaging 
        n_chunks = n // (avg*4) # make the end evenly divisible thanks
        n_end = n_start + avg*4*n_chunks
        # we now know number of tasks based on start and end
        n_tasks = int((n_end - n_start) / (avg*4))
        n_start_list = [n_start + ni*avg*4 for ni in range(n_tasks)]
        print(n_start_list)
        # chunking the processes
        print(f"N = {n_end}, chunk size = {avg*4}, N_Chunks {n_chunks}")
        with Pool(processes=n_workers) as pool:
            # chunksize to use
            n_tasks_per_chunk = ceil(n_tasks / len(pool._pool))
            # report details for reference
            print(f'chunksize={n_tasks_per_chunk}, n_workers={len(pool._pool)}')
            # preset the partial with the list of files and number of files to average at a time
            if rollsub:
                func = partial(self.dot_xfile_n_avg_rollsub, f_list=f_list, n=avg*4, roll=roll)
            else: 
                func = partial(self.dot_xfile_n_avg, f_list=f_list, n=avg*4)
            # then the map function will iter over the list of starting positions
            results = pool.map(func, n_start_list, chunksize=n_tasks_per_chunk)
        dot_results = np.vstack(results)
        return dot_results

########## CALC HELPER FUNCTIONS ##########

def split_data(data_cube):
    arr_s1 = data_cube[::4,:,:]
    arr_s2 = data_cube[1::4,:,:]
    arr_s3 = data_cube[2::4,:,:]
    arr_s4 = data_cube[3::4,:,:]
    return arr_s1, arr_s2, arr_s3, arr_s4

def split_data_roll(dot_data, roll=0):
    print("split roll ", np.array(dot_data).shape)
    return np.array([dot_data[i::4,(i+roll)%4] for i in range(4)])

def split_data_dot_roll(dot_data, roll=0):
    print("split roll ", np.array(dot_data).shape)
    return np.array([dot_data[i::4, roll] for i in range(4)])

def pick_roll(data_stack):
    '''
    Picks the roll from maximum average 
    '''
    data_allroll = np.array([split_data_dot_roll(data_stack, roll=i) for i in range(4)])
    roll_avgs = np.mean(data_allroll, axis = (1,2))
    roll_i = np.argmax(roll_avgs)
    print(f'Choosing roll index {roll_i}')
    dot_deroll = split_data_dot_roll(data_stack, roll=roll_i)
    return dot_deroll

# rolling calculations
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_stdv(a, n=3):
    ret = np.array([np.std(a[i:i+n]) for i in np.arange(len(a-n))])
    return ret[:-n+1]

def return_rolling(dot_mat, n=100):
    dot_avgs = moving_average(dot_mat, n=n)
    dot_stds = moving_stdv(dot_mat, n=n)
    return dot_avgs, dot_stds


def plot_time_series(data_split_stack, plot_name, ref_avg=[1,1,1,1], hz = 2000, total_s = -1, params=True):
    #using the file directory to pull 
    # set up plot
    colors = pl.cm.tab20(np.arange(4))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,3), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(f"{plot_name}", y=0.99)
    #generating t axis 
    frame_n = data_split_stack.shape[1]
    if total_s == -1:
        total_s = 4*frame_n/hz
    x_axis = np.arange(0, total_s, total_s/frame_n)
    #plot each split in a complementary color
    for e in range(4):
        axs.plot(x_axis, data_split_stack[e]/ref_avg[e], lw=2, alpha = 0.7, label = f'ref {e}', c = colors[e])
        #axs.set_ylabel(f"sparkle split {e}")
        #axs.set_ylim(bottom=0, top=0.005)
    og_est = np.mean([data_split_stack[e]/ref_avg[e] for e in range(4)], axis=0)
    plt.plot(x_axis, og_est, ls='--', c='grey', label=f'OG avg {np.mean(og_est):0.3f}')
    plt.legend()
    plt.xlabel('seconds (s)')
    return plt

def plot_time_series_div(data_split_stack, ref_avg, plot_name, hz = 2000, total_s = -1, params=True):
    #using the file directory to pull 
    # set up plot
    colors = pl.cm.tab20(np.arange(4))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,3), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(f"{plot_name}", y=0.99)
    #generating t axis 
    frame_n = data_split_stack.shape[1]
    if total_s == -1:
        total_s = 4*frame_n/hz
    x_axis = np.arange(0, total_s, total_s/frame_n)
    #plot each split in a complementary color
    for e in range(4):
        axs.plot(x_axis, data_split_stack[e], lw=2, alpha = 0.7, label = f'ref {e}', c = colors[e])
        #axs.set_ylabel(f"sparkle split {e}")
        #axs.set_ylim(bottom=0, top=0.005)
    plt.legend()
    plt.xlabel('seconds (s)')
    return plt

def plot_time_series_roll(data_roll_stack, plot_name, ref_avg=[1,1,1,1], n_avg = 1000, hz = 2000, total_s = -1, params=True):
    #generating t axis 
    frame_n = data_roll_stack.shape[2]
    if total_s == -1:
        total_s = 4*frame_n/hz
    x_axis = np.arange(0, total_s, total_s*n_avg/frame_n)
    # set up plot
    colors = pl.cm.tab20(np.arange(4))
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12,8), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(f"{plot_name}", y=0.99)
    #plot each split in a complementary color
    for f in range(4):
        for e in range(4):
            axs[f].plot(data_roll_stack[f, e, :]/ref_avg[e], lw=2, alpha = 0.7, label = f'ref {e}', c = colors[e])
        axs[f].set_ylabel(f'Roll {f}')
        #axs[f].set_ylim(bottom=np.min(data_roll_stack), top=np.max(data_roll_stack))
    axs[0].legend()
    plt.xlabel('seconds (s)')
    return plt