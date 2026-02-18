
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
                 p_dark_l, dir_calib, file_mask, n_avg=10000):
        # setting up file references
        self.sky_obs_name = sky_obs_name
        self.lab_obs_name = lab_obs_name
        self.dir_calib  = dir_calib
        self.dt_start = dt_start
        #saving the file names that correspond to this 
        self.p_dark_s = p_dark_s
        self.p_dark_s = p_dark_l
        self.f_mask = file_mask
        self.n_avg = n_avg
        # Saving the data to reference later
        self.dark_data_sky = fits.open(p_dark_s)[0].data
        self.dark_data_lab = fits.open(p_dark_l)[0].data
        self.mask_data = fits.open(dir_calib + file_mask)[0].data
        # NAN mat - used later
        self.mask_nan = self.mask_data.copy()
        self.mask_nan[self.mask_nan == 0] = np.nan
        # Using the observation names to get an observation span that can be used in lookyloo
        #self.sky_obs_span = verify_obs(sky_obs_name, self.dt_start, n=sky_obs_n) 
        #self.lab_obs_span = verify_obs(lab_obs_name, self.dt_start)
        #self.check_datasets()
        # setting None for specific check
        #self.gen_lab_ref()
        return 
    
    def set_data(self, dt_start=datetime.datetime(2025, 4, 1, tzinfo=datetime.timezone.utc), 
                 sky_obs_n=0, lab_obs_n=0):
        """
        Changing the behavior so that 
        """
        self.dt_start = dt_start
        self.sky_obs_n = sky_obs_n
        self.lab_obs_n = lab_obs_n
        self.sky_obs_span = verify_obs(self.sky_obs_name, self.dt_start, n=sky_obs_n) 
        self.lab_obs_span = verify_obs(self.lab_obs_name, self.dt_start, n= lab_obs_n)
        self.check_datasets()
        # setting None for specific check
        self.gen_lab_ref()
    
    
    def check_datasets(self, fsize=512):
        sky_list = gen_file_list(self.sky_obs_span)
        lab_list = gen_file_list(self.lab_obs_span)
        print("Number of DATA files: ", len(sky_list)*fsize)
        print("Number of REF files: ", len(lab_list)*fsize)
        return

    ##################### LAB REFERENCE GENERATION ######################
    
    def gen_lab_ref(self, n_lab = 1024, n_start = 0, klip=3, plt_img=False):
        """ 
        Pull a bunch of files, take an average
        """
        print("GENERATING REFERENCE PCA basis")
        file_lists =  gen_file_list(self.lab_obs_span)
        # LAB+TRUE for file sampling, so using the Lab dark!
        data_ar, timing_ar = self.file_sample_n_clean(file_lists, n = n_lab, n_start = n_start, lab=True)
        # creating the PCA basis
        Z_KL, Z_KL_img = self.pca_basis(data_ar, klip=klip)
        self.ref_pca = Z_KL
        self.ref_pca_img = Z_KL_img
        # TODO: save a plot of these images

        # Then, we're going to create a self reference, for each roll. 
        lab_proj = self.pca_projection(data_ar) # this is for all frames
        # averages by the sparkle pattern
        lab_avgs = np.array([np.mean(lab_proj[i::4,:], axis=0) for i in range(4)])
        # These should be rolled later for best effect
        self.ref_proj = lab_avgs

        return Z_KL, Z_KL_img, lab_avgs, timing_ar

    ################### PCA basis definitions ###################
    
    def pca_basis(self, data_ar, klip=3):
        # need to delete the mean here:
        data_ar = data_ar - np.average(data_ar, axis=0)
        # generate a PCA basis from these files
        K, x, y = data_ar.shape
        N = x*y
        image_shape = (x, y)
        reference_lab = data_ar.reshape(K, N)
        # calc covariance
        E = np.cov(reference_lab) * (N - 1)
        # find eigenvalues and eigenvectors
        lambda_values_out, C_out = np.linalg.eigh(E)
        lambda_values = np.flip(lambda_values_out)
        C = np.flip(C_out, axis=1)
        # generate the KL basis 
        Z_KL_lab = reference_lab.T @ (C * np.power(lambda_values, -1/2))
        Z_KL_lab_truc = Z_KL_lab[:,:klip]
        # reshape for actual images
        Z_KL_lab_images = Z_KL_lab.T.reshape((reference_lab.shape[0],) + image_shape)
        Z_KL_lab_images_truc = Z_KL_lab_images[:klip,:,:]
        return Z_KL_lab_truc, Z_KL_lab_images_truc

    def pca_projection(self, data_ar):
        # project each frame of the data onto the PCA basis
        data_ar = data_ar - np.average(data_ar, axis=0)
        # reshaping 
        reference = data_ar.reshape(data_ar.shape[0], data_ar.shape[1]*data_ar.shape[2])
        # project each frame onto PCA basis
        projection = np.array([self.Z_KL.T @ ref for ref in reference])
        return projection
    
    ################### file sampling ###################

    def file_sample_n_clean(self, file_lists, n, n_start=0, norm=True, lab=False):
        data, timing = pull_n_files(file_lists, n, n_start=n_start)
        data_clean = self.file_clean(data, norm=norm, lab=lab)
        return data_clean, timing

    def xfile_sample_clean(self, p, norm=True, lab=False):
        # collecting many xrif files
        data, t = pull_file_xrif(p) # this is gonna be 512
        # cleaning those files with lab data
        data_clean = self.file_clean(data, lab=lab)
        return data_clean, t
    
    def file_clean(self, data, norm=True, lab=False):
        """ Clean a single file 
        recently removed the ref
        TODO: subtract the average?
        """
        if lab:
            data_dark_sub = data - self.dark_data_lab
        else:
            data_dark_sub = data - self.dark_data_sky
        # Normalization standard proc.
        if norm:
            data_dark_sub_mask = data_dark_sub * self.mask_data
            data_normed = np.divide(data_dark_sub_mask, np.sum(data_dark_sub_mask)) # this sum should be one 
            data_return = data_normed
        else: 
            data_return = data_dark_sub
        return data_return

    ################### process data ###################

    def dot_data(self, data):
        """ Automatically takes the dot product with the saved ref norm """
        dot_4way = np.array([np.nansum(ref_split.flatten()*data.flatten()) for ref_split in self.labref_norm])
        return dot_4way
    
    def dot_xfile(self, p, norm=True, lab=False):
        """ Return dot product of single file """
        data, t = self.xfile_sample_clean(p, norm=norm, lab=lab)
        dot_4way = np.array([self.dot_data(d) for d in data])
        return dot_4way, t

    def proj_xfile(self, p, norm=True, lab=False):
        """
        Docstring for proj_xfile
        """
        # pull data. lab indicates what dark to use
        data_arr, t = self.xfile_sample_clean(p, norm=norm, lab=lab)
        # now each frame should be projected onto the PCA basis
        data_proj = self.pca_projection(data_arr) 
        # first axis file number, second axis PCA return
        return data_proj

    def proj_pool(self, n_start=0, n=100, n_workers=4):
        """
        Uses the list of files the object was generated with. 
        """
        f_list = gen_file_list(self.sky_obs_span) # number of xrif files
        n_end = n_start + n
        # changing so these 
        n_start_x, n_end_x = xrif_list_check(n_start, n_end, len(f_list), n_workers)
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
        dot_results = np.vstack([r[0] for r in results])
        return dot_results

    ####################### normalize data by lab returns #######################

    def proj_norm(self, data_proj):
        """
        Take in a data array, normalize by the lab_projections
        This is dependent on roll, so we will have 4 options for each PCA basis, per frame
        data_proj: [N, klip]
        data_proj_norm: [N, klip, 4] 
        """
        
    def proj_rms(self, data_proj, rms_n=100):
        """
        This compares the RMS between lab and sky projections, picked with 
        """

############ General looky loo xrif pulls ####################

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

def verify_obs_all(obs_name, dt_start):
    '''
    Takes in a observation name
    Returns a list of obs span
        - for now, will default to first. 
        - set n to desired new place in list
    ''' 
    run_start_dt =  dt_start
    all_obs_spans, _ = lookyloo.core.get_new_observation_spans(data_roots=[glob_telem_p], existing_observation_spans=set(), start_dt=run_start_dt)
    obs_spans_with_name = [x for x in all_obs_spans if obs_name in x.title] #this can return multiple spans
    # so we mingt get maore 
    obs_spans_with_name = sorted(obs_spans_with_name, key=lambda obs: obs.begin)
    print([f'{obs.end - obs.begin} \n' for obs in obs_spans_with_name] )
    return obs_spans_with_name

def verify_obs(obs_name, dt_start, n=0):
    '''
    Takes in a observation name
    Returns index n of list of obs span
        - for now, will default to first. 
        - set n to desired new place in list
    ''' 
    obs_spans_with_name = verify_obs_all(obs_name, dt_start)
    N = len(obs_spans_with_name)
    # check on list index
    if n > N:
        n = -1
    return obs_spans_with_name[n]

def gen_file_list(obs_span, device = 'camwfs'):
    '''
    takes in a lookyloo obs span
    will then get all the matching paths for the file lsits. 
    '''
    extension = 'xrif'
    datetime_newer = obs_span.begin
    datetime_older = obs_span.end
    # checking the files associated with
    glob_data_p_dev = glob_data_p + device + '/'
    obs_file_list = lookyloo.core.get_matching_paths(glob_data_p_dev, device, extension, datetime_newer, datetime_older)
    return sorted(obs_file_list, key=lambda obs: obs.timestamp)

def pull_n_files(file_lists, n, n_start=0, fsize=512):
    '''
    We can only pull in chunks of 512
    - take n and pull at least that many 
    - check timings
    '''
    print(f"PULLING {n} FILES")
    if n==0:
        print("This is an error!")
        return np.array([[[]]]), np.array([[]])
    # make a pull of (n//512 + 1) * 512 files
    xrif_count = (n // fsize) + 1
    n_start_xrif = n_start // fsize # how many xrif files over to shift
    n_offset = n_start % fsize # within anxrif, the offset
    print(f"FILE no {xrif_count}, n_start {n_start}, n {n}, n_offset {n_offset}")
    print(f"XRIF index {n_start_xrif}, no of files {xrif_count}, len list {len(file_lists)}")
    # make a NP array
    data_conglom = []
    timing_conglom = []
    # one by one and open
    for i in np.arange(n_start_xrif, n_start_xrif+xrif_count):
        d, ts = pull_file_xrif(file_lists[i], fsize=fsize)
        data_conglom.append(d)
        timing_conglom.append(ts)
    # stack it (only if we need to)
    if len(data_conglom) > 1:
        data_stack = np.vstack(data_conglom)[n_offset : n_offset + n]
        timing_stack = np.vstack(timing_conglom)[n_offset : n_offset + n]
    else:
        print("   => actual shape", np.array(data_conglom[0]).shape)
        print("   => index at", n_offset + n)
        data_stack = np.array(data_conglom[0])[n_offset : n_offset + n]
        timing_stack = np.array(timing_conglom[0])[n_offset : n_offset + n]
    print("file n pull", data_stack.shape)
    #todo: might need resize arrays
    return data_stack, timing_stack

def pull_file_xrif(file_obs, fsize=512):
    '''
    Take a path, return the data and the timing
    '''
    with open(file_obs.path, 'rb') as fh:
        data = fixr.xrif2numpy(fh)
        timing = fixr.xrif2numpy(fh)
    #TODO: check if this is the right approach
    n_files = int(data.size / 120 / 120)
    return np.reshape(data, (n_files,120,120)),  np.reshape(timing, (n_files,5))

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

def plot_KL_modes(KL_imgs, n):
    plt.figure(figsize=(7, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.title(f"KL mode {i}")
        plt.imshow(KL_imgs[i])
    # TODO:
    #plt.suptitle(f"KL modes \n {obs_plot_name} {lab_name}")
    plt.tight_layout()
    return plt