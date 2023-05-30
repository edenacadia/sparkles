## sparkles.py
# Eden McEwen 
# Last edited September 14th

from pickle import TRUE
from astropy.io import fits
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import tempfile
import imageio
import os

# calibration files
calib_path = '../data/calib/'
mask = 'aol1_wfsmask.fits'
ref = 'aol1_wfsref.fits'
dark = 'camwfs-dark_bin2_2000.000000_600.000000_-45.000000__T20220422005142597501326.fits'

def return_files(file_path):
    dir_list = os.listdir(file_path)
    dir_list.sort()
    return dir_list


# function to sample randomly files, or to request certain files back
def file_sample(n, dir_list, dir_path,  n_start=-1):
    # randomly generate a number between 0 and len (dir list)
    if (n_start < 0) or (n_start > len(dir_list) - n):
        n_start = np.random.randint(len(dir_list) - n)
    # lists to store relevant params
    data_cube = []
    frame_list = []
    wrt_list = []
    #cont = TRUE
    # Iter from start to start+leng
    for file in dir_list[n_start:n_start+n]:
        with fits.open(dir_path+file) as hdul:
            frame_list.append(hdul[0].header['FRAMENO'])
            wrt_list.append(hdul[0].header['WRTSEC'] + hdul[0].header['WRTNSEC']*10**(-9))
            data_cube.append(hdul[0].data)
            
    return data_cube, n_start, frame_list, wrt_list

def check_cont(frame_list, wrt_list, Hz = 2000):
    if np.max(np.diff(frame_list)) > 1: 
        return False
    if np.max(np.diff(wrt_list)) > 2/Hz :
        return False
    return True

# function, takes in data matrix, returns cleaned matrix
def clean_data_cube(data_cube, calib_path, dark, mask, ref):
    dark_data = fits.open(calib_path + dark)[0].data
    mask_data = fits.open(calib_path + mask)[0].data
    ref_data = fits.open(calib_path + ref)[0].data
    dark_mat =  np.repeat(dark_data[np.newaxis, :, :], len(data_cube), axis=0)
    mask_mat =  np.repeat(mask_data[np.newaxis, :, :], len(data_cube), axis=0)
    ref_mat =  np.repeat(ref_data[np.newaxis, :, :], len(data_cube), axis=0)
    # NORMALIZATION
    mat_dark_sub = data_cube - dark_mat
    mat_normed = np.divide(mat_dark_sub * mask_mat, np.sum(mat_dark_sub * mask_mat, axis=(1,2))[:,None,None])
    # BUG: our ref not 1 normalized, even with the mask, so forcing it to be normal here
    ref_normed = np.divide(ref_mat * mask_mat, np.sum(ref_mat * mask_mat, axis=(1,2))[:,None,None])
    mat_ref_sub = mat_normed - ref_normed
    #mat_ref_sub = mat_normed - ref_mat*mask_mat # if you didn't want the ref normed

    return mat_ref_sub, dark_mat, mask_mat, ref_mat, ref_normed

# splitting into quartiles
def split_data(data_cube):
    arr_s1 = data_cube[::4,:,:]
    arr_s2 = data_cube[1::4,:,:]
    arr_s3 = data_cube[2::4,:,:]
    arr_s4 = data_cube[3::4,:,:]
    return arr_s1, arr_s2, arr_s3, arr_s4

# every four frames, subtract out. 
def split_data_sub(data_cube):
    arr_s1 = data_cube[::4,:,:]
    arr_s2 = data_cube[1::4,:,:]
    arr_s3 = data_cube[2::4,:,:]
    arr_s4 = data_cube[3::4,:,:]
    arr_avg = (arr_s1 + arr_s2 + arr_s3 + arr_s4)/4
    arr_s1 -= arr_avg
    arr_s2 -= arr_avg
    arr_s3 -= arr_avg
    arr_s4 -= arr_avg
    return arr_s1, arr_s2, arr_s3, arr_s4


# for the dot product calculations
def return_dot_mat(splits_arr, mean_arr):
    mean_splts_arr = np.nanmean(mean_arr, axis=1)
    mean_splts_arr_norm = np.array([avg_split/np.linalg.norm(avg_split.flatten())for avg_split in mean_splts_arr])
    dot_mat = np.array([[np.dot(mean_splts_arr_norm[e].flatten(), array.flatten()) for array in split] for e, split in enumerate(splits_arr)])
    return dot_mat

# rolling calculations
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_stdv(a, n=3):
    ret = np.array([np.std(a[i:i+n]) for i in np.arange(len(a-n))])
    return ret[:-n+1]

def return_rolling(dot_mat, n=100):
    dot_avgs = np.array([moving_average(dot, n=n) for dot in dot_mat])
    dot_stds = np.array([moving_stdv(dot, n=n) for dot in dot_mat])
    return dot_avgs, dot_stds

def return_rolling_simp(dot_mat, n=100):
    dot_avgs = moving_average(dot_mat, n=n)
    dot_stds = moving_stdv(dot_mat, n=n)
    return dot_avgs, dot_stds