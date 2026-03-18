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
import math
import fixr
import lookyloo
from lookyloo.core import get_matching_paths


glob_dir_calib = '/home/eden/data/calib/'
glob_dark = 'camwfs-dark_bin2_2000.000000_600.000000_-45.000000__T20220422005142597501326.fits'
glob_mask = 'aol1_wfsmask.fits'
glob_ref = 'aol1_wfsref.fits'
# to be sent to lookyloo scrips
glob_data_p = pathlib.Path('/data/rawimages/') 
glob_telem_p = pathlib.Path('/srv/aoc/opt/MagAOX')

############ General looky loo xrif pulls ####################
    
def verify_obs_all(obs_name, dt_start, dt_end=None, partial=True):
    '''
    Takes in a observation name
    Returns a list of obs span
    ''' 
    all_obs_spans, _ = lookyloo.core.get_new_observation_spans(data_roots=[glob_telem_p], existing_observation_spans=set(), start_dt=dt_start, end_dt=dt_end)
    # sometimes we want more, sometimes we want less
    if partial:
        obs_spans_with_name = [x for x in all_obs_spans if obs_name in x.title] # strict match on obs name
    else:
        obs_spans_with_name = [x for x in all_obs_spans if obs_name == x.title] # strict match on obs name
    # so we mingt get maore 
    obs_spans_with_name = sorted(obs_spans_with_name, key=lambda obs: obs.begin)
    seps = [f'{e}. \t LENGTH: {obs.end - obs.begin}, \t START: {obs.begin}, \t NAME: {obs.title} ' for e, obs in enumerate(obs_spans_with_name)]
    print("\n".join(seps))
    return obs_spans_with_name

def verify_obs(obs_name, dt_start, n=0, partial=True, dt_end=None):
    '''
    Takes in a observation name
    Returns index n of list of obs span
        - for now, will default to first. 
        - set n to desired new place in list
    ''' 
    obs_spans_with_name = verify_obs_all(obs_name, dt_start, partial=partial, dt_end=dt_end)
    N = len(obs_spans_with_name)
    # check on list index
    if n > N:
        n = -1
    return obs_spans_with_name[n]

def gen_file_list(obs_span, device = 'camtip'):
    '''
    takes in a lookyloo obs span
    will then get all the matching paths for the file lsits. 
    '''
    extension = 'xrif'
    datetime_newer = obs_span.begin
    datetime_older = obs_span.end
    date_folder = datetime_newer.strftime('%Y_%m_%d')
    # checking the files associated with
    glob_data_p_dev = glob_data_p / pathlib.Path(device + '/')
    print(f"DATAPATH: {glob_data_p_dev}")
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
        timing_conglom.extend(ts)
    # stack it (only if we need to)
    if len(data_conglom) > 1:
        data_stack = np.vstack(data_conglom)[n_offset : n_offset + n]
        timing_stack = timing_conglom[n_offset : n_offset + n]
    else:
        print("   => actual shape", np.array(data_conglom[0]).shape)
        print("   => index at", n_offset + n)
        data_stack = np.array(data_conglom[0])[n_offset : n_offset + n]
        timing_stack = np.array(timing_conglom[0])[n_offset : n_offset + n]
    print("file n pull", data_stack.shape)
    #todo: might need resize arrays
    return data_stack, timing_stack

def pull_file_xrif(file_obs, fsize=512, n_pix_w=120):
    '''
    Take a path, return the data and the timing
    '''
    with open(file_obs.path, 'rb') as fh:
        data = fixr.xrif2numpy(fh)
        timing = fixr.xrif2numpy(fh)
    #TODO: check if this is the right approach
    n_files = int(data.size / n_pix_w / n_pix_w)
    data_reshaped = np.reshape(data, (n_files,n_pix_w,n_pix_w))
    timing_reshaped = np.reshape(timing, (n_files,5))
    # timings at datetimes
    timing_dt = np.array(timing_datetimes(file_obs, timing_reshaped))
    # return data an associated datetime
    return data_reshaped, timing_dt

def timing_datetimes(file_obs, timing):
    # cleaning up timing 
    dt_start = file_obs.timestamp
    dt_s = timing[:, 1] - timing[0, 1]
    dt_s_us = dt_s + (timing[:, 2]/1e9)
    frame_dt = [dt_start + datetime.timedelta(seconds = int(dt_i), microseconds = int((dt_i - math.floor(dt_i))*1e6)) for dt_i in dt_s_us]
    return frame_dt

