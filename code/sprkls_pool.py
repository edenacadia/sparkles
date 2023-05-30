
# sprkls_pool.py
# Eden McEwen 
# pooling the sparkles average calculations
# doing this file by file for optimization

from astropy.io import fits
from matplotlib import pyplot as plt
from functools import partial
import multiprocessing as mp
from scipy import signal
import numpy as np
from multiprocessing import Pool
from math import ceil
import os
import pandas as pd
import glob
import matplotlib.pylab as pl

glob_dir_calib = '../data/calib/'
glob_mask = 'aol1_wfsmask.fits'
glob_ref = 'aol1_wfsref.fits'
glob_dark = 'camwfs-dark_bin2_2000.000000_600.000000_-45.000000__T20220422005142597501326.fits'

class Spark:
    Hz = 2000
    n_pool = 4

    def __init__(self, dir_data, dir_lab, dir_calib, file_dark, file_mask, file_ref, n_avg=10000, ref_norm = False):
        # setting up file references
        self.dir_data = dir_data
        self.dir_lab = dir_lab
        self.dir_calib  = dir_calib
        #saving the file names that correspond to this 
        self.f_dark = file_dark
        self.f_mask = file_mask
        self.f_ref = file_ref
        self.n_avg = n_avg
        # Saving the data to reference later
        self.dark_data = fits.open(dir_calib + file_dark)[0].data
        self.mask_data = fits.open(dir_calib + file_mask)[0].data
        self.ref_data = fits.open(dir_calib + file_ref)[0].data
        self.ref_norm = ref_norm
        if self.ref_norm:
            self.ref_normed = np.divide(self.ref_data * self.mask_data, np.sum(self.ref_data * self.mask_data))
        else:
            self.ref_normed = np.divide(self.mask_data, np.sum(self.mask_data))
        # NAN mat
        self.mask_nan = self.mask_data.copy()
        self.mask_nan[self.mask_nan == 0] = np.nan
        # Getting freuency
        self.lab_HZ = self.get_Hz(dir_lab)
        self.data_HZ = self.get_Hz(dir_data)
        # Getting spark params
        self.check_datasets()
        self.set_spark_params()
        # setting None for specific check
        self.set_lab_ref()
        return 

    def check_datasets(self):
        print("Number of DATA files: ", len(self.file_lister(self.dir_data)))
        print("Number of REF files: ", len(self.file_lister(self.dir_lab)))
        return

    def gen_lab_ref(self, n_lab = 100, n_start = 0):
        """ 
        Pull a bunch of files, take an average
        """
        print("GENERATING REFERENCE")
        all_return = self.file_sample_n_clean(self.dir_lab, n = n_lab, n_start = n_start)
        # NAN matrix for fun other things
        data_ar = np.array([ar[0] for ar in all_return])
        frame_ar = [ar[1] for ar in all_return]
        wrt_ar = [ar[2] for ar in all_return]
        # generate a mean for each split
        data_splt = split_data(data_ar)
        mean_splts_arr = [np.nanmean(d_splt, axis=0) for d_splt in data_splt]
        mean_splts_sub = [mean - np.average(mean_splts_arr, axis=0) for mean in mean_splts_arr]
        mean_splts_arr_norm = np.array([avg_split/np.linalg.norm(avg_split.flatten())for avg_split in mean_splts_sub])
        ## dot product lab with its own mean to get worse case amplitude
        ref_self_dot = [self.dot_data_wo_ref(data, mean_splts_arr) for data in data_ar]
        ref_self_dot_avg = self.gen_dot_lab_avgs(split_data_roll(np.array(ref_self_dot)))
        self.ref_avg = ref_self_dot_avg
        #TODO: Check to make sure this is continuos 
        return mean_splts_sub, mean_splts_arr_norm, frame_ar, wrt_ar
    
    def set_lab_ref(self, n_lab = 100, n_start = 0):
        # TODO: If there is instead ref files, pull those instead

        # pull the lab name, switch to make it the ref and splits list ones
        saved_lab_ref = self.dir_lab.replace("camwfs/", "lab_ref.fits") # normalize lab ref
        # if these files exist, pull them.
        if os.path.isfile(saved_lab_ref):
            print("Found saved lab ref at: ", saved_lab_ref)
            with fits.open(saved_lab_ref) as hdul:
                lab_ref_data = hdul[0].data
            self.labref, self.labref_norm = lab_ref_data, lab_ref_data
            return
        
        self.labref, self.labref_norm, __, __ = self.gen_lab_ref(n_lab = n_lab, n_start = n_start)
        return
    
    def gen_dot_lab_avgs(self, ref_self_dot_avg):
        lab_avgs = np.average(ref_self_dot_avg, axis=1)
        return lab_avgs
    
    def dot_list_pool(self, n=100, n_start = 0, n_workers=4):
        """
        return a list of dot products
        """
        f_list = self.file_lister(self.dir_data)
        n_end = n_start + n
        # check the end number 
        if n_start > len(f_list):
            print("ERROR: n_start isn't usable")
            n_start = 0
        if n_end == -1:
            n_end = (len(f_list)//4)*4
        elif n_end < n_start:
            print("ERROR: n_end isn't usable")
            n_end = n_start + 100
        elif n_end > len(f_list):
            print("ERROR: changing n_end to file count")
            n_end = (len(f_list)//4)*4
        # number of tasks to execute
        n_tasks =  n_end - n_start
        # create the multiprocessing pool
        with Pool(processes=n_workers) as pool:
            # chunksize to use
            n_tasks_per_chunk = ceil(n_tasks / len(pool._pool))
            # report details for reference
            print(f'chunksize={n_tasks_per_chunk}, n_workers={len(pool._pool)}')
            # issue tasks and process results
            results = pool.map(partial(self.dot_file, path=self.dir_data), f_list[n_start: n_end], chunksize=n_tasks_per_chunk)
        # return the dot products:
        #print(results)
        dot_results = [r[0] for r in results]
        frame_results = [r[1] for r in results]
        wrt_results = [r[2] for r in results]
        return dot_results

    def dot_list(self, n=100):
        """ This does the same as pool but without parallelism"""
        #TODO: testing
        f_list = self.file_lister(self.dir_data)
        n_start = 0
        n_end = n
        results = [self.dot_file(f, self.dir_data, norm=True) for f in f_list[n_start: n_end]]
        # return the dot products:
        dot_results = [r[0] for r in results]
        frame_results = [r[1] for r in results]
        wrt_results = [r[2] for r in results]
        return dot_results

    def dot_file(self, f, path, norm=True):
        """ Return dot product of single file"""
        data, frame, wrt = self.file_sample_clean(f, path, norm=norm)
        dot_4way = self.dot_data(data)
        return dot_4way, frame, wrt

    def dot_chunk_pool(self, chunk_size, n_chunks, n_start=0, n_workers=4):
        n_end = n_start + chunk_size*4*n_chunks
        f_list = self.file_lister(self.dir_data) #just for checking length
        # check the end number 
        #
        # number of tasks to execute
        n_tasks = int((n_end - n_start) / (chunk_size*4))
        print(f"processing from file {n_start} to {n_end}, Chunk size {chunk_size*4} for {n_tasks} tasks")
        #creating the list of n_starts
        n_start_list = [n_start + n*chunk_size*4 for n in range(n_tasks)]
        # create the multiprocessing pool
        with Pool(processes=n_workers) as pool:
            # chunksize to use
            n_tasks_per_chunk = ceil(n_tasks / len(pool._pool))
            # report details for reference
            print(f'chunksize={n_tasks_per_chunk}, n_workers={len(pool._pool)}')
            # issue tasks and process results
            results = pool.map(partial(self.dot_chunk, path=self.dir_data, n=chunk_size*4), n_start_list, chunksize=n_tasks_per_chunk)
        # return the dot products:
        dot_results = results
        return dot_results
    
    def dot_chunk(self, n_start, n,  path, norm=True):
        # use the given numbers to pull those files
        all_files = self.file_sample_n_clean(path, n, n_start=n_start, norm=norm)
        # pull only the datafiles
        actual_files = np.array([r[0] for r in all_files])
        # average four way
        split_averages = np.array([np.average(actual_files[e::4], axis=0) for e in range(4)])
        # get the dot values for each set of average files
        dot_4way = [self.dot_data(data) for data in split_averages]
        return dot_4way

    def dot_data(self, data):
        """ Automatically takes the dot product with the saved ref norm"""
        dot_4way = np.array([np.nansum(ref_split.flatten()*data.flatten()) for ref_split in self.labref_norm])
        return dot_4way
    
    def dot_data_wo_ref(self, data, ref):
        dot_4way = np.array([np.nansum(ref_split.flatten()*data.flatten()) for ref_split in ref])
        return dot_4way

    def file_sample_n_clean(self, path, n, n_start=0, norm=True):
        """ Returns the first n files, cleaned """
        print(f"=> Sampling {n} files, starting at {n_start}")
        f_list = self.file_lister(path)
        all_files = [self.file_sample_clean(f, path, norm=norm) for f in f_list[n_start:n_start+n]]
        return all_files

    def file_sample_clean(self, file, path, norm=True):
        """Pull and clean data"""
        data, frame, wrt = self.file_sample(file, path)
        return self.file_clean(data, norm=norm), frame, wrt

    def file_sample(self, file, path):
        """
        Take file name, pull, sample value, return
        """
        f_all = path + file
        #check to make sure path exists
        if not os.path.isfile(f_all):
            print(f"WARNING! {f_all} not found")
            return None, None, None
        # Open file
        try:
            with fits.open(f_all) as hdul:
                frame = hdul[0].header['FRAMENO']
                wrt = hdul[0].header['WRTSEC'] + hdul[0].header['WRTNSEC']*10**(-9)
                data = hdul[0].data
        except Exception as e:
            print(f"WARNING! {e}")
            return None, None, None
        # Return other metrics
        return data, frame, wrt
    
    def file_clean(self, data, norm = True):
        """Clean a single file"""
        data_dark_sub = data - self.dark_data
        if norm:
            data_dark_sub_mask = data_dark_sub * self.mask_data
            data_normed = np.divide(data_dark_sub_mask, np.sum(data_dark_sub_mask)) # this sum should be one 
            data_ref_sub = data_normed - self.ref_normed #
            data_return = data_ref_sub
        else: 
            data_return = data_dark_sub

        return data_return

    def file_lister(self, file_path):
        """
        File lister
        takes in: 
            directory (string)
        returns:
            file_list (list) list of all files in given directory
        """
        # check that the path exists
        #if not os.path.ispath(file_path):
        #    return False
        # Collect all the files in the directory
        dir_list = glob.glob("camwfs*.fits", root_dir=file_path)
        # check that it found files
        if not dir_list:
            return False
        dir_list.sort() # sort to make it in order
        return dir_list

    def get_Hz(self, dir):
        """ Pulling the frequency from the fiven directory """
        hdr = get_hdr_from_dir(dir)
        fps = hdr["HIERARCH CAMWFS FPS"]
        print(f"HZ value: {fps}")
        return fps

    def get_spark_params(self, dir):
        hdr = get_hdr_from_dir(dir)
        s_MOD = hdr['HIERARCH TWEETERSPECK MODULATING']
        s_TRIG = hdr['HIERARCH TWEETERSPECK TRIGGERED']
        s_FREQ = hdr['HIERARCH TWEETERSPECK FREQUENCY']
        s_SEPS = hdr['HIERARCH TWEETERSPECK SEPARATIONS']
        s_ANGS = hdr['HIERARCH TWEETERSPECK ANGLES']
        s_AMPS = hdr['HIERARCH TWEETERSPECK AMPLITUDES']
        s_CRSS = hdr['HIERARCH TWEETERSPECK CROSSES']
        dict_param = {"MOD": s_MOD, "TRIG":s_TRIG, "FREQ":s_FREQ, 
                      "SEPS":s_SEPS, "ANGS":s_ANGS, "AMPS":s_AMPS, "CROSS":s_CRSS}
        return dict_param

    def set_spark_params(self):
        # get the lab spark params
        self.lab_spark_params = self.get_spark_params(self.dir_lab)
        print_sparkle_params(self.lab_spark_params)
        # get the data spark params
        self.data_spark_params = self.get_spark_params(self.dir_data)
        print_sparkle_params(self.data_spark_params)
        return
    
    def get_datetime_start(self, dir):
        hdr = get_hdr_from_dir(dir)
        date = hdr["DATE-OBS"]
        print(f"DATETIME start: {date}")
        return date
        
########## HELPER FUNCTIONS

def get_Hzs(dir):
    """ Pulling the frequency from the fiven directory """
    hdr = get_hdr_from_dir(dir)
    fps = hdr["HIERARCH CAMWFS FPS"]
    print(f"HZ value: {fps}")
    return fps

def get_spark_params(dir):
    hdr = get_hdr_from_dir(dir)
    s_MOD = hdr['HIERARCH TWEETERSPECK MODULATING']
    s_TRIG = hdr['HIERARCH TWEETERSPECK TRIGGERED']
    s_FREQ = hdr['HIERARCH TWEETERSPECK FREQUENCY']
    s_SEPS = hdr['HIERARCH TWEETERSPECK SEPARATIONS']
    s_ANGS = hdr['HIERARCH TWEETERSPECK ANGLES']
    s_AMPS = hdr['HIERARCH TWEETERSPECK AMPLITUDES']
    s_CRSS = hdr['HIERARCH TWEETERSPECK CROSSES']
    dict_param = {"MOD": s_MOD, "TRIG":s_TRIG, "FREQ":s_FREQ, 
                    "SEPS":s_SEPS, "ANGS":s_ANGS, "AMPS":s_AMPS, "CROSS":s_CRSS}
    return dict_param

def get_spark_params_file(file):
    hdr = get_hdr_from_file(file)
    s_MOD = hdr['HIERARCH TWEETERSPECK MODULATING']
    s_TRIG = hdr['HIERARCH TWEETERSPECK TRIGGERED']
    s_FREQ = hdr['HIERARCH TWEETERSPECK FREQUENCY']
    s_SEPS = hdr['HIERARCH TWEETERSPECK SEPARATIONS']
    s_ANGS = hdr['HIERARCH TWEETERSPECK ANGLES']
    s_AMPS = hdr['HIERARCH TWEETERSPECK AMPLITUDES']
    s_CRSS = hdr['HIERARCH TWEETERSPECK CROSSES']
    dict_param = {"MOD": s_MOD, "TRIG":s_TRIG, "FREQ":s_FREQ, 
                    "SEPS":s_SEPS, "ANGS":s_ANGS, "AMPS":s_AMPS, "CROSS":s_CRSS}
    return dict_param

def print_sparkle_params(param_dict):
    for key in param_dict:
        print(key, ":" , param_dict[key], "  " ,end = '')
    print('\b')

def split_data(data_cube):
    arr_s1 = data_cube[::4,:,:]
    arr_s2 = data_cube[1::4,:,:]
    arr_s3 = data_cube[2::4,:,:]
    arr_s4 = data_cube[3::4,:,:]
    return arr_s1, arr_s2, arr_s3, arr_s4

def split_data_roll(dot_data, roll=0):
    return [dot_data[i::4,(i+roll)%4] for i in range(4)]

def get_hdr_from_dir(dir):
    dir_list = glob.glob("camwfs*.fits", root_dir=dir)
    dir_list.sort()
    hdr = get_hdr_from_file(dir + dir_list[0])
    return hdr   

def get_hdr_from_file(file):
    hdul = fits.open(file)
    hdr = hdul[0].header
    hdul.close()
    return hdr

def get_spark_params_selfRM(selfRM_f, log_csv):
    hdr = get_hdr_from_file(selfRM_f)
    date_rm = hdr['DATE']
    return get_spark_params_date(date_rm, log_csv)

def get_spark_params_date(date, log_csv):
    df_check = pd.read_csv(log_csv)
    idx_time = df_check.UT.searchsorted(date)
    df_time = df_check.iloc[[idx_time]]
    return df_time.to_numpy()[0]

def file_lister(file_path):
    dir_list = glob.glob("camwfs*.fits", root_dir=file_path)
    # check that it found files
    if not dir_list:
        return False
    dir_list.sort() # sort to make it in order
    return dir_list

# rolling calculations
def moving_average(a, n=3) :
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

#run the pool and save the data
def main(data_dir, lab_data, dir_calib=glob_dir_calib, dark=glob_dark, mask=glob_mask, ref=glob_ref):
    sp_data = Spark(data_dir, lab_data, dir_calib, dark, mask, ref)
    sp_lab = Spark(lab_data, lab_data, dir_calib, dark, mask, ref)
    f_list_data = sp_data.file_lister(sp_data.dir_data)

    data_splits_list = []

    for n in range(len(f_list_data)//int(sp_data.data_HZ*60)):
        ni = int(n*sp_data.data_HZ*60)
        print(f"START: Block {n} starting with file {ni}!")
        try:
            data_split_s = np.array(sp_data.dot_list_pool(n_start=ni, n=1800*60, n_workers=60))
            data_splits_list.append(data_split_s)
        except Exception as e:
            print(f"We had an error at file {ni}!")
            continue
        print(f"=> END: Block {n}")

    # make sure to save this result!
    np.save(data_dir.replace("camwfs/", "data_splits_list.npy"), data_splits_list)
    lab_split_list = np.array(sp_lab.dot_list_pool(n=int(sp_data.data_HZ*60), n_workers=10))
    np.save(data_dir.replace("camwfs/", "lab_splits_list.npy"), lab_split_list)


def main_chunk(data_dir, lab_data, dir_calib=glob_dir_calib, dark=glob_dark, mask=glob_mask, ref=glob_ref):
    sp_data = Spark(data_dir, lab_data, dir_calib, dark, mask, ref)
    sp_lab = Spark(lab_data, lab_data, dir_calib, dark, mask, ref)
    f_list_data = sp_data.file_lister(sp_data.dir_data)

    data_splits_list = []

    for n in range(len(f_list_data)//int(sp_data.data_HZ*60)):
        ni = int(n*sp_data.data_HZ*60)
        chunk_n = int(sp_data.data_HZ/4)
        print(f"START: Block {n} starting with file {ni}!")
        try:
            data_split_s = np.array(sp_data.dot_chunk_pool(chunk_n, 60, n_start=ni, n_workers=15))
            data_splits_list.append(data_split_s)
        except Exception as e:
            print(f"We had an error at file {ni}!")
            continue
        print(f"=> END: Block {n}")

    # make sure to save this result!
    np.save(data_dir.replace("camwfs/", "data_chunks_list.npy"), data_splits_list)
    lab_split_list = np.array(sp_lab.dot_list_pool(n=int(sp_data.data_HZ*60), n_workers=10))
    np.save(data_dir.replace("camwfs/", "lab_splits_c_list.npy"), lab_split_list)

def sky_chunk(data_dir, lab_data, dir_calib=glob_dir_calib, dark=glob_dark, mask=glob_mask, ref=glob_ref):
    sp_data = Spark(data_dir, lab_data, dir_calib, dark, mask, ref)
    f_list_data = sp_data.file_lister(sp_data.dir_data)

    data_splits_list = []

    for n in range(len(f_list_data)//int(sp_data.data_HZ*60)):
        ni = int(n*sp_data.data_HZ*60)
        chunk_n = int(sp_data.data_HZ/4)
        print(f"START: Block {n} starting with file {ni}!")
        try:
            data_split_s = np.array(sp_data.dot_chunk_pool(chunk_n, 60, n_start=ni, n_workers=15))
            data_splits_list.append(data_split_s)
        except Exception as e:
            print(f"We had an error at file {ni}!")
            continue
        print(f"=> END: Block {n}")

    # make sure to save this result!
    np.save(data_dir.replace("camwfs/", "data_chunks_list.npy"), data_splits_list)
    return

def lab_archive(dir_lab, dir_calib=glob_dir_calib, dark=glob_dark, mask=glob_mask, ref=glob_ref, remove=True):
    if not os.path.isdir(dir_lab):
        print("There is no: ", dir_lab)
        return
    print(dir_lab)
    lab_dotseries_f = dir_lab.replace("camwfs/", "lab_splits_total_list.npy")
    if os.path.isfile(lab_dotseries_f):
        print(lab_dotseries_f, " is already created, moving on.")
        return

    sp_lab = Spark(dir_lab, dir_lab, dir_calib, dark, mask, ref)
    f_list_data = sp_lab.file_lister(sp_lab.dir_data)

    data_splits_list = []
    ten_sc_int = int(sp_lab.data_HZ*10)
    n_files = int(sp_lab.data_HZ*60)
    block_iter = len(f_list_data)//(n_files) + 1
    
    for n in range(block_iter): #todo; change
        ni = n*n_files
        print(f"START: Block {n} of {block_iter} starting with file {ni}!")
        try:
            data_split_s = np.array(sp_lab.dot_list_pool(n=n_files, n_workers=10))
            data_splits_list.append(data_split_s)
        except Exception as e:
            print(f"We had an error at file {ni}!")
            continue
        print(f"=> END: Block {n}")

    # make sure to save this result!
    lab_dotseries_f = dir_lab.replace("camwfs/", "lab_splits_total_list.npy")
    np.save(lab_dotseries_f, np.vstack(data_splits_list))

    # now save the reference
    fits_file = dir_lab.replace("camwfs/", "lab_ref.fits")
    fits.writeto(fits_file, sp_lab.labref_norm, overwrite=True)

    if remove:
        print("Removing all files over 10s: ..")
        # delete all but 1s of data:
        for f in f_list_data[ten_sc_int:]:
            os.remove(dir_lab+f)
        print(f"Done! {len( f_list_data[ten_sc_int:])} files deleted.")
    return


### PLOTS

def plot_dotseries(dot_series, ylim_top=0.001, ylim_bottom=-0.001, title="Dot Series", save=False, savefile="plot.png"):
    colors = pl.cm.tab20(np.arange(4))

    # Redid the ref code, now checking the series
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12,8), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(title, y=0.95)

    for e in range(4):
        for i in range(4):
            axs[e].plot(dot_series[e::4,i], lw=2, alpha = 0.5, label = f'ref {i}', c = colors[i])
        axs[e].set_ylabel(f"sparkle split {e}")
        axs[e].set_ylim(top=ylim_top, bottom=ylim_bottom)

    plt.legend()
    plt.xlabel('frame of split')
    if save:
        plt.savefig(savefile, dpi=200)
    else:
        plt.show()
    return

def plot_dotseries_diag(dot_series, lab_path, ylim_top=0.01, ylim_bottom=0, title="Dot Series", save=False, savefile="plot.png"):
    colors = pl.cm.tab20(np.arange(4))

    # Redid the ref code, now checking the series
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,3), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(title, y=0.99)

    #generating t axis 
    hz = get_Hzs(lab_path)
    frame_n = dot_series.shape[0]/4
    total_s = frame_n/hz
    x_axis = np.arange(0, total_s, total_s/frame_n)

    for e in range(4):
        axs.plot(x_axis, dot_series[e::4,e], lw=2, alpha = 0.7, label = f'ref {e}', c = colors[e])
        #axs.set_ylabel(f"sparkle split {e}")
        axs.set_ylim(bottom=ylim_bottom, top=ylim_top)

    plt.legend()
    plt.xlabel('seconds (s)')
    if save:
        plt.savefig(savefile, dpi=200)
    else:
        plt.show()
    return