
# sprkls_pool.py
# Eden McEwen 
# pooling the sparkles average calculations
# doing this file by file for optimization

from astropy.io import fits
from functools import partial
import numpy as np
from multiprocessing import Pool
from math import ceil
import glob
import os

import tarfile

#from sparkles.file_reader import * # these assume not tar
from sparkles.spark import Spark

glob_dir_calib = '../data/calib/'
glob_mask = 'aol1_wfsmask.fits'
glob_ref = 'aol1_wfsref.fits'
glob_dark = 'camwfs-dark_bin2_2000.000000_600.000000_-45.000000__T20220422005142597501326.fits'

class SparkTar(Spark):
    Hz = 2000
    n_pool = 4

    def __init__(self, dir_data, dir_lab, dir_calib, file_dark, file_mask, file_ref, n_avg=10000, ref_norm = False):
        Spark.__init__(self, dir_data, dir_lab, dir_calib, file_dark, file_mask, file_ref, n_avg, ref_norm)
       
        # NAN mat
        self.mask_nan = self.mask_data.copy()
        self.mask_nan[self.mask_nan == 0] = np.nan
        # Getting freuency
        self.lab_HZ = get_Hz(dir_lab)
        self.data_HZ = get_Hz(dir_data)
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

    # def gen_lab_ref(self, n_lab = 100, n_start = 0) # use superclass
    
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
    
    # def gen_dot_lab_avgs(self, ref_self_dot_avg) # use superclass
    
    # def dot_list_pool(self, n=100, n_start = 0, n_workers=4) # use superclass

    def dot_list(self, n=100):
        """ This does the same as pool but without parallelism"""
        #TODO: testing
        f_list = file_lister(self.dir_data)
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
        f_list = file_lister(self.dir_data) #just for checking length
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
    
    # def dot_chunk(self, n_start, n,  path, norm=True) # use superclass

    # def dot_data(self, data) # use superclass
    
    # def dot_data_wo_ref(self, data, ref) # use superclass

    # def file_sample_n_clean(self, path, n, n_start=0, norm=True) # use superclass

    def file_sample_clean(self, file, path, norm=True):
        #TODO:
        """Pull and clean data"""
        data, frame, wrt = self.file_sample(file, path)
        return self.file_clean(data, norm=norm), frame, wrt
    
    def file_clean(self, data, norm = True):
        #TODO:
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

    def set_spark_params(self):
        #TODO:
        # get the lab spark params
        self.lab_spark_params = get_spark_params(self.dir_lab)
        print_sparkle_params(self.lab_spark_params)
        # get the data spark params
        self.data_spark_params = get_spark_params(self.dir_data)
        print_sparkle_params(self.data_spark_params)
        return
    
    def file_lister(self, file_path):
        """
        File lister
        takes in: 
            directory (string)
        returns:
            file_list (list) list of all files in given directory
        """
        ### TODO: make tar useable
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
    
    def file_sample(file, path):
        """
        Take file name, pull, sample value, return
        """
        ### TODO: make tar useable
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
        
########## CALC HELPER FUNCTIONS ##########

''''
def split_data(data_cube):
    arr_s1 = data_cube[::4,:,:]
    arr_s2 = data_cube[1::4,:,:]
    arr_s3 = data_cube[2::4,:,:]
    arr_s4 = data_cube[3::4,:,:]
    return arr_s1, arr_s2, arr_s3, arr_s4

def split_data_roll(dot_data, roll=0):
    return [dot_data[i::4,(i+roll)%4] for i in range(4)]

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
    '''