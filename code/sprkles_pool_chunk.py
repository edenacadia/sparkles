
# sparkle_pool.py
# Eden McEwen 
# pooling the sparkles average calculations

from astropy.io import fits
from matplotlib import pyplot as plt
import multiprocessing as mp
from scipy import signal
import numpy as np
import os

class Spark:
    Hz = 2000
    n_pool = 4

    def __init__(self, dir_data, dir_lab, dir_calib, file_dark, file_mask, file_ref, n_avg=10000):
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
        # setting None for specific check
        self.labref = self.gen_lab_ref()
        # TODO: set the right HZ number
        return 


    def file_sample_n_clean(n, dir_list, dir_path,  n_start=-1):
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
                self.data_clean
                
        return data_cube, n_start, frame_list, wrt_list



    
    def dot_main_pool(self, dir_data, len_avg, ref_redo=False):
        """
        Main fuction
        uses: 
            dir_data (string) directiory for datm, 
            dir_ref (string) directory for references, 
            len (int) seconds length for chunks to evaluate
        returns:
            dot_list (list) a list of the dot products of each individual frame
        => Notes: in this version I don't think I need to summarise by ten second averages becayse of the referene
            this means the pool action will happen just over the four split 
            another question is how to know which split ref to multiply onto which data cube
                might want to do all the four rolls to achieve this.
            split files by the splits
                this means that i can't take in the cubd form of the lookyloo script
                maybe need to ask avalon for her fits file sorting script
                    basically need this to check that each sequential file has the same header as the previous
        """
        #TODO: rolling the ref avg to account for the different splits
        print("START: Main sparkle dot")

        # call ref average creation
        print("=> Checking lab reference")
        n = 10000 # number of files to include in average
        if (ref_redo == True) or (self.lab_ref == None):
            self.labref = self.gen_lab_ref() # generates avg from lab
            
        # call file lister
        print("=> Checking lab reference")
        files_data = file_lister(self.dir_data) # list of all files in directory

        # Pool: pulling files and cleaning (easier in chunks)
        n_files = len(files_data)
        n_batch = n_files//self.n_pool
        # Start the pool
        pool = mp.Pool(self.n_pool)
        results_async = []
        print(f'calc_stats in parallel with {self.n_pool} cores.')
        for ni in range(self.n_pool):
            # seperating files into equal batched
            print(f"=> pulling from file {ni*n_batch} to file {(ni+1)*n_batch}")
            list_of_files = files_data[ni*n_batch:(ni+1)*n_batch]

            # call a pooled version of the file cleaning:
            results = pool.apply_async(self.file_pull_pool, (self, list_of_files))
            # adding in results?
            results_async.append([ni, results])
        # clean up the pool functions
        pool.close()
        pool.join()
        
        # sort and stack the results
        idx_list = results_async[:,0]
        data_list = results_async[:,1]
        data_sort = data_list[np.argsort(idx_list)]
        data_all = np.hstack(data_sort) # one numpy array

        # split into the four variations
        data_split = split_data(data_all)

        # Pool: on split, dot product calculation
         # Start the pool
        pool = mp.Pool(self.n_pool)
        dot_async = []
        ## We should use four cores since we'll always have 4 splits
        for n in range(4):
            # TODO: figure out how to match splits on each 
            rint(f"=> Dot product for split {n}")
            # call a pooled version of the file cleaning:
            results = pool.apply_async(self.dot_split_pool, (self, data_split[n]))
            # adding in results?
            dot_async.append([n, results])

            # dot pro
        pool.close()
        pool.join()

        return # dot_list

    def dot_split_pool(self, data_split, ref_mean):
        """
        Pool funtion
        input:
            data_split (string list) a list of all files in a split
        uses:
            self.labref (mat) the average frame from lab
        returns:
            dot_list_split (nparray) a list of the dot product values for each frame in the given file list
        """
        # TODO: this will be a one by one split? or does a cube split make more sense?

        return

    def gen_lab_ref(self, n=-1):
        """
        ref generation
        => This funtion will calculate the split average avgs 
        takes in:
            ref_files (string lists)
            len_avg (int)
        returns:
            split_average
        process: take a sample of len_avg length, sum,
        """
        # pick length of the ref averge:
        print("START: generating lab reference")
        if n == -1: n = self.n_avg
        else: self.n_avg = n # change the class variable
        
        # sample files from the list
        files_labref = file_lister(self.dir_ref) # list of all files in ref directory
        print(f"=> sampling {n} files")
        data_labref, n_start, frame_list, wrt_list = file_sample(n, files_labref, self.dir_ref,  n_start=-1) # pull data
        
        # check that these files are alright:
        print(f"=> Checking continuity from {n_start}... ")
        check = check_cont(frame_list, wrt_list, Hz = self.Hz)
        print(f"... continuity {check}") #TODO: make this more impressive
        
        # if check passes, take the four way average
        print("=> Cleaning lab reference ")
        data_labref_clean, _, mask_mat, _, _ = self.clean_data_cube(self, data_labref, self.dir_calib, self.f_dark, self.f_mask, self.f_ref)

        # NAN matrix for fun other things
        mask_nan = mask_mat.copy()
        mask_nan[mask_nan == 0] = np.nan

        # Average the lab reference
        print("=> Averaging lab reference")       
        mat_ref_sub_split = np.array(split_data(data_labref_clean))
        # taking average after multiplying on the nan mask
        mat_avg_split = np.nanmean(mat_ref_sub_split*mask_nan, a=1)

        print("END: generating lab reference")
        return mat_avg_split

    
    ########## MAIN organization files ##########

    # function, takes in data matrix, returns cleaned matrix
    def clean_data_cube(self, data_cube):
        """
        Clean data cube
        input: 
            data_cube (numpy array []) cube to be cleaned
        """
        data_dark = fits.open(self.dir_calib + self.f_dark)[0].data
        data_mask = fits.open(self.dir_calib + self.f_mask)[0].data
        data_ref = fits.open(self.dir_calib + self.f_ref)[0].data
        # generating matrix files
        # TODO: if trying to parallelize, do it here
        dark_mat =  np.repeat(data_dark[np.newaxis, :, :], len(data_cube), axis=0)
        mask_mat =  np.repeat(data_mask[np.newaxis, :, :], len(data_cube), axis=0)
        ref_mat =  np.repeat(data_ref[np.newaxis, :, :], len(data_cube), axis=0)
        # NORMALIZATION
        mat_dark_sub = data_cube - dark_mat
        mat_normed = np.divide(mat_dark_sub * mask_mat, np.sum(mat_dark_sub * mask_mat, axis=(1,2))[:,None,None])
        # BUG: our ref not 1 normalized, even with the mask, so forcing it to be normal here
        ref_normed = np.divide(ref_mat * mask_mat, np.sum(ref_mat * mask_mat, axis=(1,2))[:,None,None])
        mat_ref_sub = mat_normed - ref_normed
        #mat_ref_sub = mat_normed - ref_mat*mask_mat # if you didn't want the ref normed

        return mat_ref_sub, dark_mat, mask_mat, ref_mat, ref_normed 

    ########## POOL helper functions #############
    def file_pull_pool(self, file_list_split):
        """
        simplifies function calling for the pool functions
        """
        data_cube = self.file_sample_pool(self, file_list_split)
        data_cube_clean = self.file_clean_pool(self, data_cube)
        return np.array(data_cube_clean)

    def file_sample_pool(self, file_list_split):
        """
        file sample pool
            this function preloads the file sample to streamline pooling
        inputs:
            file_list_split (list string)
        returns:
            data_cube (nparray float) only the data
        """
        n = len(file_list_split)
        data_cube, _ , frame_list, wrt_list = file_sample(n, file_list_split, self.dir_path)
        if not check_cont(frame_list, wrt_list, self.Hz):
            print("WARNING: not continuous") # TODO: make this an actual error catch
        return data_cube

    def file_clean_pool(self, data_mat_pool):
        """
        file clean pool
            this function preloads the file sample to streamline pooling
        inputs:
            data_mat_pool (list string) the data from the pool file pull
        returns:
            data_cube_clean (nparray float) data from the pool cleaned
        """
        data_pool_clean, _, _, _, _ = self.clean_data_cube(self, data_mat_pool, self.dir_calib, self.f_dark, self.f_mask, self.f_ref)
        return data_pool_clean
                               


########## Support functions ##########
############# not in class ############

def split_data(data_cube):
    arr_s1 = data_cube[::4,:,:]
    arr_s2 = data_cube[1::4,:,:]
    arr_s3 = data_cube[2::4,:,:]
    arr_s4 = data_cube[3::4,:,:]
    return arr_s1, arr_s2, arr_s3, arr_s4

def file_lister(file_path):
    """
    File lister
    takes in: 
        directory (string)
    returns:
        file_list (list) list of all files in given directory
    exception:
        directory not found
        list of files not found
    """
    # check that the path exists
    if not os.path.ispath(file_path):
        return False
    # Collect all the files in the directory
    dir_list = os.listdir(file_path)
    # check that it found files
    if not dir_list:
        return False
    dir_list.sort() # sort to make it in order
    return dir_list

def file_sample(n, dir_list, dir_path, n_start=-1):
    """
    file sampling
    takes in: 
        n (int) number of files to sample
        dir_list (string list) files in a directory
        fir_path (stirng) prefix for files
        n_start (int, optional) if you want to specify where to start pulling files
    returns:
        data_cube (float, list list) all of the files
        n_start (int) the file number started on, if not specified
        frame_list (int, list) all the frames no. included from the pull
        wrt_list (float, list) all the frames times included from the pull
    """
    # randomly generate a number between 0 and len (dir list)
    if (n_start < 0) or (n_start > len(dir_list) - n):
        n_start = np.random.randint(len(dir_list) - n)
    # lists to store relevant params
    data_cube = []
    frame_list = []
    wrt_list = []
    cont = True
    # Iter from start to start+leng
    for file in dir_list[n_start:n_start+n]:
        with fits.open(dir_path+file) as hdul:
            frame_list.append(hdul[0].header['FRAMENO'])
            wrt_list.append(hdul[0].header['WRTSEC'] + hdul[0].header['WRTNSEC']*10**(-9))
            data_cube.append(hdul[0].data)
    data_cube = np.array(data_cube)
    return data_cube, n_start, frame_list, wrt_list

def check_cont(frame_list, wrt_list, Hz = 2000):
    """
    check continuity of a list of frames
    inputs:
        frame_list (int list) list of the frame numbers
        wrt_list (float list) list of the frame write times
        Hz (int) the data aquisition rate on the camera
    returns:
        True: good to use files
        False: some files have more than one frame appart or timing seperation
    """
    if np.max(np.diff(frame_list)) > 1: 
        return False
    if np.max(np.diff(wrt_list)) > 2/Hz :
        return False
    return True