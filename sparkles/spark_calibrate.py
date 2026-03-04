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

glob_data_p = pathlib.Path('/data/rawimages/')
glob_telem_p = pathlib.Path('/srv/aoc/opt/MagAOX')

glob_dir_calib = '/home/eden/data/calib/'
glob_mask = 'aol1_wfsmask.fits'

class SparkCalibrate(object):

    def __init__(self, dir_calib, spark_sep, spark_ang, spark_amp, wfs_hz=2000):
        # where to store calibrations
        self.dir_calib  = dir_calib
        # TODO: check that this exists 
        # TODO: Update later to query the database 
        self.spark_sep = spark_sep
        self.spark_ang = spark_ang
        self.spark_amp = spark_amp
        self.wfs_hz = wfs_hz
        # create a subfolder based on sparkle params
        self.calib_folder = f"sep{spark_sep}_ang{spark_ang}_amp{spark_amp}_freq{wfs_hz}"
        self.calib_path = f"{self.dir_calib}/{self.calib_folder}"
        # check if this folder exists, if not, make it
        if pathlib.Path(self.calib_path).exists():
            print(f"Path exists: {self.calib_path}")
        else:
            print(f"making directory for this calibration: {self.calib_path}")
            pathlib.Path(f"{self.calib_path}").mkdir(parents=True, exist_ok=True)
        # TODO: make this more scalable? CACAO probably has a pointer for this
        self.mask_data = fits.open(glob_dir_calib + glob_mask)[0].data
        # The next steps are scripts, seperated in functions
        # set_data()
        # gen_lab_ref() 
        # save_reference()
        # there is still a lot of reliance of xrif pull methods

    def set_data(self, lab_obs, lab_dt, lab_obs_n=0):
        # Obs name, the datetime string we need for lookyloo
        self.lab_obs = lab_obs
        # setting a specific datetime better than setting an n
        self.lab_dt = lab_dt
        self.lab_obs_n = lab_obs_n
        self.lab_obs_span = verify_obs(lab_obs, lab_dt, n=lab_obs_n)
        return

    def gen_lab_ref(self, n_lab = 1024, n_start = 0, klip=3, plt_img=False):
        """ 
        Pull a bunch of files, take an average
        """
        print("GENERATING REFERENCE PCA basis")
        # TODO: Do I want to enforce number of files?
        file_lists = gen_file_list(self.lab_obs_span)
        print(f"Found {len(file_lists)} xrifs, total {len(file_lists)*512} files")
        # LAB+TRUE for file sampling, so using the Lab dark!
        data_ar, timing_ar = self.file_sample_n_clean(file_lists, n = n_lab, n_start = n_start)
        self.n_files = data_ar.shape[0]
        # creating the PCA basis
        Z_KL, Z_KL_img = self.pca_basis(data_ar, klip=klip)
        self.ref_pca = Z_KL
        self.ref_pca_img = Z_KL_img
        # TODO: save a plot of these images

        # Then, we're going to create a self reference, for each roll. 
        lab_proj = self.pca_projection(data_ar) # this is for all frames
        # shape: [N, klip]
        # averages by the sparkle pattern
        # These should be rolled later for best effect
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
        self.save_fits(self.ref_pca, f"{self.dir_calib}/{self.calib_folder}/ref_pca.fits")
        self.save_fits(self.ref_proj, f"{self.dir_calib}/{self.calib_folder}/ref_proj.fits")
        self.save_fits(self.ref_rms, f"{self.dir_calib}/{self.calib_folder}/ref_rms.fits")
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
        hdu.writeto(filename)
        return

    def file_sample_n_clean(self, file_lists, n, n_start=0, norm=True):
        data, timing = pull_n_files(file_lists, n, n_start=n_start)
        data_clean = self.file_clean(data, norm=norm)
        return data_clean, timing

    def file_clean(self, data, norm=True):
        """ Clean a single file 
        recently removed the ref
        TODO: subtract the average?
        """
        # not dark sub bc we're average subtracting later
        # Normalization standard proc.
        if norm:
            data_dark_sub_mask = data * self.mask_data
            data_normed = np.divide(data_dark_sub_mask, np.sum(data_dark_sub_mask)) # this sum should be one 
            data_return = data_normed
        else: 
            data_return = data
        return data_return

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
        projection = np.array([self.ref_pca.T @ ref for ref in reference])
        return projection

####### file pulling - only relevand for pulling observations

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

def gen_file_list(obs_span, device = 'camwfs'):
    '''
    takes in a lookyloo obs span
    will then get all the matching paths for the file lsits. 
    '''
    extension = 'xrif'
    datetime_newer = obs_span.begin
    datetime_older = obs_span.end
    # checking the files associated with
    glob_data_p_dev = glob_data_p / device 
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


def dir_to_lookyloo(obs_str, target_name="beta_pic"):
    # searching for the lookyloo format
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})_(.+)$', obs_str)
    if not m:
        raise ValueError(f"Unrecognized format: {obs_str!r}")
    # break down the date into the datetime format
    Y, M, D, hh, mm, ss, rest = m.groups()
    dt = datetime.datetime(int(Y), int(M), int(D), int(hh), int(mm), int(ss), tzinfo=datetime.timezone.utc)
    
    # take the target name out of the remaining string
    if target_name:
        # if you find this string, replace it. return what's left
        if rest.find(target_name) != -1:
            rmndr = rest.replace(target_name, "")
            return dt, rmndr[1:]
    # fallback: return the first token after the datetime
    first_token = rest.split('_')[-1]
    return dt, first_token