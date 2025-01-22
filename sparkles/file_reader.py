# file_reader.py
# Eden McEwen
# May 2023
## This code will run the basic file conection, ref, and sorting

import os
import glob
import pandas as pd
from astropy.io import fits


def file_sample(file, path):
    """
    Take file name, pull, sample value, return
    """
    f_all = path + file
    #check to make sure path exists
    if not os.path.isfile(f_all):
        print(f"WARNING! {f_all} not found")
        return None
    # Open file
    try:
        with fits.open(f_all) as hdul:
            #frame = hdul[0].header['FRAMENO']
            #wrt = hdul[0].header['WRTSEC'] + hdul[0].header['WRTNSEC']*10**(-9)
            data = hdul[0].data
    except Exception as e:
        print(f"WARNING! {e}")
        return None
    # Return other metrics
    return data

def file_lister(file_path):
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

# retrieve values from a file
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
    return param_dict

def get_Hz(dir):
    """ Pulling the frequency from the fiven directory """
    hdr = get_hdr_from_dir(dir)
    fps = hdr["HIERARCH CAMWFS FPS"]
    print(f"HZ value: {fps}")
    return fps

def get_datetime_start(dir):
    hdr = get_hdr_from_dir(dir)
    date = hdr["DATE-OBS"]
    print(f"DATETIME start: {date}")
    return date

def get_spark_params_selfRM(selfRM_f, log_csv):
    hdr = get_hdr_from_file(selfRM_f)
    date_rm = hdr['DATE']
    print(date_rm)
    return get_spark_params_date(date_rm, log_csv)

def get_spark_params_date(date, log_csv):
    try:
        df_check = pd.read_csv(log_csv)
        idx_time = df_check.UT.searchsorted(date)
        df_time = df_check.iloc[[idx_time]]
        return df_time.to_numpy()[0]
    except Exception as e:
        print("error: ", e)
        #returning the first index
        df_time = df_check.iloc[[-1]]
        return df_time.to_numpy()[0] 
