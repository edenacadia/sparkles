# log_readers.py
# Eden McEwen

import os
import pandas as pd
from astropy.io import fits
import re

#### MAKING a dataframe from telem/log ####

def make_teldump_df(file_teldump):
    # define a dataframe with columns
    file = open(file_teldump,'r')
    rows=[]
    # iterate through the lines of the text file
    i = 0 
    for line in file.readlines():
        if re.search(r"TELM \[speckles\] modulating at", line):
            m = re.search(r"TELM \[speckles\] modulating at ([-+]?[0-9]+\.[0-9]+)? Hz seps\: ([-+]?[0-9]+\.[0-9]+)? angs\: ([-+]?[0-9]+\.[0-9]+)? amps\: ([-+]?[0-9]+\.[0-9]+)? \+", line)
            e_time = line[0:19]
            try:
                e_freq = float(m.group(1))
                e_seps = float(m.group(2))
                e_angs = float(m.group(3))
                e_amps = float(m.group(4))
            except Exception as e:
                print(f"Error at: {line}, \n {e}")
                continue
            df_new_row = pd.Series({'UT': e_time, 'MOD': True, 'TRIG': False, 'HZ': e_freq, 
                        'SEPS': e_seps, 'ANGS': e_angs, 'AMPS': e_amps})
            rows.append(df_new_row)
        elif re.search(r"TELM \[speckles\] modulating by", line):
            m = re.search(r"TELM \[speckles\] modulating by trigger seps\: ([-+]?[0-9]+\.[0-9]+)? angs\: ([-+]?[0-9]+\.[0-9]+)? amps\: ([-+]?[0-9]+\.[0-9]+)? \+", line)
            e_time = line[0:19]
            try:
                e_seps = float(m.group(1))
                e_angs = float(m.group(2))
                e_amps = float(m.group(3))
            except Exception as e:
                print(f"Error at: {line}, \n {e}")
                continue
            df_new_row = pd.Series({'UT': e_time, 'MOD': True, 'TRIG': True, 'HZ': None, 
                                    'SEPS': e_seps, 'ANGS': e_angs, 'AMPS': e_amps})
            rows.append(df_new_row)

        elif re.search(r"TELM \[speckles\] not modulating", line):
            e_time = line[0:19]
            df_new_row = pd.Series({'UT': e_time, 'MOD': False, 'TRIG': False, 'HZ': None, 
                                    'SEPS': None, 'ANGS': None, 'AMPS': None})
            rows.append(df_new_row)
    df = pd.DataFrame(rows)
    return df

def make_teldump_csv(file_teldump, new_file=""):
    #pulling from df creation funciton
    df = make_teldump_df(file_teldump)
    # saving the new file
    if new_file == "":
        new_file = file_teldump.replace(".txt", "_clean.csv")
    df.to_csv(new_file,  index=False)
    # return the name of wher

    return new_file

def query_teldump_csv(file_teldump, date_time_q):
    # taking a csv file and query with a datetime string
    df_dump = pd.read_csv(file_teldump)
    print(df_dump)
    idx_time = df_dump.UT.searchsorted(date_time_q)
    df_time = df_dump.iloc[[idx_time]]
    #return to dict
    return df_time.to_dict(orient='records')

def query_teldump_csv_selfRM(self_RM_file, file_teldump):
    try:
        hdr = get_hdr_from_file(self_RM_file)
    except Exception as e:
        print(self_RM_file, e)
        return pd.DataFrame()
    date_time_q = hdr['DATE']
    return query_teldump_csv(file_teldump, date_time_q)
    
def get_hdr_from_file(file):
    hdul = fits.open(file)
    hdr = hdul[0].header
    hdul.close()
    return hdr

def print_sparkle_params(param_dict):
    for key in param_dict:
        print(key, ":" , param_dict[key], " " ,end = '\t')
    print('\b')
    return param_dict

def make_dimm_df(file_dimmdump):
    # define a dataframe with columns
    file = open(file_dimmdump,'r')
    rows=[]
    # iterate through the lines of the text file
    i = 0 
    for line in file.readlines():
        if re.search(r"TELM \[telsee\] dimm", line):
            dimm = re.search(r"TELM \[telsee\] dimm\[ t: (\w+) el: ([-+]?[0-9]+\.[0-9]+)? fw: ([-+]?[0-9]+\.[0-9]+)? fw-cor: ([-+]?[0-9]+\.[0-9]+)?", line)
            baade = re.search(r"mag1\[ t: (\w+) el: ([-+]?[0-9]+\.[0-9]+)? fw: ([-+]?[0-9]+\.[0-9]+)? fw-cor: ([-+]?[0-9]+\.[0-9]+)?", line)
            e_time = line[0:19]
            d_seeing = float(dimm.group(3))
            d_seeing_corr = float(dimm.group(4))
            b_seeing = float(baade.group(3))
            b_seeing_corr = float(baade.group(4))
            df_new_row = pd.Series({'UT': e_time, 'DSEEING': d_seeing, 'DSEEING_COR': d_seeing_corr, 'BSEEING': b_seeing, 'BSEEING_COR': b_seeing_corr})
            rows.append(df_new_row)
    df = pd.DataFrame(rows)
    # saving the new file
    return df

def make_dimm_csv(file_dimmdump, new_file=""):
    #pulling from df creation funciton
    df = make_dimm_df(file_dimmdump)
    # saving the new file
    if new_file == "":
        new_file = file_dimmdump.replace(".txt", "_clean.csv")
    df.to_csv(new_file,  index=False)
    # return the name of wher
    return new_file