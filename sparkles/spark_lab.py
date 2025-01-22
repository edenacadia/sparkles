# stuff for gettign through lab data
from astropy.io import fits
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import tempfile
import imageio
from scipy import signal
from importlib import reload
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pylab as pl
# super special:
import sparkles.spark as spkl
import sparkles.spark_plots as sp


# function that processess fits and saves the time series dot product as npy 
def proc_lab_data(spkl_obj, lab_dir, filename = "syncspark", workers=20, fps=2000):
    # what makes a mintue of data:
    dir_data = spkl_obj.dir_data
    data_splits_list = []
    # processing and saving the files:
    f_list = spkl_obj.file_lister(dir_data)
    n_files = len(f_list)
    n_blocks = n_files//(fps*workers)
    print(f"File count: {n_files}, Block size: {fps*workers}, Blocks: {n_blocks+1}")
    # iter over data block chunks
    for b in range(n_blocks + 1):
        ni = b*fps*workers # file number is the block number * files per worker * no of workers
        print(f"START: Block {b} starting with file {ni}!")
        data_split_s_lab = np.array(spkl.split_data_roll(np.array(spkl_obj.dot_list_pool(n_start=ni, n=fps*workers, n_workers=workers))))
        data_splits_list.append(data_split_s_lab)
        try:
            continue
        except Exception as e:
            print(f"We had an error at file {ni}!")
            print(e)
            continue
        print(f"=> END: Block {b}")
    # make sure to save this result!
    data_splits_stack = np.hstack(data_splits_list)
    file_path = f"{lab_dir}/data_splits_stack_{filename}.npy"
    np.save(file_path, data_splits_stack)
    print(f"SAVED: {file_path}")
    return data_splits_stack

# function that processess fits and saves the time series dot product as npy 
def proc_lab_data_roll(spkl_obj, lab_dir, filename = "syncspark", workers=20, fps=2000):
    # what makes a mintue of data:
    dir_data = spkl_obj.dir_data
    file_path = f"{lab_dir}/data_roll_stack_{filename}.npy"
    # check if this exists:
    if os.path.isfile(file_path):
        print(f'We did this already: {file_path}')
        #return np.load(file_path)
    # processing and saving the files:
    f_list = spkl_obj.file_lister(dir_data)
    n_files = len(f_list)
    n_blocks = n_files//(fps*workers)
    print(f"File count: {n_files}, Block size: {fps*workers}, Blocks: {n_blocks+1}")
    # roll indicies
    data_roll_stack = []
    for i in range(4):
        # iter over data block chunks
        print(f"Start roll orientation {i}")
        data_splits_list = []
        for b in range(n_blocks + 1):
            ni = b*fps*workers # file number is the block number * files per worker * no of workers
            print(f"START: Block {b} starting with file {ni}!")
            data_split_s_lab = np.array(spkl.split_data_roll(np.array(spkl_obj.dot_list_pool(n_start=ni, n=fps*workers, n_workers=workers)), roll=i))
            data_splits_list.append(data_split_s_lab)
            try:
                continue
            except Exception as e:
                print(f"We had an error at file {ni}!")
                print(e)
                continue
            print(f"=> END: Block {b}")
        # make sure to save this result!
        data_splits_stack = np.hstack(data_splits_list)
        data_roll_stack.append(data_splits_stack)
    file_path = f"{lab_dir}/data_roll_stack_{filename}.npy"
    np.save(file_path, np.array(data_roll_stack))
    print(f"SAVED: {file_path}")
    return data_roll_stack

# plotting the results of reducing data
def plot_time_series(data_split_stack, file_loc, plot_name, hz = 2000, total_s = -1, params=True):
    #using the file directory to pull 
    if params == True:
        spark_params = spkl.get_spark_params(file_loc)
        spark_param_print = " ".join([key + ':'+ str(spark_params[key]) + "," for key in spark_params])
    else:
        spark_param_print = ' '
    # set up plot
    colors = pl.cm.tab20(np.arange(4))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,3), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(f"{plot_name} \n {spark_param_print}", y=0.99)
    #generating t axis 
    frame_n = data_split_stack.shape[1]
    if total_s == -1:
        total_s = 4*frame_n/hz
    x_axis = np.arange(0, total_s, total_s/frame_n)
    #plot each split in a complementary color
    for e in range(4):
        axs.plot(x_axis, data_split_stack[e, :], lw=2, alpha = 0.7, label = f'ref {e}', c = colors[e])
        #axs.set_ylabel(f"sparkle split {e}")
        #axs.set_ylim(bottom=0, top=0.005)
    plt.legend()
    plt.xlabel('seconds (s)')
    return plt

def plot_time_series_smoothed(data_split_stack, file_loc, plot_name, n_avg = 1000, hz = 2000, total_s = -1):
    #using the file directory to pull 
    spark_params = spkl.get_spark_params(file_loc)
    spark_param_print = " ".join([key + ':'+ str(spark_params[key]) + "," for key in spark_params])
    
    #generating t axis 
    frame_n = data_split_stack.shape[1]
    if total_s == -1:
        total_s = 4*frame_n/hz
    x_axis = np.arange(0, total_s, total_s*n_avg/frame_n)
    #calculate the roling values:
    dot_avgs, dot_stds = spkl.return_rolling(data_split_stack, n=n_avg)
    # set up plot
    colors = pl.cm.tab20(np.arange(4))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,3), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(f"{plot_name} \n {spark_param_print}", y=0.99)
    #plot each split in a complementary color
    for e in range(4):
        dot_avgs, dot_stds = spkl.return_rolling(data_split_stack[e], n=n_avg)
        axs.fill_between(np.arange(dot_avgs.shape[0]), dot_avgs-dot_stds, dot_avgs+dot_stds, alpha=0.2, color = colors[e])
        axs.plot(dot_avgs, lw=2, label = f'split {e}, LAB', color = colors[e])
    axs.set_ylim(bottom=0, top=np.max(data_split_stack))
    plt.legend()
    plt.xlabel('seconds (s)')
    return plt

## ROLLED DATA

def plot_time_series_roll(data_roll_stack, file_loc, plot_name, n_avg = 1000, hz = 2000, total_s = -1, params=True):
    #using the file directory to pull 
    if params == True:
        spark_params = spkl.get_spark_params(file_loc)
        spark_param_print = " ".join([key + ':'+ str(spark_params[key]) + "," for key in spark_params])
    else:
        spark_param_print = ' '
    #generating t axis 
    frame_n = data_roll_stack.shape[2]
    if total_s == -1:
        total_s = 4*frame_n/hz
    x_axis = np.arange(0, total_s, total_s*n_avg/frame_n)
    # set up plot
    colors = pl.cm.tab20(np.arange(4))
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12,8), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(f"{plot_name} \n {spark_param_print}", y=0.99)
    #plot each split in a complementary color
    for f in range(4):
        for e in range(4):
            axs[f].plot(data_roll_stack[f, e, :], lw=2, alpha = 0.7, label = f'ref {e}', c = colors[e])
        axs[f].set_ylabel(f'Roll {f}')
        axs[f].set_ylim(bottom=np.min(data_roll_stack), top=np.max(data_roll_stack))
    axs[0].legend()
    plt.xlabel('seconds (s)')
    return plt

def plot_time_series_roll_smoothed(data_roll_stack, file_loc, plot_name, n_avg = 1000, hz = 2000, total_s = -1, params=True):
    #using the file directory to pull 
    if params == True:
        spark_params = spkl.get_spark_params(file_loc)
        spark_param_print = " ".join([key + ':'+ str(spark_params[key]) + "," for key in spark_params])
    else:
        spark_param_print = ' '
    #generating t axis 
    frame_n = data_roll_stack.shape[2]
    if total_s == -1:
        total_s = 4*frame_n/hz
    x_axis = np.arange(0, total_s, total_s*n_avg/frame_n)
    # set up plot
    colors = pl.cm.tab20(np.arange(4))
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12,6), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(f"{plot_name} \n {spark_param_print}", y=0.99)
    #plot each split in a complementary color
    for f in range(4):
        for e in range(4):
            #calculate the roling values:
            dot_avgs, dot_stds = spkl.return_rolling(data_roll_stack[f, e], n=n_avg)
            axs[f].fill_between(np.arange(dot_avgs.shape[0]), dot_avgs-dot_stds, dot_avgs+dot_stds, alpha=0.2, color = colors[e])
            axs[f].plot(dot_avgs, lw=2, label = f'split {f}, LAB', color = colors[e])
        axs[f].set_ylabel(f'Roll {f}')
        axs[f].set_ylim(bottom=0, top=np.max(data_roll_stack))
    axs[0].legend()
    plt.xlabel('seconds (s)')
    return plt