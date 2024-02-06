# spark_plots.py
## Eden McEwden
# plotting common file things

import matplotlib.pylab as pl
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np

from sparkles.file_reader import *

def plot_selfRM_gains(sky_selfRM, lab_selfRM, title = '', RM_dir = "/home/eden/data/self_RMs/"):
    selfRM_sky = fits.open(RM_dir + sky_selfRM)[0].data
    selfRM_lab = fits.open(RM_dir + lab_selfRM)[0].data

    gain_series = np.diag(selfRM_sky[5])/np.diag(selfRM_lab[5])
    gain_avg = np.average(gain_series)

    # TODO: read the selfRM

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 6), facecolor='white')
    fig.suptitle("SelfRM raw and ratios /n " + title, fontsize=18, y=0.95)

    axs[0].set_ylabel("Raw SelfRM")
    axs[0].plot(np.diag(selfRM_sky[5]), label="data", alpha=0.5)
    axs[0].plot(np.diag(selfRM_lab[5]), label="lab", alpha=0.5)
    axs[0].legend()

    axs[1].set_ylabel("Ratio SelfRM")
    axs[1].plot(gain_series, label="data/lab", alpha=0.5)
    axs[1].axhline(gain_avg, label=f"gain avg = {gain_avg:0.4f}")
    axs[1].legend()
    plt.show()
    return


def plot_dotSeries_all(title, avg_dot, min = -0.0005, max=0.0005):
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12,8), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(f"{title}: dot series 1s avg", y=0.92)

    for e in range(4):
        for i in range(4):
            axs[e].plot(avg_dot[:,e,i], lw=2, alpha = 0.5, label = f'ref {i}')
        axs[e].set_ylabel(f"sparkle split {e}")
        axs[e].set_ylim(top=max, bottom=min)

    plt.legend()
    plt.xlabel('seconds')
    plt.show()
    return


def plot_dotSeries_avg():
    return


def plot_sotSeries_pick():
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
    hz = get_Hz(lab_path)
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
