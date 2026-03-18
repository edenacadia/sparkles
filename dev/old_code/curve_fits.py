## curve_fits.py
# Eden McEwen
# April 4 2023

from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from scipy import optimize

def find_freq(y_data, plot=True, title="Peridogram"):
    y_data = y_data - np.average(y_data)
    y_data = np.hstack([y_data, np.zeros(10000)])
    f, Pxx_den = signal.periodogram(y_data)
    max_f = f[np.argmax(Pxx_den)]
    if plot:
        plt.semilogy(f, Pxx_den, label = "periodogram")
        plt.ylim([1e-12, 1e-1])
        plt.axvline(max_f, c='m',label=f"guess = {1/max_f:0.02f}")
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.yscale('log')
        plt.xscale('log')
        plt.title(title)
        plt.legend()
        plt.show()
    return max_f

## The function to fit
def test_func_fix(max_f, x, a, ph, off):
    return a * np.sin(2*np.pi*max_f * x + ph) + off

def simple_fit(x_start, x_end, y_data_all, po_guess=[0.0001, 0.1, 0.0003], plot=True):
    # Setting up the data ranges
    x_data = np.arange(x_end-x_start)
    y_data = y_data_all[x_start:x_end]
    # find and set the frequency
    max_f = find_freq(y_data, plot=False)
    func_p = partial(test_func_fix, max_f)

    params, params_covariance = optimize.curve_fit(func_p, x_data, y_data, p0=po_guess)
    print(params)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.scatter(x_data, y_data, label='Data', alpha=0.5)
        plt.plot(x_data, test_func_fix(max_f, x_data, params[0], params[1], params[2]),
                label='Fitted function', c ='m')
        plt.legend(loc='best')
        plt.show()

    return params

def window_fit(y_data, n_hz, po_guess=[0.0001, 0.1, 0.0003]):
    # NOTE: data should be sent to this function pre-cropped
    data_length = y_data.shape[0]
    sin_fits = []
    sin_cov = []
    sin_std = []
    max_f = find_freq(y_data, plot=False)
    
    func_p = partial(test_func_fix, max_f)
    for n in range(data_length-n_hz):
        x_data = np.arange(n_hz)
        y_data_tmp = y_data[n:n+n_hz]
        params, params_covariance = optimize.curve_fit(func_p, x_data, y_data_tmp, p0=po_guess)
        sin_fits.append(params)
        sin_cov.append(params_covariance)
        sin_std.append(np.sqrt(np.diag(params_covariance)))
        po_guess = params
    sin_fits = np.array(sin_fits)
    sin_cov = np.array(sin_cov)
    sin_std = np.array(sin_std)

    return sin_fits, sin_std

def window_fits_all(x_start, x_end, dot_data, n_hz):
    # iteratively assign starting guesses
    fits_all = []
    for ii in range(4):
        jj_tmp = []
        for jj in range(4):
            y_data = dot_data[x_start:x_end, ii, jj]
            off_t = np.average(y_data)
            amp_t = np.max(y_data) - np.min(y_data)
            po_t  = [amp_t, 0.1, off_t]
            sin_fits, sin_std = window_fit(y_data, n_hz, po_guess=po_t)
            jj_tmp.append(sin_fits)
        fits_all.append(jj_tmp)
    return np.array(fits_all)

def plot_fits(sin_fits, sin_std, title): 
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,8), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.suptitle(title + "fitting params, fixed freq", y=0.92)

    param_names = ["amp", "phase", "off"]

    for e in range(3):
        axs[e].plot(sin_fits[:,e])
        axs[e].fill_between(np.arange(sin_fits.shape[0]), sin_fits[:,e]-sin_std[:,e], sin_fits[:,e]+sin_std[:,e], alpha=0.2)
        axs[e].set_ylabel(param_names[e])

    plt.xlabel('time (s),rolling chunk')
    plt.show() 