# spark_plots.py
## Eden McEwden



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


def plot_dotSeries_all():
    return


def plot_dotSeries_avg():
    return


def plot_sotSeries_pick():
