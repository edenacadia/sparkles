# pca_pipeline
# 2/25/2026
# written so I can run pca scripts independently
import sparkles.spark_xrif_pca as spca
import datetime
import re

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


def run_main(obs_str_test, lab_str_test, redu_saves):
    dark_dir = '/opt/MagAOX/calib/camwfs-dark/'
    lab_dark = dark_dir + 'camwfs-dark_bin2_2000.000000_10.000000_-45.000000__T20240330051127513106835.fits'
    sky_dark = dark_dir + 'camwfs-dark_bin2_1800.000000_600.000000_-45.000000__T20240324234326026219038.fits'
    # disect strings
    obs_str_test = "2025-12-03_052203_beta_pic_piaa"
    sky_dt, sky_obs = dir_to_lookyloo(obs_str_test)
    lab_str_test = "2025-12-03_042134_tau_ceti_lab_efc_unsats"
    lab_dt, lab_obs = dir_to_lookyloo(lab_str_test, target_name="tau_ceti")
    # set up with redu options
    sparkPCA = spca.SparkXrif(sky_dark, lab_dark, spca.glob_dir_calib, spca.glob_mask, spca.glob_ref)
    # might need more options here
    sparkPCA.set_data(sky_obs, lab_obs, sky_dt, lab_dt)

    # TODO: save the PCA images

    # LARGE REDUCTIONS
    # project onto all the data

    # normalize those projections

    # TODO: save those values
    # TODO: save those plots