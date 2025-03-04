# Eden McEwen
# err_budget.py
# in order to determine the expected error for MagAO-X
import numpy as np
import functools

def calc_strehl(s_mag=5, r_o=0.16, n_act = 1564 , Hz = 2000, rms_jitter = 0, v_wind=10, OG=1):
    # Inputs that are mostly constant:
    t_int = 1/Hz
    t_s_lag = 0.00075 + t_int #s usually 10 time control loop speed, control bandwidth
    D_tel=6.5
    lam_ro = 500 #nm
    lam_wave = 810 # nm
    ang_zenith = 0 #ang
    # Caluclations
    airmass = 1 / np.cos(ang_zenith * np.pi / 180) # r_o comes pre scaled, not updating 
    seeing_550 = 0.1/r_o
    ro_scaled = r_o * (lam_wave / lam_ro)**(6/5) * airmass**(-3/5)
    D_div_ro = D_tel / ro_scaled
    frq_GW = 0.43 * v_wind / ro_scaled
    # actuator spacing
    n_act_acr = n_act**(1/2)
    act_acr_ap = 2*np.sqrt(n_act/np.pi)
    act_spacing = D_tel / act_acr_ap
   
    # magnitude calc
    mag0_flux = 452 # p / cm^2 / s / Ang
    area = (D_tel*100)**2 * np.pi / 4 #cm^2
    delt_lam = 3570 # ang
    throughput = 0.1 # div b y sqrt 2  for excess noise calculation
    photons = mag0_flux * area * delt_lam * throughput
    # photons should be about 4.2e10 phot per sec hit 0 mag, from website
    phot_flux = photons*t_int*10**(-s_mag/2.5)
    # err in rad for strehl
    err_fit_rad = calc_err_fit_rad(act_spacing, ro_scaled)
    err_temp_rad = calc_err_temp_rad(t_s_lag, frq_GW)
    err_phot_rad = calc_err_phot_rad(n_act, phot_flux, OG=OG)
    # what's going on here
    #print(f"MAG {s_mag} r {r_o:0.4} and ro_scaled {ro_scaled:0.4} ")
    #print(f"fit {err_fit_rad:0.3} temp {err_temp_rad:0.3} phot {err_phot_rad:0.3}")
    # calc Strehl from rad err
    err_tot_sq = err_fit_rad**2 + err_temp_rad**2 + err_phot_rad**2
    strehl = np.exp(-err_tot_sq)
    strehl_tt = 1 #calc_strehl_tt(rms_jitter, D_tel, lam_wave) # not using rn
    return strehl*strehl_tt

def calc_strehl_iter(s_mag=5, r_o=0.16, n_act=1600, Hz=2000, rms_jitter=0, v_wind=15):
    # thee might need to be 
    iter_limit = 2
    d_strehl_limit = 0.01
    # for each iteration, use strehl from previouss
    old_strehl = 1 # shart by assuming no strehl issue
    strehl_diff = 1
    k=0
    new_strehl=1
    calc_strehl_vars = functools.partial(calc_strehl, s_mag=s_mag, r_o=r_o, n_act=n_act, Hz=Hz,  rms_jitter=rms_jitter, v_wind=v_wind)
    while (k < iter_limit and strehl_diff > d_strehl_limit):
        new_strehl  = calc_strehl_vars(OG=old_strehl)
        #now can update variables
        strehl_diff = old_strehl - new_strehl
        old_strehl = new_strehl
        k += 1
        #print(f'iter {k}: Strehl {new_strehl}, diff {strehl_diff}')
    return new_strehl

# Error Fitting
def calc_err_fit_rad(act_space, ro_scaled):
    # 0.28 continuous phase sheet
    err = 0.28 * (act_space / ro_scaled)**(5/3)
    return err**0.5

def calc_err_fit_nm(act_space, ro_scaled, lam_wave):
    err_rad = calc_err_fit_rad(act_space, ro_scaled)
    return lam_wave*err_rad / (2*np.pi)

# Error temporal
def calc_err_temp_rad(t_s_lag, f_o):
    t_o = 0.134 / f_o # Fried 1990
    err = (t_s_lag/t_o)**(5/3)
    return err**0.5

def calc_err_temp_nm(t_s_lag, t_o, lam_wave):
    err_rad = calc_err_temp_rad(t_s_lag, t_o)
    return lam_wave*err_rad / (2*np.pi)

# Error photon noise
def calc_err_phot_rad(n_act, phot_flux, OG=1):
    # extra beta_p term from pywfs term
    beta_p = 2*np.sqrt(2) / np.sqrt(OG)
    err = beta_p**2 * n_act / phot_flux
    return err**0.5

def calc_err_phot_nm(n_act, phot_flux, lam_wave, OG=1):
    err_rad = calc_err_phot_rad(n_act, phot_flux, OG=OG)
    return lam_wave*err_rad / (2*np.pi)

# Error from TT residuals
def calc_strehl_tt(rms_jitter, D_tel, lam_wave):
    #print("RMS Jitter [mas]: ", rms_jitter)
    k = 0.98 # from central obstruction
    coeff  =  8*np.log(2) / k**2
    psf_width_rad = (lam_wave*1e-9 /  D_tel)
    psf_width_mas  = psf_width_rad * 206265 * 1000
    #print("PSF FWHM in [mas]: ", psf_width_mas)
    gauss = rms_jitter / psf_width_mas
    denom = 1 + coeff*(gauss**2)
    strehl_tt = 1/denom
    #print("Strehl tt: ", strehl_tt)
    return strehl_tt

# seeing conversion functions
def ro_to_seeing_dumb(r_o):
    return 0.1/r_o

def ro_to_seeing(r_o, lamda=500):
    seeing_rad = 0.98*lamda*1e-9/r_o
    factor_as = 60*60*180/(np.pi) 
    factor = 206265
    return seeing_rad*factor

def seeing_to_r_o(seeing, lam_seeing=500):
    as_factor = 1/206265
    r_o = (0.98*lam_seeing*1e-9) / (as_factor*seeing)
    return r_o

def el_scaling_DIMM(el, DIMM):
    # change elevation to zenith
    z = 90 - el 
    # take this as an angle
    c_mod_ro = (np.cos(np.deg2rad(z)))**(3/5)
    # turn DIMM to r_o
    ro = seeing_to_r_o(DIMM)
    # take this coefficient, turn into 
    ro_mod = ro*c_mod_ro
    # convert back for dimm
    DIMM_mod = ro_to_seeing(ro_mod)
    # print intermediate
    #print(f'z: {z}, \n c_mod_ro: {c_mod_ro}, \n ro_mod: {ro_mod}, \n DIMM: {DIMM}, \n DIMM_mod: {DIMM_mod}')
    # return the modifed  DIMM
    return DIMM_mod

def opt_modes(s_mag, r_o, Hz, v_wind):
    modes = np.arange(0, 1564)
    SR_list = [calc_strehl(r_o = r_o, n_act = m, s_mag = s_mag, Hz=Hz, v_wind=v_wind) for m in modes]
    SR = np.max(SR_list)
    n_mode = np.argmax(SR_list)
    return  SR, n_mode