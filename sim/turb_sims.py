from hcipy import *
from hcipy import atmosphere
from hcipy import fourier
import numpy as np
from matplotlib import pyplot as plt

# for cropping 

def window_field(field, new_grid, h, x0=0, y0=0):
    cropped_field = field.shaped[(y0-h//2):(y0+h//2), (x0 - h//2):(x0+h//2)]
    return Field(cropped_field.ravel(), new_grid)

# classic equations

def noll(r0, lam=500e-9):
    # pre-corrected WFE
    return np.sqrt(1.031*(6.5/r0)**(5/3) ) * (lam/(2*np.pi))

def hugens(r0, lam=500e-9):
    # fitting error, post corrected WFE
    return np.sqrt(0.28*(.135/r0)**(5/3) ) * (lam/(2*np.pi))

def inv_hugens(WFE_res, r_o_lam=500e-9):
    normed_WFE = (WFE_res / (r_o_lam/(2*np.pi)))**2
    inv = (1/0.135)*(normed_WFE/0.28)**(3/5)
    return 1/inv

# convenient conversion stuff

def seeing_to_r_o(seeing, lam_seeing=500):
    as_factor = 1/206265
    r_o = (0.98*lam_seeing*1e-9) / (as_factor*seeing)
    return r_o

def ro_to_seeing(r_o, lamda=500):
    seeing_rad = 0.98*lamda*1e-9/r_o
    factor_as = 60*60*180/(np.pi) 
    factor = 206265
    return seeing_rad*factor

def phase_to_WFE(phase, wave):
    return np.std(phase*wave/(2*np.pi))
    

def phase_to_WFE_messy(phase, wave):
    mean = np.mean(phase)*wave/(2*np.pi)
    return np.sqrt(np.mean((phase*wave/(2*np.pi) - mean)**2))

def ro_to_variance(r_o, Dtel, lam_ro, lam_wave):
    ro_scaled = r_o * (lam_wave / lam_ro)**(6/5)
    var = 1.03*(Dtel/ro_scaled)**(5/3)
    return var

def variance_to_ro(var, Dtel, lam_ro, lam_wave):
    # note variance should be in radians
    ro_scaled = Dtel*(1.03/var)**(3/5)
    r_o = ro_scaled * (lam_ro / lam_wave)**(6/5)
    return r_o

def err_rad_to_nm(err_rad, lam):
    err_nm = lam*err_rad / (2*np.pi)
    return err_nm

def err_nm_to_rad(err_nm, lam):
    err_rad = err_nm*(2*np.pi) / lam
    return err_rad

# simulating and filtering

def turb_phase_sl(r0, grid, ff, circ_ap, wavelength=800.0e-9, ao_eff=1.0, L0=25):
    cn2 = atmosphere.Cn_squared_from_fried_parameter(r0)
    layer = atmosphere.InfiniteAtmosphericLayer(grid, cn2, L0=L0)
    # Phase multiplied by telescope aperture
    phase = layer.phase_for(wavelength)
    phase_ms = (phase - np.mean(phase[circ_ap>0])) # mean subtracted
    phase_filt = np.real(ff.forward(phase_ms + 0j)) # low pass filter
    phase_res = phase_ms - ao_eff * phase_filt # HP filter
    phase_res *= circ_ap
    #phase_res *= wfe_res / np.std(phase_res[aperture>0])
    return phase, phase_filt, phase_res

def turb_phase(r0, grid, ff, circ_ap, wavelength=800.0e-9, ao_eff=1.0, L0=25):
    layers = atmosphere.make_las_campanas_atmospheric_layers(grid, r0=r0, L0=L0)
    ml_atmsph = atmosphere.MultiLayerAtmosphere(layers)
    # Phase multiplied by telescope aperture
    phase = ml_atmsph.phase_for(wavelength)
    phase_ms = (phase - np.mean(phase[circ_ap>0])) # mean subtracted
    phase_filt = np.real(ff.forward(phase_ms + 0j)) # low pass filter
    phase_res = phase_ms - ao_eff * phase_filt # HP filter
    phase_res *= circ_ap
    #phase_res *= wfe_res / np.std(phase_res[aperture>0])
    return phase, phase_filt, phase_res

def turb_phase_padded_crop(ro, grid, grid_pad, ff_pad, circ_ap_pad, h_reg, h_pad, ao_eff=1.0, L0=25, sl=False):
    if sl:
        phase, phase_filt, phase_res = turb_phase_sl(ro, grid_pad, ff_pad, circ_ap_pad, ao_eff=ao_eff, L0=L0)
    else:
        phase, phase_filt, phase_res = turb_phase(ro, grid_pad, ff_pad, circ_ap_pad, ao_eff=ao_eff, L0=L0)
    #cropping these down to the regular grid size
    phase_crop = window_field(phase, grid, h_reg, x0=h_pad//2, y0=h_pad//2)
    phase_filt_crop = window_field(phase_filt, grid, h_reg, x0=h_pad//2, y0=h_pad//2)
    phase_res_crop = window_field(phase_res, grid, h_reg, x0=h_pad//2, y0=h_pad//2)
    # return in the same order as before
    return phase_crop, phase_filt_crop, phase_res_crop

def turb_phase_PL_r0(r0, grid, ff, circ_ap, wavelength=800.0e-9, ao_eff=1.0):
    #TODO: this is wrongggg
    # Regular surface aberration
    wfe_res = np.sqrt(0.28*(.135/r0)**(5/3) ) * (500e-9/(2*np.pi))
    sa2 = SurfaceAberration(grid, wavelength, Dtel, exponent=-11/3)
    phase = sa2.surface_sag*circ_ap
    #phase processing
    phase_ms = (phase - np.mean(phase))*circ_ap # mean subtracted
    phase_filt = np.real(ff.forward(phase_ms + 0j)) # low pass filter
    phase_res = phase_ms - ao_eff * phase_filt # HP filter
    # scale to espected wfe
    phase_res *= wfe_res / np.std(phase_filt[aperture>0])
    phase *=  wfe_res / np.std(phase_filt[aperture>0])
    phase_filt *=  wfe_res / np.std(phase_filt[aperture>0])

    return phase, phase_filt, phase_res


def PL_phase_padded_crop():
    return 0