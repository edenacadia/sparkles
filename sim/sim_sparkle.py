# Eden McEwen
# March 2023

from hcipy import *
import numpy as np
from hcipy import atmosphere
from hcipy import fourier
# might not need these because they're for plotting
from matplotlib import animation, rc
from IPython.display import HTML
from astropy.io import fits

# degining some global variables
wavelength_wfs =  800.0e-9#650E-9#842.0E-9       # Sensing wavelength
telescope_diameter = 6.5        # Diameter of the Magellan Clay
zero_magnitude_flux = 3.9E10    # Zero maf star plux, in photos/s
stellar_magnitude = 0           # zero magnitude star

# Camera variables: # NOTE: multiplied by 2 for aliasing
num_pupil_pixels = 60*2           # has a distance of 60 pixels between the pupils
num_pupil_pixels_tel = 56*2       # number pixels used across the telescope
#double this to keep from aliasing 
# Pixels agross/ actual pixels multiplied by diameter gives you pupil diameter spacing?
pupil_grid_diameter = num_pupil_pixels / num_pupil_pixels_tel * telescope_diameter
spatial_resolution = wavelength_wfs / telescope_diameter
modulation = 3*spatial_resolution               #radius in lam/D

# Deformable Mirror Variables
num_actuators_across_pupil = 50
rcond = 1E-3 # reconstruction matrix value

#AO system parameters
delta_t = 1E-3          # integration time for exposures TODO: check with pywfs
leakage = 0.0           # for leaky integrator
gain = 0.5              # for leaky integrator


###########################################
######### AO System setup #################
###########################################

# Designing the Magellan Pupil
def create_mag_pupil():
    # Grid defined with number of puils, and then the diameter, scaled to our pixel ratios
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
    # Actual Pyramid doubles the pupil count and pixel 
    pwfs_grid = make_pupil_grid(2*num_pupil_pixels, 2 * pupil_grid_diameter)
    # taking the pupil grid, ovevrsampling by 6 to make up for the spiders pixelization
    magellan_aperture = evaluate_supersampled(make_magellan_aperture(), pupil_grid, 4)

    return pupil_grid, pwfs_grid, magellan_aperture

######### DM setup #################

def create_DM(pupil_grid):
    # this is meters per actuator
    actuator_spacing = telescope_diameter / num_actuators_across_pupil
    # influence functions given the same pupil gir, num actuators, and their spacing
    influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)
    # creating DM with the above configureations for interaction
    deformable_mirror = DeformableMirror(influence_functions)
    return deformable_mirror

def create_DM_fourier(pupil_grid):
    # this is meters per actuator
    actuator_spacing = telescope_diameter / num_actuators_across_pupil
    fourier_grid = make_pupil_grid(num_actuators_across_pupil, 2*np.pi/actuator_spacing)
    fourier_grid = fourier_grid.rotate(0.6)  #number in radians) # rotating? 
    # influence functions given the same pupil gir, num actuators, and their spacing
    fourier_influence_funcs = make_fourier_basis(pupil_grid, fourier_grid)
    # creating DM with the above configureations for interaction
    fourier_deformable_mirror = DeformableMirror(fourier_influence_funcs)
    return fourier_deformable_mirror

def create_DM_zern(pupil_grid, num_modes, aperature):
    # influence functions given the same pupil gir, num actuators, and their spacing
    zern_influence_funcs = make_zernike_basis(num_modes, pupil_grid_diameter, pupil_grid, 2) # 2 - start index, skips the piston, goes to TT
    # scale and zero non illuminated pixels
    zern_influence_funcs = ModeBasis([mode*(aperature>0) / np.std(mode[aperature>0]) for mode in zern_influence_funcs], pupil_grid)

    # creating DM with the above configureations for interaction
    zern_deformable_mirror = DeformableMirror(zern_influence_funcs)
    return zern_deformable_mirror

######### WFS setup #################

def create_mod_PWFS(pupil_grid, pwfs_grid, n_steps=12): # you might want to use more (20 lam over d . / 12 => have gaps)
    # creating a pyramid wfs on the grid of our pupil, seperated similarly to how our pupil itself is
    pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=pupil_grid_diameter, pupil_diameter=telescope_diameter, wavelength_0=wavelength_wfs, q=4)
    # Simulating OCAM as a noiseless detector
    camera = NoiselessDetector(pwfs_grid) # can bin here for less aliasing effects
    # Takes the defined wfs and modulates it through the mod code, modulation is a radius
    mpwfs = ModulatedPyramidWavefrontSensorOptics(pwfs, modulation, num_steps=n_steps)

    return mpwfs, camera
    
def create_wf(magellan_aperture):
    # Creating a wavefront from the constraints of the telescpe
    wf = Wavefront(magellan_aperture, wavelength_wfs)
    # normalize the power
    wf.total_power = 1

    return wf

def create_pwfs(mpwfs, wf):
    # propagate the wavefront 
    wfs_pywfs = mpwfs.forward(wf)

    return wfs_pywfs

def mod_forward_int(wfs_pywfs):
    """This iterates on wfs and returns an average of the power
    """
    image_final = 0
    for e, wfs_i in enumerate(wfs_pywfs):
        image_final += wfs_i.power 

    return image_final / ( e + 1) #/num_mod_STEPS


###########################################
######### Interaction Matrix setup ########
###########################################

def create_int_mat():
    #TODO: make the interaction matrix for non-modulated
    return

def create_int_mat_mod(deformable_mirror, wf, mpwfs, image_ref):
    probe_amp = 0.01 * wavelength_wfs
    slopes = []
    num_modes = deformable_mirror.num_actuators
    #iterate over the modes
    for ind in range(num_modes):
        if ind % 10 == 0:
            print("Measure response to mode {:d} / {:d}".format(ind+1, num_modes))
        slope = 0
        deformable_mirror.flatten()
        for s in [1, -1]:
            # setting the DM mirrors
            deformable_mirror.actuators[ind] += s * probe_amp
            # forwarding the wfs through the dm
            dm_wf = deformable_mirror.forward(wf)
            # MODULATION: use the mod pywfs
            mwfs_wf = mpwfs.forward(dm_wf)
            # MODULATION: use the modulation forwarding
            image = mod_forward_int(mwfs_wf)
            #show the differences in the slope
            slope += s * (image)/(2 * probe_amp)

            deformable_mirror.actuators[ind] -= s * probe_amp
        slopes.append(slope)
    slopes = np.array(slopes)
    slopes = slopes/np.sum(slopes*slopes)
    return slopes

def create_int_mat_mod_file(mode_file, deformable_mirror, wf, mpwfs, image_ref):
    probe_amp = 0.01 * wavelength_wfs
    slopes = []
    num_modes = deformable_mirror.num_actuators
    mode_pattern = fits.open(mode_fits).data[0]
    #iterate over the modes
    for ind in range(num_modes):
        if ind % 10 == 0:
            print("Measure response to mode {:d} / {:d}".format(ind+1, num_modes))
        slope = 0
        deformable_mirror.flatten()
        for s in [1, -1]:
            # setting the DM mirrors
            #deformable_mirror.actuators[ind] = s * probe_amp
            deformable_mirror.actuators = s * probe_amp * mode_pattern[ind]
            # forwarding the wfs through the dm
            dm_wf = deformable_mirror.forward(wf)
            # MODULATION: use the mod pywfs
            mwfs_wf = mpwfs.forward(dm_wf)
            # MODULATION: use the modulation forwarding
            image = mod_forward_int(mwfs_wf)
            #show the differences in the slope
            slope += s * (image-image_ref)/(2 * probe_amp)
        slopes.append(slope)
   #slopes = np.array(slopes).T
    slopes = ModeBasis(slopes)
    return slopes

def create_recon_mat(slopes):
    # inversing the slopes
    reconstruction_matrix = inverse_tikhonov(slopes, rcond) # previously 1e-3
    #reconstruction_matrix = inverse_tikhonov(slopes, 0)
    #reconstruction_matrix = inverse_truncated(slopes)
    return reconstruction_matrix

def init_prop(pupil_grid, wf, n_airy=28):
    focal_grid = make_focal_grid(q=8, num_airy=n_airy, spatial_resolution=spatial_resolution)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)
    norm = prop(wf).power.max()
    
    return prop, norm, focal_grid

###########################################
######### ATMOSPHERE WORK  ##############
###########################################

def high_pass_filter(diameter):
    # using 1 - circle to leave the residual high frequency, just circle is a low pass filter
    def func(grid):
        return 1 - 0.8*evaluate_supersampled(make_circular_aperture(diameter), grid, 4)
    return func

def LCO_atmosphere(pupil_grid):
    layers = atmosphere.make_las_campanas_atmospheric_layers(pupil_grid, r0=0.16)
    ml_atmsph = atmosphere.MultiLayerAtmosphere(layers)
    return ml_atmsph

def atms_phase(atmsph, reset=True):
    if reset:
        atmsph.reset()
    phase = atmsph.phase_for(wavelength_wfs)
    return phase

def filtered_atmosphere(phase, pupil_grid):
    # set up the high pass filter
    ff = fourier.FourierFilter(pupil_grid, high_pass_filter(2 * np.pi * 48 / 6.5))
    phase_res = np.real(ff.forward(phase + 0j) )
    return phase_res
    

###########################################
######### SPARKLE CODE  ##############
###########################################


# creating an actuator pattern from hcipy
def genSpark(seperation=15, amp = 0.01, m_angle = 0.0, angleOff = 28.0, m_cross = True):
    
    act_grid = make_pupil_grid(num_actuators_across_pupil, telescope_diameter).rotate(angleOff - m_angle) # DM based on real params

    # converting from MagAO-X param (um) to HCIPY OPD values (m)
    m_amp = amp * 1e-6 # pretty sure this should be -6

    # Like Jared's code, sine and cosine, but explicitly for x and y
    pattern_mx = np.cos(2 * np.pi * act_grid.x * seperation / telescope_diameter)
    pattern_my = np.cos(2 * np.pi * act_grid.y * seperation / telescope_diameter)

    pattern_nx = np.sin(2 * np.pi * act_grid.x * seperation / telescope_diameter)
    pattern_ny = np.sin(2 * np.pi * act_grid.y * seperation / telescope_diameter)

    #### Sparkle four counts #### 

    # Spark 0: a cross of sine and cosine
    spark0 = (pattern_mx + pattern_ny)
    if m_cross:
        spark0 -= (-1*pattern_nx + pattern_my)
    spark0 *= m_amp
    # Spark 1: equal and opposite to spark0
    spark1 = -spark0

    # Spark 2: 90 deg shift from spark0
    spark2 = -1* pattern_mx + pattern_ny 
    if m_cross:
        spark2 -= (-1*pattern_nx + pattern_my)
    spark2 *= m_amp
    # Spark 3: equal and opposie to spark2
    spark3 = -spark2

    return spark0, spark1, spark2, spark3
    
###########################################
######### RUNNING AO System  ##############
###########################################

def closed_loop(t, wf, mpwfs, deformable_mirror, magellan_aperture, image_ref, reconstruction_matrix, prop):
    PSF = prop(deformable_mirror(wf)).power

    for i in range(t):
        # forward deformable mirror
        wf_dm = deformable_mirror.forward(wf)
        # forward pywfs given dm wf
        wf_pyr = mpwfs.forward(wf_dm)
        wfs_image = mod_forward_int(wf_pyr)

        # difference between what we have and what we expect
        diff_image = wfs_image - image_ref
        deformable_mirror.actuators = (1-leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(diff_image)

        # setting the phase
        phase = magellan_aperture * deformable_mirror.surface
        phase -= np.mean(phase[magellan_aperture>0])

        # determining the central core
        psf = prop(deformable_mirror(wf)).power


def __main__():
    # walking through the tutorial

    # Creating the telescope pupil
    pupil_grid, pwfs_grid, magellan_aperture = create_mag_pupil()

    # Creating the DM
    deformable_mirror = create_DM(pupil_grid)

    # create mod PWFS
    mpwfs, camera = create_mod_PWFS(pupil_grid, pwfs_grid)

    # create wf
    wf = create_wf(magellan_aperture)
    wfs_pywfs = create_pwfs(mpwfs, wf)

    image_ref = mod_forward_int(wfs_pywfs)

    # create interaction matrix 
    slopes = create_int_mat_mod(deformable_mirror, wf, mpwfs, image_ref)






    
