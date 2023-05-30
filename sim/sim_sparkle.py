# Eden McEwen
# March 2023

from hcipy import *
import numpy as np
# might not need these because they're for plotting
from matplotlib import animation, rc
from IPython.display import HTML

# degining some global variables
wavelength_wfs = 842.0E-9       # Sensing wavelength
telescope_diameter = 6.5        # Diameter of the Magellan Clay
zero_magnitude_flux = 3.9E10    # Zero maf star plux, in photos/s
stellar_magnitude = 0           # zero magnitude star

# Camera variables:
num_pupil_pixels = 60           # has a distance of 60 pixels between the pupils
num_pupil_pixels_tel = 56       # number pixels used across the telescope 
# Pixels agross/ actual pixels multiplied by diameter gives you pupil diameter spacing?
pupil_grid_diameter = num_pupil_pixels/num_pupil_pixels_tel * telescope_diameter
modulation = 3                 #

# Deformable Mirror Variables
num_actuators_across_pupil = 50
rcond = 1E-3 # reconstruction matrix value

#AO system parameters
spatial_resolution = wavelength_wfs / telescope_diameter
delta_t = 1E-3          # integration time for exposures TODO: check with pywfs
leakage = 0.0           # for leaky integrator
gain = 0.5              # for leaky integrator


# Designing the Magellan Pupil
def create_mag_pupil():
    # Grid defined with number of puils, and then the diameter, scaled to our pixel ratios
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
    # Actual Pyramid doubles the pupil count and pixel 
    pwfs_grid = make_pupil_grid(2*num_pupil_pixels, 2 * pupil_grid_diameter)
    # taking the pupil grid, ovevrsampling by 6 to make up for the spiders pixelization
    magellan_aperture = evaluate_supersampled(make_magellan_aperture(), pupil_grid, 6)

    return pupil_grid, pwfs_grid, magellan_aperture

def create_DM(pupil_grid):
    # this is meters per actuator
    actuator_spacing = telescope_diameter / num_actuators_across_pupil
    # influence functions given the same pupil gir, num actuators, and their spacing
    influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)
    # creating DM with the above configureations for interaction
    deformable_mirror = DeformableMirror(influence_functions)

    return deformable_mirror

def create_mod_PWFS(pupil_grid, pwfs_grid, n_steps=12):
    # creating a pyramid wfs on the grid of our pupil, seperated similarly to how our pupil itself is
    pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=pupil_grid_diameter, pupil_diameter=telescope_diameter, wavelength_0=wavelength_wfs, q=3)
    # Simulating OCAM as a noiseless detector
    camera = NoiselessDetector(pwfs_grid)
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
            deformable_mirror.actuators[ind] = s * probe_amp
            # forwarding the wfs through the dm
            dm_wf = deformable_mirror.forward(wf)
            # MODULATION: use the mod pywfs
            mwfs_wf = mpwfs.forward(dm_wf)
            # MODULATION: use the modulation forwarding
            image = mod_forward_int(mwfs_wf)
            #show the differences in the slope
            slope += s * (image-image_ref)/(2 * probe_amp)
        slopes.append(slope)
    slopes = ModeBasis(slopes)

    return slopes

def create_recon_mat(slopes):
    # inversing the slopes
    reconstruction_matrix = inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)

    return reconstruction_matrix

def init_prop(pupil_grid, wf):
    focal_grid = make_focal_grid(q=8, num_airy=20, spatial_resolution=spatial_resolution)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)
    norm = prop(wf).power.max()
    
    return prop, norm


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





# creating an actuator pattern from hcipy
def genSpark(seperation=15, m_angle = 0.0, angleOff = 28.0, amp = 0.01, m_cross = True):
    
    act_grid = make_pupil_grid(num_actuators_across_pupil, telescope_diameter).rotate(angleOff - m_angle) # DM based on real params

    # converting from MagAO-X param (um) to HCIPY OPD values (m)
    m_amp = amp * 10e-6

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






    
