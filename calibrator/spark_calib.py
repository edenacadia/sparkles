import sys
import logging
from enum import Enum
import time
import numpy as np

import xconf
import fixr

from magaox.indi.device import XDevice, BaseConfig
from magaox.camera import XCam
from magaox.constants import StateCodes

import purepyindi2 as indi
from purepyindi2 import device, properties, constants
from purepyindi2.messages import DefNumber, DefSwitch, DefLight, DefText

import os
import datetime
from zoneinfo import ZoneInfo
import hcipy as hp
from astropy.io import fits
from scipy.optimize import minimize
import pathlib

import sys
from pathlib import Path
# repo root that contains the `sparkles/` package folder
repo_root = Path("/home/eden/code/sparkles")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sparkles import pca

# TODO: Need to switch prints to logs

# If this isn't the aol1_imWFS2 streamwriter, than we need to dark subtract
STREAM_WRITER_NAME = 'aol1_imWFS2'
OBS_FOLDER_FORMTAT = "%Y_%m_%d"
MODIFIED_TIME_FORMAT = f"{STREAM_WRITER_NAME}_%Y%m%d%H%M%S%f000.xrif"
FILE_FORMAT = "%Y%m%d%H%M%S%f000"
DATA_PATH = pathlib.Path(f"/opt/MagAOX/rawimages/{STREAM_WRITER_NAME}/")
CHILE_TZ = ZoneInfo("America/Santiago")

@xconf.config
class CameraConfig:
    """
    """
    shmim : str = xconf.field(help="Name of the camera device (specifically, the associated shmim, if different)")
    dark_shmim : str = xconf.field(help="Name of the dark frame shmim associated with this camera device")
    #TODO: Does camtip have a dark frame?

@xconf.config
class sparkCalibConfig(BaseConfig):
    """ Configure  """
    # if want default need to be a cam config obj
    camera : CameraConfig = xconf.field(help="Camera to use")

class SparkCalib(): # not sure yet if I need to make this an XDevice... 
    config: sparkCalibConfig

    def __init__(self):
        # This class can run standalone (outside XDevice), so create a logger here.
        self.log = logging.getLogger(self.__class__.__name__)

    def setup(self, dir_spark_calib, n_frames=4000):
        self.dir_spark_calib  = dir_spark_calib #TODO: make a more permanent location
        # TODO: does XDevice auto connect to a client? is that why we use it?
        self.n_frames = n_frames
        # Re-assert logger for defensive safety if __init__ is bypassed.
        if not hasattr(self, "log"):
            self.log = logging.getLogger(self.__class__.__name__)
        self.client = indi.client.IndiClient()
        self.client.connect()
        self.client.get_properties(['tweeterSpeck', STREAM_WRITER_NAME+'-sw'])
        self.wait_for_required_properties()
        return

    def wait_for_required_properties(self, timeout_s=10.0, poll_s=0.1):
        """
        Wait for required INDI properties to appear before continuing.
        get_properties() subscriptions are asynchronous.
        """
        required_props = [
            'tweeterSpeck.modulating.toggle',
            f'{STREAM_WRITER_NAME}-sw.writing.toggle',
            f'{STREAM_WRITER_NAME}-sw.fsm.state',
        ]
        t0 = time.time()
        missing = list(required_props)
        while time.time() - t0 < timeout_s:
            missing = []
            for prop in required_props:
                try:
                    _ = self.client[prop]
                except KeyError:
                    missing.append(prop)
            if not missing:
                print(f"INDI properties ready: {', '.join(required_props)}")
                return
            time.sleep(poll_s)

        raise RuntimeError(
            "Timed out waiting for required INDI properties: "
            + ", ".join(missing)
        )

    def calibrate(self):
        """ Main function
        1. Check the sparkle params, update save directory
        2. Grab a stack of frames, with the dark subtracted
        3. Generate the PCA basis, and the self norm
        4. Save the PCA basis and self norm to the save directory
        """
        # check the sparkle params, update save directory
        if not self.checkParams():
            print("WARNING: Sparkle params are not set correctly, calibrating with current params")
        # check to make sure directory exists, if not, make it 
        self.check_calib_dir()
        # start logging, saving from streamwriter
        self.save_cube()
        # grab that stack of frames, with the dark subtracted
        data = self.grab_cube()
        # generate the PCA basis, and the self reference
        self.gen_lab_ref(data)
        # save the PCA basis and self reference to the save directory
        self.save_reference()
        return

    def save_cube(self):
        """ Toggle the stream writers, save the start/stop times"""
        # Now save the datetime 
        self.ts_start = datetime.datetime.now(CHILE_TZ).astimezone(datetime.timezone.utc)
        self.client[f'{STREAM_WRITER_NAME}-sw.writing.toggle'] = indi.ON
        # Waiting for confirmation before storing time. 
        counter = 0
        while self.client[f'{STREAM_WRITER_NAME}-sw.fsm.state'] != 'OPERATING': 
            counter += 1
            time.sleep(0.01)
            if counter > 100:
                print("WARNING: Camera stream is not writing after 1 seconds, check the camera and the stream writer")
                return
        # some kind of wait 4 seconds or whatever
        #time.sleep(4) #TODO: hardcode this or make it a config
        dt = self.n_frames/self.freq
        print(f"Waiting for {dt} seconds")
        time.sleep(dt)
        # and the toggle the streamwriter 
        self.client[f'{STREAM_WRITER_NAME}-sw.writing.toggle'] = indi.OFF
        self.ts_end = datetime.datetime.now(CHILE_TZ).astimezone(datetime.timezone.utc)
        # and then save when we turn it off 
        return 

    def grab_cube(self):
        # you have to have taken the data first
        xrif_files = self.file_list()
        self.n_files = len(xrif_files)
        # send those xrif files to the pull
        data_conglom = [self.pull_file_xrif(file)[0] for file in xrif_files]
        data_stack = np.vstack(data_conglom)
        # stack into one cube
        return data_stack

    def file_list(self):
        # use the start and end of obs to search for all xrif files
        date_fldr = self.ts_start.strftime(OBS_FOLDER_FORMTAT)
        folder_path = DATA_PATH / date_fldr
        # get all files
        all_matching_files = list(
            sorted(folder_path.glob(f"{STREAM_WRITER_NAME}_*.xrif"))
        )
        print(f"Found {len(all_matching_files)} files in the folder {folder_path}")
        # using the timestamps to make a sort function
        xrif_start = self.ts_start.strftime(MODIFIED_TIME_FORMAT)
        xrif_end = self.ts_end.strftime(MODIFIED_TIME_FORMAT)
        # then make sure 
        span_files = []
        for file_path in all_matching_files:
            if file_path.name > xrif_start:
                if file_path.name < xrif_end:
                    span_files.append(file_path)
        # sort before sending back
        print(f"Found {len(span_files)} files between {xrif_start} and {xrif_end}")
        return sorted(span_files)

    def pull_file_xrif(self, xfile, fsize=512):
        '''
        Take a path, return the data and the timing
        '''
        with open(xfile, 'rb') as fh:
            data = fixr.xrif2numpy(fh)
            timing = fixr.xrif2numpy(fh)
        #TODO: check if this is the right approach
        n_files = int(data.size / 120 / 120)
        return np.reshape(data, (n_files,120,120)), np.reshape(timing, (n_files,5))

    def gen_lab_ref(self, data, klip=3):
        """ 
        Pull a bunch of files, take an average
        """
        print("GENERATING REFERENCE PCA basis")
        #TODO: this darta pull probably needs to get cleaned, which requires a mask...
        # creating the PCA basis
        Z_KL, Z_KL_img = pca.pca_basis(data, klip=klip)
        self.ref_pca = Z_KL
        self.ref_pca_img = Z_KL_img
        # Then, we're going to create a self reference, for each roll. 
        lab_proj = pca.pca_projection(data, self.ref_pca) # this is for all frames
        # shape: [N, klip]
        # averages by the sparkle pattern
        lab_avgs = np.array([np.mean(lab_proj[i::4,:], axis=0) for i in range(4)])
        self.ref_proj = lab_avgs
        # shape: [spark, klip]
        # we will usually use these as a stdv across the time axis
        lab_rms = np.std(lab_proj, axis=0)
        self.ref_rms = lab_rms
        # shape: [klip]
        return Z_KL, Z_KL_img, lab_avgs
    
    def checkParams(self):
        """ Check the sparkle params, update sparkle save directory """
        self.sep = self.client['tweeterSpeck.separation.current']
        self.ang = self.client['tweeterSpeck.angle.current']
        self.amp = self.client['tweeterSpeck.amp.current']
        self.freq = self.client['tweeterSpeck.frequency.current']
        # make the folder for saving the sparkle data. 
        self.calib_folder = f"sep{int(self.sep):02d}_ang{int(self.ang):02d}_amp{self.amp:01.3f}_freq{int(self.freq):02d}"
        print("Updated save location: {:s}".format(self.calib_folder))
        # check to see if these are in the default state
        self.dwell = self.client['tweeterSpeck.dwell.current']
        self.delay = self.client['tweeterSpeck.delay.current']
        self.trigger = self.client['tweeterSpeck.trigger.toggle']
        # check to see if they're modulating, if not, warning 
        mod = self.client['tweeterSpeck.modulating.toggle']
        if mod == indi.SwitchState.OFF:
            #self.log.warning("Sparkle is not modulating")
            print("Sparkle is not modulating")
            return False
        if self.dwell > 1: 
            print(f"WARNING: we are dwelling {self.frames} frames")
        if self.delay > 0:
            print(f"WARNING: we are delaying for {self.delay} frames")
        if self.trigger == indi.SwitchState.OFF:
            print("WARNING: we are not triggering on the modulator")
        return True

    def setParams(self, sep, ang, amp):
        # turn off, then back on again: 
        self.client['tweeterSpeck.modulating.toggle'] =  indi.SwitchState.OFF
        # Setting the sparkle params
        self.client['tweeterSpeck.separation.target'] = sep
        self.client['tweeterSpeck.angle.target'] = ang
        self.client['tweeterSpeck.amp.target'] = amp
        time.sleep(1)
        # turn sparkles back on
        self.client['tweeterSpeck.modulating.toggle'] = indi.SwitchState.ON
        time.sleep(1)
        # Calling the check params to make sure that these are applied
        return self.checkParams()

    def check_calib_dir(self):
        self.calib_path = f"{self.dir_spark_calib}/{self.calib_folder}"
        if pathlib.Path(self.calib_path).exists():
            print(f"Path exists: {self.calib_path}")
        else:
            print(f"making directory for this calibration: {self.calib_path}")
            pathlib.Path(f"{self.calib_path}").mkdir(parents=True, exist_ok=True)
        return 
    
    def save_reference(self):
        self.check_calib_dir()
        self.metadata = self._get_save_metadata()
        # Using the save function to have all these saved
        self.save_fits(self.ref_pca, f"{self.calib_path}/ref_pca.fits")
        self.save_fits(self.ref_proj, f"{self.calib_path}/ref_proj.fits")
        self.save_fits(self.ref_rms, f"{self.calib_path}/ref_rms.fits")
        self.save_metadata_txt(f"{self.calib_path}/metadata.txt")
        return

    def _get_save_metadata(self):
        """Build shared metadata used for FITS headers and text saves."""
        return {
            'SPK_sep': self.sep,
            'SPK_ang': self.ang,
            'SPK_amp': self.amp,
            'WFS_hz': self.freq,
            'SPK_dwell': self.dwell,
            'SPK_delay': self.delay,
            'n_files': getattr(self, 'n_files', -1),
            'ts_start': self.ts_start.strftime(FILE_FORMAT),
            'ts_end': self.ts_end.strftime(FILE_FORMAT),
        }

    def save_metadata_txt(self, filename):
        """Save cube timestamps and calibration parameters to a text file."""
        metadata = self.metadata
        with open(filename, "w") as fh:
            fh.write("# SparkCalib cube save metadata\n")
            for key, value in metadata.items():
                fh.write(f"{key}: {value}\n")
        return

    def save_fits(self, data, filename):
        # generic fits saving function bc I'm putting all the same info in each header 
        # saving the PCA outputs with relavant info in the headers
        hdu = fits.PrimaryHDU()
        hdu.data = data
        metadata = self.metadata
        # saving sparkle params (will keep later)
        hdu.header['SPK_sep'] = metadata['SPK_sep']
        hdu.header['SPK_ang'] = metadata['SPK_ang']
        hdu.header['SPK_amp'] = metadata['SPK_amp']
        hdu.header['WFS_hz'] = metadata['WFS_hz']
        hdu.header['SPK_dwll'] = metadata['SPK_dwell']
        hdu.header['SPK_dlay'] = metadata['SPK_delay']
        hdu.header['n_files'] = metadata['n_files']
        hdu.header['ts_start'] = metadata['ts_start']
        hdu.header['ts_end'] = metadata['ts_end']
        # saving and overwriting if needed
        hdu.writeto(filename, overwrite=True)
        return