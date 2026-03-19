# spark_save.py
# March 19 2026
# this is the program that saves the sparkle calibration data and metadata
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

class SparkSave(): # not sure yet if I need to make this an XDevice... 
    config: sparkCalibConfig

    def __init__(self):
        # This class can run standalone (outside XDevice), so create a logger here.
        self.log = logging.getLogger(self.__class__.__name__)

    def setup(self, dir_spark_calib, n_frames=4000):
        #TODO: make a more permanent location
        self.dir_spark_calib  = dir_spark_calib 
        # These are the number of frames wewant to take when starting the program
        self.n_frames = n_frames
        # TODO: set up a logger here
        self.client = indi.client.IndiClient()
        self.client.connect()
        self.client.get_properties(['tweeterSpeck', STREAM_WRITER_NAME+'-sw'])
        self.wait_for_required_properties()
        return

    def take_sparkle_data(self):
        """ 
        This function polls indi poroperties to take the sparkle calibration data
        1. Check parameters 
        """
        # check the sparkle params, update save directory
        if not self.checkParams():
            print("WARNING: Sparkle params are not set correctly, calibrating with current params")
        # check to make sure directory exists, if not, make it 
        self.check_calib_dir()
        # start logging, saving from streamwriter
        self.save_cube()
        # Save the metadata to the calib directory
        self.save_savedata_txt()
        return

    def setParams(self, sep, ang, amp):
        """
        Changes the sparkle parameters, and updates the save directory
        """
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

    def checkParams(self):
        """ 
        Check the sparkle params, update sparkle save directory 
        """
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
    
    def check_calib_dir(self):
        """
        Check to see if a directory exists for the sparkle calibration
        """
        self.calib_path = f"{self.dir_spark_calib}/{self.calib_folder}"
        if pathlib.Path(self.calib_path).exists():
            print(f"Path exists: {self.calib_path}")
        else:
            print(f"making directory for this calibration: {self.calib_path}")
            pathlib.Path(f"{self.calib_path}").mkdir(parents=True, exist_ok=True)
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

    def save_savedata_txt(self):
        """
        Save the metadata relavant to the observation
        """
        filename = f"{self.calib_path}/savedata.txt"
        metadata = self._get_save_metadata()
        with open(filename, "w") as fh:
            fh.write("# SparkCalib cube obs metadata\n")
            for key, value in metadata.items():
                fh.write(f"{key}: {value}\n")
        return

    def _get_save_metadata(self):
        """Build metadata needed by SparkCalib processing."""
        return {
            'SPK_sep': self.sep,
            'SPK_ang': self.ang,
            'SPK_amp': self.amp,
            'WFS_hz': self.freq,
            'SPK_dwell': self.dwell,
            'SPK_delay': self.delay,
            'ts_start': self.ts_start.strftime(FILE_FORMAT),
            'ts_end': self.ts_end.strftime(FILE_FORMAT),
        }