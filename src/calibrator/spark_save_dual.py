#!/usr/bin/env python3
"""
Save synchronized sparkle calibration data from two streamwriters.

This mirrors the SparkSave flow but toggles both:
- camwfs-sw
- aol1_imWFS2-sw
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import time
from zoneinfo import ZoneInfo

import purepyindi2 as indi

CHILE_TZ = ZoneInfo("America/Santiago")
FILE_FORMAT = "%Y%m%d%H%M%S%f000"

PRIMARY_STREAM_WRITER = "camwfs"
SECONDARY_STREAM_WRITER = "aol1_imWFS2"
STREAM_WRITERS = (PRIMARY_STREAM_WRITER, SECONDARY_STREAM_WRITER)


class SparkSaveDual:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def setup(self, dir_spark_calib, n_frames=4000):
        self.dir_spark_calib = dir_spark_calib
        self.n_frames = n_frames

        self.client = indi.client.IndiClient()
        self.client.connect()

        subscribed_devices = ["tweeterSpeck"] + [f"{name}-sw" for name in STREAM_WRITERS]
        self.client.get_properties(subscribed_devices)
        self.wait_for_required_properties()
        return

    def setParams(self, sep, ang, amp):
        self.client["tweeterSpeck.modulating.toggle"] = indi.SwitchState.OFF
        self.client["tweeterSpeck.separation.target"] = sep
        self.client["tweeterSpeck.angle.target"] = ang
        self.client["tweeterSpeck.amp.target"] = amp
        time.sleep(1)
        self.client["tweeterSpeck.modulating.toggle"] = indi.SwitchState.ON
        time.sleep(1)
        return self.checkParams()

    def checkParams(self):
        self.sep = self.client["tweeterSpeck.separation.current"]
        self.ang = self.client["tweeterSpeck.angle.current"]
        self.amp = self.client["tweeterSpeck.amp.current"]
        self.freq = self.client["tweeterSpeck.frequency.current"]
        self.dwell = self.client["tweeterSpeck.dwell.current"]
        self.delay = self.client["tweeterSpeck.delay.current"]
        self.trigger = self.client["tweeterSpeck.trigger.toggle"]

        self.calib_folder = (
            f"sep{int(self.sep):02d}_ang{int(self.ang):02d}_"
            f"amp{self.amp:01.3f}_freq{int(self.freq):02d}"
        )
        print(f"Updated save location: {self.calib_folder}")

        mod = self.client["tweeterSpeck.modulating.toggle"]
        if mod == indi.SwitchState.OFF:
            print("Sparkle is not modulating")
            return False
        return True

    def take_sparkle_data(self):
        if not self.checkParams():
            print("WARNING: Sparkle params are not set correctly, saving with current params")

        self.check_calib_dir()
        self.save_cube()
        self.save_savedata_txt()
        return

    def wait_for_required_properties(self, timeout_s=10.0, poll_s=0.1):
        required_props = ["tweeterSpeck.modulating.toggle"]
        for name in STREAM_WRITERS:
            required_props.append(f"{name}-sw.writing.toggle")
            required_props.append(f"{name}-sw.fsm.state")

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
            "Timed out waiting for required INDI properties: " + ", ".join(missing)
        )

    def check_calib_dir(self):
        self.calib_path = f"{self.dir_spark_calib}/{self.calib_folder}"
        if pathlib.Path(self.calib_path).exists():
            print(f"Path exists: {self.calib_path}")
        else:
            print(f"making directory for this calibration: {self.calib_path}")
            pathlib.Path(self.calib_path).mkdir(parents=True, exist_ok=True)
        return

    def _set_all_writing(self, toggle_state):
        for name in STREAM_WRITERS:
            self.client[f"{name}-sw.writing.toggle"] = toggle_state

    def _all_streams_operating(self):
        for name in STREAM_WRITERS:
            if self.client[f"{name}-sw.fsm.state"] != "OPERATING":
                return False
        return True

    def save_cube(self, startup_timeout_s=2.0, poll_s=0.01):
        self.ts_start = datetime.datetime.now(CHILE_TZ).astimezone(datetime.timezone.utc)
        self._set_all_writing(indi.ON)

        t0 = time.time()
        while not self._all_streams_operating():
            time.sleep(poll_s)
            if time.time() - t0 > startup_timeout_s:
                self._set_all_writing(indi.OFF)
                raise RuntimeError(
                    "One or more streamwriters did not enter OPERATING state in time"
                )

        dt = self.n_frames / self.freq
        print(f"Waiting for {dt} seconds")
        time.sleep(dt)

        self._set_all_writing(indi.OFF)
        self.ts_end = datetime.datetime.now(CHILE_TZ).astimezone(datetime.timezone.utc)
        return

    def _get_save_metadata(self):
        return {
            "SPK_sep": self.sep,
            "SPK_ang": self.ang,
            "SPK_amp": self.amp,
            "WFS_hz": self.freq,
            "SPK_dwell": self.dwell,
            "SPK_delay": self.delay,
            "ts_start": self.ts_start.strftime(FILE_FORMAT),
            "ts_end": self.ts_end.strftime(FILE_FORMAT),
            "stream_primary": f"{PRIMARY_STREAM_WRITER}-sw",
            "stream_secondary": f"{SECONDARY_STREAM_WRITER}-sw",
        }

    def save_savedata_txt(self):
        filename = f"{self.calib_path}/savedata.txt"
        metadata = self._get_save_metadata()
        with open(filename, "w") as fh:
            fh.write("# SparkCalib dual-stream cube metadata\n")
            for key, value in metadata.items():
                fh.write(f"{key}: {value}\n")
        return

