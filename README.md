# Sparkles

Sparkels are incoherent speckles produced by the DM. They are pairs of delta functions on the focal plane, which correspondingly are sign waves on the pupil plane, as sensed by the WFS. 

This package contains admittedly too much code surrounding sensing the amplitude of the sparkles relative to lab calibrated sparkles. 

## Calibrator

    The calibration scripts allow `aol1_imWFS2-sw` or the streamwriter that logs the processed camWFS images (dark subtracted, normalized) to be saved with the known sparkle parameters. This is how the basis for each sparkle combination is generated. 

    Running a single calibration:

    `python3 sparkles/calibrator/calibrate.py --sep 22 --ang 18 --amp 0.03`

    Running a grid, a large 84 set by default: 

    `python3 sparkles/calibrator/calibrate_grid.py`

    Options that might be helpful:
        `--dry-run` for checking what might get run
        `--cores 4` for specifiying 4 ocres being used, for example
        `--force-rerun` for repeating created directories


## Sparkles 

    These set of files 
