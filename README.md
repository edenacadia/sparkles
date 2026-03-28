# Sparkles

Sparkles are incoherent speckles produced by the DM. They are paired spots in the focal plane (and corresponding sine-wave structure in the pupil/WFS space). This repository contains tools to:

- save sparkle calibration data
- process that data into PCA calibration products
- re-run processing for existing saved calibration folders

## Running The Calibrator

All commands below are shown from the repository root (`sparkles/`).

### Single sparkle run (`calibrate.py`)

Run save + process in one command:

```bash
python3 src/calibrator/calibrate.py --sep 22 --ang 18 --amp 0.03
```

Run save only:

```bash
python3 src/calibrator/calibrate.py --sep 22 --ang 18 --amp 0.03 --save-only
```

Run process only (for existing saved data):

```bash
python3 src/calibrator/calibrate.py --sep 22 --ang 18 --amp 0.03 --process-only --freq 2000
```

Use dual-stream save mode (`camwfs-sw + aol1_imWFS2-sw`):

```bash
python3 src/calibrator/calibrate.py --sep 22 --ang 18 --amp 0.03 --dual-save
```

### Grid runs (`calibrate_grid.py`)

Run the default parameter grid:

```bash
python3 src/calibrator/calibrate_grid.py
```

Preview what would run without executing:

```bash
python3 src/calibrator/calibrate_grid.py --dry-run
```

Force reruns even when outputs already exist:

```bash
python3 src/calibrator/calibrate_grid.py --force-rerun
```

Grid save-only or process-only modes:

```bash
python3 src/calibrator/calibrate_grid.py --save-only
python3 src/calibrator/calibrate_grid.py --process-only --freq 2000
```

### Batch reprocess existing camwfs runs

Re-run processing for all folders under a root that contain `savedata.txt` and a `camwfs-sw` directory:

```bash
python3 src/calibrator/calibrate.py --recalibrate-camwfs-all --spark-ao-dir /home/eden/data/spark_AO
```

Limit PCA frames used in this batch mode:

```bash
python3 src/calibrator/calibrate.py --recalibrate-camwfs-all --n-pca-max 4000
```

## Useful CLI options

`calibrate.py`:

- `--sep`, `--ang`, `--amp`: required for single-run mode
- `--calib-dir`: calibration output directory (default: `/home/eden/data/spark_calib/`)
- `--cores`: limit numeric-library thread usage
- `--save-only` / `--process-only`: run only one stage
- `--freq`: processing folder lookup frequency (important for process-only mode)
- `--dual-save`: save both streamwriters
- `--recalibrate-camwfs-all`: batch reprocess mode
- `--spark-ao-dir`: root folder for batch mode
- `--n-pca-max`: max PCA frames in batch mode

`calibrate_grid.py`:

- `--dry-run`: print run/skip status only
- `--force-rerun`: do not skip existing outputs
- `--stop-on-error`: stop at first failure
- `--save-only` / `--process-only`
- `--dual-save`
- `--freq`, `--cores`, `--calib-dir`

Use `--help` on either script for the latest full argument list:

```bash
python3 src/calibrator/calibrate.py --help
python3 src/calibrator/calibrate_grid.py --help
```
