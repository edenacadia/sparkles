from __future__ import annotations

import datetime as dt
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


DEFAULT_DARK_DIR = Path("/opt/MagAOX/calib/camwfs-dark")
TS_RE = re.compile(r"__T(\d{14})(\d*)")


@dataclass(frozen=True)
class DarkMatch:
    target_dt_utc: dt.datetime
    dark_path: Path
    dark_dt_utc: dt.datetime
    delta_seconds: float


def parse_timestamp_token(token: str) -> dt.datetime | None:
    s = token.strip()

    try:
        iso = s.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        pass

    if re.fullmatch(r"\d{14}", s):
        parsed = dt.datetime.strptime(s, "%Y%m%d%H%M%S")
        return parsed.replace(tzinfo=dt.timezone.utc)

    return None


def parse_timestamp_from_name(path_like: str) -> dt.datetime:
    name = Path(path_like).name
    m = TS_RE.search(name)
    if not m:
        raise ValueError(f"No __T timestamp found in filename: {name}")

    main = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(
        tzinfo=dt.timezone.utc
    )
    frac_digits = m.group(2)
    if not frac_digits:
        return main

    usec = int((frac_digits[:6]).ljust(6, "0"))
    return main + dt.timedelta(microseconds=usec)


def parse_target_time(target: str | dt.datetime) -> dt.datetime:
    if isinstance(target, dt.datetime):
        if target.tzinfo is None:
            return target.replace(tzinfo=dt.timezone.utc)
        return target.astimezone(dt.timezone.utc)

    parsed = parse_timestamp_token(target)
    if parsed is not None:
        return parsed
    return parse_timestamp_from_name(target)


def _load_dark_catalog(dark_dir: Path) -> List[Tuple[float, Path, dt.datetime]]:
    if not dark_dir.exists():
        raise FileNotFoundError(f"Dark directory does not exist: {dark_dir}")

    dark_files = sorted(dark_dir.glob("camwfs-dark*.fits"))
    if not dark_files:
        raise FileNotFoundError(f"No camwfs dark FITS files in: {dark_dir}")

    catalog: List[Tuple[float, Path, dt.datetime]] = []
    for p in dark_files:
        try:
            t = parse_timestamp_from_name(p.name)
        except ValueError:
            continue
        catalog.append((t.timestamp(), p, t))

    if not catalog:
        raise RuntimeError("No parseable dark timestamps found.")

    catalog.sort(key=lambda x: x[0])
    return catalog


def _closest_by_time(
    target_time: dt.datetime, catalog: Sequence[Tuple[float, Path, dt.datetime]]
) -> Tuple[Path, dt.datetime, float]:
    target_epoch = target_time.timestamp()
    epochs = [x[0] for x in catalog]
    idx = bisect_left(epochs, target_epoch)

    candidates = []
    if idx < len(catalog):
        candidates.append(catalog[idx])
    if idx > 0:
        candidates.append(catalog[idx - 1])

    best = min(candidates, key=lambda x: abs(x[0] - target_epoch))
    delta_s = best[0] - target_epoch
    return best[1], best[2], delta_s


def find_closest_camwfs_dark(
    target: str | dt.datetime, dark_dir: str | Path = DEFAULT_DARK_DIR
) -> DarkMatch:
    target_dt_utc = parse_target_time(target)
    catalog = _load_dark_catalog(Path(dark_dir))
    dark_path, dark_dt_utc, delta_seconds = _closest_by_time(target_dt_utc, catalog)
    return DarkMatch(
        target_dt_utc=target_dt_utc,
        dark_path=dark_path,
        dark_dt_utc=dark_dt_utc,
        delta_seconds=delta_seconds,
    )
