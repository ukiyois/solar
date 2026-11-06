"""Download unlabeled GONG H-alpha full-disk images (gong2.nso.edu FTP) for self-training.

Same instrument/archive as the competition data (its JSON 'url' fields point to
gong2.nso.edu/HA/hag). Archive JPEGs carry text/logo overlays in the corners,
outside a radius of ~1024 px; the disk content matches the competition images
(center mean |diff| 1.8 grey levels). Output: grayscale JPEGs with everything
beyond radius 1024 from the image center set to 0.

Selection: for every observation day of the competition data (train and test),
other images of the same day from the six competition sites, plus random extra
days 2011-2022. The competition files themselves are excluded.

File names start with the observation time and site, YYYYMMDDhhmmss + site code (e.g. 20150527082014Th),
like the competition files. Per day at most one frame per site is chosen (--per-competition-day or
--per-random-day sites). Frames of test-image days are downloaded but never pseudo-labelled
(scripts/pseudo_label.excluded_days). Output in --output: images/*.jpeg, listings/<YYYYMMDD>.json (cached
FTP listings) and the selection file (--selection-file) with the chosen names; another selection file name
starts a further batch that skips the days and files of earlier batches.

usage (repository root): python scripts/download_gong.py --output data/gong_extra
"""
import argparse
import datetime as dt
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftplib import FTP
from pathlib import Path

import numpy as np
from PIL import Image

SITES = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
HOST = "gong2.nso.edu"


def ftp():
    """Anonymous FTP login to the NSO GONG archive (120 s timeout)."""
    client = FTP(HOST, timeout=120)
    client.login()
    return client


LISTING_CACHE = None  # set in main(): <output>/listings/<YYYYMMDD>.json, so restarts never re-list a day


def listing(day):
    """Names of the .jpg frames of one day in the archive folder /HA/hag/<YYYYMM>/<YYYYMMDD>.

    Successful listings are cached in LISTING_CACHE. Up to 4 attempts with growing pauses; after that an
    empty list is returned and nothing is cached, so a later run lists the day again.
    """
    cached = LISTING_CACHE / f"{day:%Y%m%d}.json" if LISTING_CACHE else None
    if cached is not None and cached.exists():
        return json.loads(cached.read_text())
    folder = f"/HA/hag/{day:%Y%m}/{day:%Y%m%d}"
    for attempt in range(4):
        try:
            client = ftp()
            names = client.nlst(folder)
            client.quit()
            result = [Path(n).name for n in names if n.endswith(".jpg")]
            if cached is not None:
                cached.write_text(json.dumps(result))
            return result
        except Exception:
            time.sleep(5 * (attempt + 1))
    return []


def clean(raw_bytes):
    """Decode an archive JPEG to grayscale and zero every pixel farther than 1024 px from the image center.

    This removes the text and logo overlays in the corners; the solar disk lies inside that radius.
    """
    image = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("L")).copy()
    h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    image[(yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2 > 1024 ** 2] = 0
    return image


_local = threading.local()


def session(reset=False):
    """One persistent FTP login per worker thread (a login per file costs several RTTs)."""
    if reset or getattr(_local, "client", None) is None:
        try:
            if getattr(_local, "client", None) is not None:
                _local.client.close()
        except Exception:
            pass
        _local.client = ftp()
    return _local.client


def fetch(name, out_dir):
    """Download one archive frame, clean() it and save it as <out_dir>/<stem>.jpeg (JPEG quality 95).

    Returns a status: "exists" (already downloaded), "ok", "bad_shape" (not 2048 x 2048, not saved) or
    "failed" after 5 attempts. Attempts after the first re-open the thread's FTP session.
    """
    target = out_dir / (Path(name).stem + ".jpeg")
    if target.exists():
        return "exists"
    day = dt.datetime.strptime(name[:8], "%Y%m%d")
    path = f"/HA/hag/{day:%Y%m}/{day:%Y%m%d}/{name}"
    for attempt in range(5):
        try:
            buffer = io.BytesIO()
            session(reset=attempt > 0).retrbinary(f"RETR {path}", buffer.write, blocksize=65536)
            image = clean(buffer.getvalue())
            if image.shape != (2048, 2048):
                return "bad_shape"
            temporary = target.with_suffix(".tmp")
            Image.fromarray(image).save(temporary, format="JPEG", quality=95)
            temporary.replace(target)
            return "ok"
        except Exception:
            time.sleep(5 * (attempt + 1))
    return "failed"


def main():
    """Select archive frames (or reuse an existing selection file) and download them in parallel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output", default="data/gong_extra")
    parser.add_argument("--per-competition-day", type=int, default=2)
    parser.add_argument("--random-days", type=int, default=300)
    parser.add_argument("--per-random-day", type=int, default=4)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--span-start", default="2011-01-01", help="random-day span (inclusive)")
    parser.add_argument("--span-end", default="2022-12-31")
    parser.add_argument("--selection-file", default="selection.json",
                        help="Use a new name to add another batch (already downloaded files are skipped)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    root = Path(args.data_root)
    competition = {p.stem for p in (root / "train/train_images").iterdir()} | \
                  {p.stem for p in (root / "test/test_images").iterdir()}
    days = {dt.datetime.strptime(s[:8], "%Y%m%d").date() for s in competition}
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    global LISTING_CACHE
    LISTING_CACHE = out / "listings"
    LISTING_CACHE.mkdir(exist_ok=True)
    # Earlier batches: their days are not re-sampled and their files are not re-selected.
    previous = set()
    for old in out.glob("selection*.json"):
        if old.name != args.selection_file:
            previous |= set(json.loads(old.read_text())["files"])
    competition |= {Path(name).stem[:16] for name in previous}
    used_days = days | {dt.datetime.strptime(n[:8], "%Y%m%d").date() for n in previous}
    span = (dt.date.fromisoformat(args.span_start), dt.date.fromisoformat(args.span_end))
    free = [span[0] + dt.timedelta(days=k) for k in range((span[1] - span[0]).days + 1)]
    free = [d for d in free if d not in used_days]
    # every unused day of the span when fewer remain than requested (no endless resampling)
    extra = set(rng.sample(free, min(args.random_days, len(free))))
    plan = ([(d, args.per_competition_day) for d in sorted(days)] if args.per_competition_day else []) + \
           [(d, args.per_random_day) for d in sorted(extra)]
    manifest_path = out / args.selection_file
    if manifest_path.exists():
        selected = json.loads(manifest_path.read_text())["files"]
    else:
        selected = []
        with ThreadPoolExecutor(16) as pool:
            futures = {pool.submit(listing, d): (d, k) for d, k in plan}
            for n, future in enumerate(as_completed(futures)):
                d, k = futures[future]
                # x[14:16] = site code, x[:16] = time + site, the stem format of the competition files
                names = [x for x in future.result() if x[14:16] in SITES and x[:16] not in competition]
                by_site = {}
                for x in names:
                    by_site.setdefault(x[14:16], []).append(x)
                sites = list(by_site)
                rng.shuffle(sites)
                selected += [rng.choice(by_site[s]) for s in sites[:k]]
                if (n + 1) % 100 == 0:
                    print(json.dumps({"listed_days": n + 1, "of": len(plan), "selected": len(selected)}), flush=True)
        manifest_path.write_text(json.dumps({"files": sorted(selected), "competition_days": len(days),
                                             "random_days": len(extra), "sites": SITES}, indent=1))
    print(json.dumps({"to_download": len(selected)}), flush=True)
    counts, started, lock = {}, time.time(), threading.Lock()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, name, out / "images") for name in selected]
        for n, future in enumerate(as_completed(futures)):
            result = future.result()
            with lock:
                counts[result] = counts.get(result, 0) + 1
            if (n + 1) % 50 == 0:
                print(json.dumps({"done": n + 1, "of": len(selected), **counts,
                                  "minutes": round((time.time() - started) / 60, 1)}), flush=True)
    print(json.dumps({"finished": counts, "minutes": round((time.time() - started) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
