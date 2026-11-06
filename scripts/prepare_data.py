"""One-time segmentation-only CPU preprocessing; no model or GPU is used.

Rasterizes the polygons of every annotator record and derives the training geometry targets (bit-packed
native masks, skeleton anchor points, per-instance ROIs; solar_filaments.geometry.derive_geometry). They are
cached as one .npz per record in <cache_dir>/<fingerprint>/; the fingerprint hashes the annotation file and
the geometry version, so changed labels or geometry code never reuse stale targets. Records whose cache file
exists are skipped. manifest.json (partial_manifest.json with --limit) records counts, bytes and time.

usage (repository root): python scripts/prepare_data.py --config configs/v9_ffinal_a.yaml
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import torch
from solar_filaments.config import read_config
from solar_filaments.data import FilamentDataset

_dataset = None  # per worker process, set by initialize()


def initialize(root, cache):
    """Worker initializer: one FilamentDataset per process, limited to one CPU thread."""
    global _dataset
    torch.set_num_threads(1)
    _dataset = FilamentDataset(root, "all", cache_dir=cache)


def prepare(index):
    """Worker: build (or find) the cache file of record `index`; returns its size in bytes."""
    path = _dataset.prepare(index)
    return path.stat().st_size


def main():
    """Prepare the geometry cache of all records (or the first --limit) in parallel and write the manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flash_large.yaml")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    settings = read_config(args.config)["train"]
    cache = settings.get("cache_dir", "data/cache")
    dataset = FilamentDataset(settings["data_root"], "all", cache_dir=cache)
    count = min(args.limit or len(dataset), len(dataset))
    start, size = time.perf_counter(), 0
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(settings["data_root"], cache)) as pool:
        for i, result in enumerate(pool.map(prepare, range(count), chunksize=1), 1):
            size += result
            if i % 25 == 0 or i == count:
                print(json.dumps({"prepared": i, "total": count, "seconds": round(time.perf_counter() - start, 2)}), flush=True)
    report = {"records": count, "total_records": len(dataset), "cache_fingerprint": dataset.fingerprint,
              "bytes": size, "seconds": time.perf_counter() - start, "source": "segmentation only"}
    folder = Path(cache) / dataset.fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ("manifest.json" if count == len(dataset) else "partial_manifest.json")).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
