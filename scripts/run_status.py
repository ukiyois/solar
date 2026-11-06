"""Compact status of training runs: process, validation history, recent losses.

Per run directory: whether the process in <run>/pid is alive (elapsed time from `ps`, so Linux/macOS), every
epoch of validation_history.jsonl (PQ at the fixed operating point and at the grid optimum), the losses of
the last two epochs in history.jsonl, and from the last 400 lines of console.log the latest errors and batch
progress. Run directories are resolved from the repository root.

usage: python scripts/run_status.py runs/<name> [runs/<name> ...]
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Print a status block for every run directory given on the command line."""
    for run in map(Path, sys.argv[1:]):
        pid = (run / "pid").read_text().strip() if (run / "pid").exists() else ""
        elapsed = subprocess.run(["ps", "-p", pid, "-o", "etime="], capture_output=True, text=True).stdout.strip() if pid else ""
        print(f"== {run.name}  {'running ' + elapsed if elapsed else 'not running'}")
        history = run / "validation_history.jsonl"
        if history.exists():
            for line in history.read_text().splitlines():
                event = json.loads(line)
                v, g = event["validation"], event["validation"].get("grid_best_counts", {})
                print(f"  ep{event['epoch']:>2} fixed={v['pq']:.4f} (tp{v['tp']} fp{v['fp']} fn{v['fn']})  "
                      f"grid={v.get('grid_best_pq', v['pq']):.4f} {v.get('grid_best', '')} "
                      f"tp{g.get('tp')} fp{g.get('fp')} fn{g.get('fn')}")
        train = run / "history.jsonl"
        if train.exists():
            for line in train.read_text().splitlines()[-2:]:
                h = json.loads(line)
                losses = h["losses"]
                print(f"  epoch {h['epoch']} {h['training_seconds']:.0f}s " + " ".join(
                    f"{k[:5]}={v:.3f}" for k, v in losses.items() if v) + f" peak={h['peak_training_gib']:.1f}G")
        log = run / "console.log"
        if log.exists():
            tail = log.read_text(errors="replace").splitlines()[-400:]
            errors = [l for l in tail if "Traceback" in l or "out of memory" in l.lower() or "Error:" in l]
            for line in errors[-3:]:
                print("  !", line[:200])
            batches = [l for l in tail if l.startswith('{"epoch"') and '"batch"' in l]
            if batches:
                b = json.loads(batches[-1])
                print(f"  now epoch {b['epoch']} batch {b['batch']}/{b['batches']} img/s {b['images_per_second']:.2f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()
