"""Single-view inference CLI: instance masks (PNG), predictions.json and a submission-format CSV.

The competition submission uses the 8-view dihedral TTA of scripts/dump_tta_candidates.py and
scripts/sweep_recall_tta.py instead; this entry point is the plain one-forward-per-image path.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .config import checkpoint_config, model_config, read_config
from .model import build_model
from .inference import instance_predictions, pad_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="runs/predictions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-untrained", action="store_true", help="Software demonstration only; not a trained segmentation model")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    document = read_config(args.config)
    config = model_config(document)
    if not args.checkpoint and not args.allow_untrained:
        parser.error("A task checkpoint is required. --allow-untrained only exercises the inference plumbing.")
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not saved.get("task_trained", False) and not args.allow_untrained:
            parser.error("Checkpoint is not marked task_trained; use --allow-untrained only for diagnostics")
        # Architecture follows training; local context size/backend/tiling can be overridden.
        operational = ("context_size", "dcn_backend", "detail_tile", "detail_halo", "backbone_checkpointing", "detail_checkpointing")
        config = checkpoint_config(saved["model_config"], **{key: getattr(config, key) for key in operational})
        model, _ = build_model(config, load_backbone=False)
        model.load_state_dict(saved["model"], strict=True)
    else:
        model, _ = build_model(config, load_backbone=True)
    device = torch.device(args.device)
    model.to(device).eval()
    source = Path(args.input)
    files = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise ValueError("No input images")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    settings = document.get("inference", {}).copy()
    amp = settings.pop("amp", True) and device.type == "cuda"
    task_trained = bool(args.checkpoint and saved.get("task_trained", False))
    manifest = {"task_trained": task_trained, "checkpoint": args.checkpoint,
                "context_size": config.context_size, "predictions": []}
    with torch.inference_mode():
        for path in files:
            image = torch.from_numpy(np.asarray(Image.open(path).convert("L")).copy()).float()[None, None] / 255
            image, original = pad_image(image)
            start = time.perf_counter()
            with torch.autocast(device.type, enabled=amp, dtype=torch.float16):
                output = model(image.to(device), tiled_detail=True)
            predictions, labels = instance_predictions(output, original_size=original, **settings)
            Image.fromarray(labels).save(folder / f"{path.stem}.instances.png")
            entry = {"file_name": path.name, "height": original[0], "width": original[1], "instances": predictions,
                     "seconds": round(time.perf_counter() - start, 3)}
            manifest["predictions"].append(entry)
            print(json.dumps({"file": path.name, "instances": len(predictions), "seconds": entry["seconds"]}), flush=True)
            del output
            # Keep a recoverable result if a long batch is interrupted.
            (folder / "predictions.json").write_text(json.dumps(manifest), encoding="utf-8")
    csv_path = folder / ("submission.csv" if task_trained else "diagnostic_predictions.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        for prediction in manifest["predictions"]:
            for instance in prediction["instances"]:
                writer.writerow({"filament_id": f"{Path(prediction['file_name']).stem}_{instance['instance_id']}",
                                 "segmentation_rle": instance["segmentation"]["counts"]})


if __name__ == "__main__":
    main()
