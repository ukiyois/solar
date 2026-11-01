"""Single-view fold-0 validation of a checkpoint over the operating-point grid.

--context-size changes the backbone view resolution at inference (the detail
path always runs at native resolution); useful to probe resolution effects.
Also reports PQ split by GT instance area via the best-config records.
"""
import argparse
import json
from pathlib import Path

import torch

from solar_filaments.config import checkpoint_config, read_config
from solar_filaments.data import FilamentDataset
from solar_filaments.model import build_model
from solar_filaments.train import validate_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-config", default="configs/v9_e9.yaml", help="fold definition, inference settings, validation grid")
    args = parser.parse_args()
    document = read_config(args.train_config)
    train = document["train"]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    overrides = dict(dcn_backend="native", backbone_checkpointing=False, detail_checkpointing=False)
    if args.context_size:
        overrides["context_size"] = args.context_size
    config = checkpoint_config(saved["model_config"], **overrides)
    model, _ = build_model(config, load_backbone=False)
    model.load_state_dict(saved["model"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()
    dataset = FilamentDataset(train["data_root"], "val", train["fold"], train["folds"], train["seed"],
                              geometry=False, image_uint8=True, decode_masks=False)
    result = validate_grid(model, dataset, device, document["inference"], train["validation_grid"],
                           report_path=Path(args.output).with_suffix(".records.json"))
    result.update(checkpoint=args.checkpoint, context_size=config.context_size)
    Path(args.output).write_text(json.dumps(result, indent=1))
    g = result["grid_best_counts"]
    print(json.dumps({"context": config.context_size, "fixed_pq": round(result["pq"], 4),
                      "grid_best_pq": round(result["grid_best_pq"], 4), "grid_best": result["grid_best"],
                      "tp": g["tp"], "fp": g["fp"], "fn": g["fn"]}))


if __name__ == "__main__":
    main()
