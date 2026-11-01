"""Profile forward, full task loss and backward. Never updates parameters."""
import argparse
import json
from pathlib import Path
import time
import torch
from solar_filaments.config import model_config, read_config
from solar_filaments.data import FilamentDataset
from solar_filaments.geometry import derive_geometry
from solar_filaments.losses import FilamentCriterion
from solar_filaments.model import build_model
from solar_filaments.runtime import configure_cuda, compile_training
from solar_filaments.train import load_warmstart
from solar_filaments.piloting import prepare_mask_pilots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flash_large.yaml")
    parser.add_argument('--checkpoint')
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, help="Keep gradients between microbatches, matching the training recipe")
    parser.add_argument("--record", default="worst", help="worst = maximum annotated instance count")
    parser.add_argument("--reserve-adam", action="store_true", help="Reserve two FP32 state tensors per parameter, no optimizer step")
    parser.add_argument("--reserve-ema", action="store_true")
    parser.add_argument("--compile", choices=["off", "aot_eager", "inductor"], default="off")
    parser.add_argument("--output", default="runs/checks/training_profile.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    configure_cuda()
    document = read_config(args.config)
    config = model_config(document)
    if args.context_size:
        config.context_size = args.context_size
    dataset = FilamentDataset(document["train"]["data_root"], "train", geometry=False)
    index = max(range(len(dataset)), key=lambda i: len(dataset.annotations[str(dataset.records[i]["id"])])) if args.record == "worst" else int(args.record)
    record = dataset.records[index]
    image, target = dataset[index]
    image = torch.nn.functional.interpolate(image[None], size=(args.size, args.size), mode="bilinear", align_corners=False, antialias=True)
    masks = torch.nn.functional.interpolate(target["masks"][None].float(), size=(args.size, args.size), mode="nearest")[0].bool().numpy()
    masks = masks[masks.any((-2, -1))]
    target = {k: torch.from_numpy(v).cuda() for k, v in derive_geometry(masks).items()}
    model, load = build_model(config, load_backbone=not bool(args.checkpoint))
    if args.checkpoint:
        source = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        missing = load_warmstart(model, source)
        load = {'source': args.checkpoint, 'source_epoch': source.get('epoch'),
                'source_validation': source.get('validation'), 'new_parameters': missing}
        del source
    model.cuda().train()
    image = image.cuda()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    image = image.repeat(args.batch_size, 1, 1, 1)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    reserved = []
    if args.reserve_adam or args.reserve_ema:
        copies = 2 * int(args.reserve_adam) + int(args.reserve_ema)
        reserved = [torch.zeros_like(p, memory_format=torch.preserve_format) for _ in range(copies) for p in model.parameters()]
    criterion = FilamentCriterion(document["train"]["point_samples"], **document.get("loss", {})).cuda()
    accumulation = args.accumulation or document["train"]["accumulation"]
    if accumulation != 1:
        raise ValueError('Gradient accumulation is disabled')
    if args.compile != "off":
        state = compile_training(model, {"compile": True, "compile_backend": args.compile})
        forward = state["forward"]
    else:
        forward = model
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for repeat in range(args.repeats):
        if repeat % accumulation == 0:
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        pilots = prepare_mask_pilots([target] * args.batch_size, config.denoising_queries)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = forward(image, denoising_masks=pilots) if pilots is not None else forward(image)
        torch.cuda.synchronize()
        after_forward = time.perf_counter()
        losses = criterion(outputs, [target] * args.batch_size)
        loss = sum(losses.values())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss: { {k: float(v.detach()) for k, v in losses.items()} }")
        torch.cuda.synchronize()
        after_loss = time.perf_counter()
        (loss / accumulation).backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        finite = bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
        if not finite:
            raise RuntimeError("Nonfinite gradients")
        timings.append({"forward": after_forward - start, "loss": after_loss - after_forward,
                        "backward": end - after_loss, "total": end - start})
        print(json.dumps({"repeat": repeat, "timing": timings[-1], "peak_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
        last_losses = {k: float(v.detach()) for k, v in losses.items()}
        del outputs, losses, loss
    report = {"optimizer_steps": 0, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "input": args.size, "context": config.context_size, "instances": len(masks), "record": record["id"],
              "parameters": sum(p.numel() for p in model.parameters()), "pretrained": load,
              "reserved_adam": args.reserve_adam, "reserved_ema": args.reserve_ema,
              "accumulation": accumulation, "batch_size": args.batch_size,
              "parameter_gib": parameter_bytes / 2**30, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "physical_gpu_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "reserved_exceeds_physical": torch.cuda.max_memory_reserved() > torch.cuda.get_device_properties(0).total_memory,
              "compile": args.compile, "timings": timings, "losses": last_losses, "all_gradients_finite": True,
              "includes": "forward + full criterion (native quality masks, topology, matching) + backward; excludes optimizer update and data loading"}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
