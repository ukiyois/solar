"""Training entry point (single GPU or torchrun DDP): python -m solar_filaments.train --config <yaml>.

--init-from warm-starts the weights of a previous stage (new optimizer and schedule); every stage of
the final model was trained this way. --resume continues an interrupted run. With `validation: false`
(train_split 'all') the EMA weights of each epoch are saved as final.pt (and final_ep<k>.pt); otherwise
fold-0 validation selects best.pt by the best PQ over the configured operating-point grid.
"""
import argparse
from datetime import timedelta
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .config import checkpoint_config, config_dict, model_config, read_config
from .data import FilamentDataset, collate
from .inference import instance_predictions
from .losses import FilamentCriterion
from .metrics import official_pq, aggregate_pq
from .model import build_model
from .ops.dcnv4 import native_function
from .runtime import configure_cuda, DevicePrefetcher, photometric, ModelEMA, compile_training
from .piloting import prepare_mask_pilots


def seed_everything(seed):
    """Seed Python, NumPy (seed mod 2**32), PyTorch and every CUDA device.

    main() seeds with seed + rank at start-up and with seed + 10000 * epoch + rank before every epoch, so
    ranks draw different random numbers and an epoch's draws do not depend on earlier epochs.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader worker_init_fn: seed Python and NumPy from the worker's torch seed; one CPU thread per worker.

    PyTorch seeds each worker with base_seed + worker_id, so workers do not share random streams.
    FilamentDataset itself draws its augmentation from per-(epoch, index) generators, independent of the
    worker that loads a sample. One intra-op thread per worker avoids oversubscribing the CPU.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def save_atomic(value, destination):
    """torch.save to a temporary file next to `destination` (suffix .tmp), then rename it over the destination.

    A crash or a full disk never leaves a truncated checkpoint, and the previous file survives a failed save.
    Free space is checked first against the bytes of all distinct tensor storages plus 64 MiB (OSError if
    short); the temporary file is always removed.
    """
    destination = Path(destination)
    temporary = destination.with_suffix(".tmp")
    storages = {}
    def measure(item):
        """Record the byte size of every tensor storage in nested dicts, lists and tuples (shared ones once)."""
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values(): measure(child)
        elif isinstance(item, (tuple, list)):
            for child in item: measure(child)
    measure(value)
    required = sum(storages.values()) + 64 * 1024**2
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError(f'Checkpoint needs {required / 1024**3:.2f} GiB free for atomic save: {destination}')
    try:
        torch.save(value, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_warmstart(model, initial):
    """Load a previous stage's weights. Modules introduced by this stage may be missing from the source
    (they keep their initialization); a widened input projection keeps the old channels, new ones at 0."""
    state = dict(initial['model'])
    current = model.state_dict()
    # Modules switched off by this configuration (mask_feedback / auxiliary_heads false) are dropped.
    cut = ('decoder.feedback.', 'decoder.feedback_gain', 'centerline_head.', 'boundary_head.')
    dropped = [k for k in state if k.startswith(cut) and k not in current]
    state = {k: v for k, v in state.items() if k not in dropped}
    # New input channels (limb-flattened detail input) start at exactly zero weight,
    # so the warm-started network computes the same function as the source.
    widened = []
    for key in ('detail.stem.project.0.weight', 'phase_head.weight'):
        if key in state and key in current and state[key].shape != current[key].shape:
            old, new = state[key], torch.zeros_like(current[key])
            if old.shape[0] != new.shape[0] or old.shape[2:] != new.shape[2:] or old.shape[1] > new.shape[1]:
                raise ValueError(f'Cannot widen {key}: {tuple(old.shape)} -> {tuple(new.shape)}')
            new[:, :old.shape[1]] = old
            state[key] = new
            widened.append(key)
    if dropped or widened:
        print(json.dumps({'warmstart_dropped': dropped, 'warmstart_widened': widened}), flush=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = ('decoder.detail_attention.', 'decoder.detail_norm.', 'decoder.detail_gain', 'decoder.pilot_labels.',
               'decoder.seed_query_projection.', 'decoder.seed_query_gain', 'decoder.seed_position_gain')
    if unexpected or any(not k.startswith(allowed) for k in missing):
        raise ValueError(f'Warm-start mismatch: missing={missing}, unexpected={unexpected}')
    return missing


def backbone_layer_id(name, depths):
    """InternImage layer-wise decay ids: stem 0, blocks 1..sum(depths), downsample -> next level."""
    if name.startswith("backbone.encoder.patch_embed."):
        return 0
    if name.startswith("backbone.encoder.levels."):
        parts = name.split(".")
        stage = int(parts[3])
        if parts[4] == "blocks":
            return sum(depths[:stage]) + int(parts[5]) + 1
        return sum(depths[:stage + 1])
    return sum(depths)


def optimizer_groups(model, settings):
    """AdamW parameter groups with layer-wise learning rates and selective weight decay.

    Backbone parameters (backbone.encoder.*) use backbone_learning_rate times stage_lr_decay ** (3 - stage),
    where downsampling layers and stage norms count with the next stage, or, if layer_lr_decay is set, times
    layer_lr_decay per block below the deepest one (backbone_layer_id). The backbone's DCN offset/mask
    projections get a further offset_lr_scale (default 0.1), so the small dataset does not disturb the
    pretrained sampling patterns. All other parameters use learning_rate. Weight decay applies to weight
    matrices and kernels only, not to biases, 1-d (norm, layer-scale) parameters or the query and level
    position embeddings. Parameters with the same (rate, decay) share a group; initial_lr keeps the group's
    base rate for the warmup/cosine schedule.
    """
    groups = {}
    stage_decay = settings.get("stage_lr_decay", .8)
    layer_decay = settings.get("layer_lr_decay")
    depths = getattr(getattr(model, "backbone", None), "depths", [5, 5, 22, 5])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith("backbone.encoder.")
        stage = 0
        if backbone and ".levels." in name:
            stage = int(name.split(".levels.")[1].split(".")[0])
            if ".downsample." in name or ".norm." in name and ".blocks." not in name:
                stage = min(3, stage + 1)
        if backbone and layer_decay:
            # The deepest block keeps backbone_learning_rate; the stem gets decay**sum(depths).
            scale = layer_decay ** (sum(depths) - backbone_layer_id(name, depths))
        else:
            scale = stage_decay ** (3 - stage) if backbone else 1.
        if backbone and ".offset_mask" in name:
            scale *= settings.get("offset_lr_scale", .1)
        decay = parameter.ndim > 1 and not name.endswith("bias") and not any(x in name for x in ("gamma", "query_pos", "level_pos"))
        rate = (settings["backbone_learning_rate"] if backbone else settings["learning_rate"]) * scale
        key = (rate, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": rate, "initial_lr": rate,
                           "weight_decay": settings["weight_decay"] if decay else 0.}
        groups[key]["params"].append(parameter)
    return list(groups.values())


def lr_factor(step, warmup, total, minimum=.02):
    """Learning-rate multiplier: linear warmup to 1 over `warmup` steps, then cosine decay to `minimum`.

    The factor reaches `minimum` at step total - 1 and stays there.
    """
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup - 1))
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def validate(model, dataset, device, inference, workers=0, report_path=None):
    """Official PQ of the model on a dataset at one operating point (the config's `inference` settings).

    Records of the same physical image share one forward (tiled native path, FP16 autocast when amp is on).
    inference.instance_predictions selects the instances; every annotator record is scored separately with
    metrics.official_pq and the counts are pooled with metrics.aggregate_pq. report_path receives the
    per-record rows as JSON. Returns the pooled metrics plus the numbers of records and physical images.
    """
    model.eval()
    scores, by_file, unique_indices = [], {}, []
    for index, record in enumerate(dataset.records):
        name = record["file_name"]
        if name not in by_file:
            by_file[name] = []
            unique_indices.append(index)
        by_file[name].append(record)
    settings = inference.copy()
    amp = settings.pop("amp", True) and device.type == "cuda"
    loader = DataLoader(Subset(dataset, unique_indices), batch_size=1, collate_fn=collate, num_workers=workers, pin_memory=device.type == "cuda")
    for image, targets in loader:
        target = targets[0]
        name = target["file_name"]
        image = image.to(device, non_blocking=True)
        if image.dtype == torch.uint8:
            image = image.float().div_(255)
        with torch.autocast(device.type, enabled=amp, dtype=torch.float16):
            output = model(image, tiled_detail=True)
        predictions, _ = instance_predictions(output, **settings)
        instances = [{'score': p['score'], 'area': p['area']} for p in predictions]
        predictions = [instance["segmentation"] for instance in predictions]
        del output
        for record in by_file[name]:
            truth = []
            for annotation in dataset.annotations[str(record["id"])]:
                rles = mask_utils.frPyObjects(annotation["segmentation"], record["height"], record["width"])
                truth.append(mask_utils.merge(rles) if isinstance(rles, list) else rles)
            scores.append({**official_pq(predictions, truth), 'file_name': name,
                           'record_id': str(record['id']), 'instances': instances})
    if report_path:
        Path(report_path).write_text(json.dumps(scores), encoding='utf-8')
    return {**aggregate_pq(scores), "records": len(scores), "physical_images": len(by_file)}


@torch.inference_mode()
def validate_grid(model, dataset, device, inference, grid, report_path=None):
    """One forward per physical image; exact official PQ for every (score, mask) pair.

    The reference operating point from `inference` is always included, so its
    PQ equals validate() on the same weights. Selecting by the grid maximum
    avoids mistaking an operating-point shift for a quality drop.
    """
    from .fast_eval import query_probabilities, owned_masks, decode_truth, pq_counts, summarize
    model.eval()
    settings = inference.copy()
    amp = settings.pop("amp", True) and device.type == "cuda"
    reference = (float(settings.get("score_threshold", .35)), float(settings.get("mask_threshold", .5)))
    pairs = sorted({reference, *((float(s), float(m)) for s in grid["scores"] for m in grid["masks"])})
    min_area = int(settings.get("min_area", 32))
    retain = float(settings.get("min_retained_fraction", .5))
    by_file, unique_indices = {}, []
    for index, record in enumerate(dataset.records):
        if record["file_name"] not in by_file:
            by_file[record["file_name"]] = []
            unique_indices.append(index)
        by_file[record["file_name"]].append(record)
    rows = {pair: [] for pair in pairs}
    loader = DataLoader(Subset(dataset, unique_indices), batch_size=1, collate_fn=collate, num_workers=2,
                        pin_memory=device.type == "cuda")
    for image, targets in loader:
        name = targets[0]["file_name"]
        image = image.to(device, non_blocking=True)
        if image.dtype == torch.uint8:
            image = image.float().div_(255)
        with torch.autocast(device.type, enabled=amp, dtype=torch.float16):
            output = model(image, tiled_detail=True)
        scores, probabilities = query_probabilities(output, 0, min(p[0] for p in pairs))
        del output
        truths = [(str(r["id"]), decode_truth(dataset.annotations[str(r["id"])], r["height"], r["width"], device))
                  for r in by_file[name]]
        for pair in pairs:
            masks, _ = owned_masks(scores, probabilities, pair[0], pair[1], min_area, retain)
            for record_id, truth in truths:
                rows[pair].append({**pq_counts(masks, truth), "record_id": record_id, "file_name": name})
        del probabilities, truths
    table = {f"s{p[0]:.3f}_m{p[1]:.3f}": summarize(r) for p, r in rows.items()}
    best_key = max(table, key=lambda k: table[k]["pq"])
    reference_key = f"s{reference[0]:.3f}_m{reference[1]:.3f}"
    if report_path:
        Path(report_path).write_text(json.dumps({"table": table, "best": best_key, "reference": reference_key,
            "best_records": rows[pairs[list(table).index(best_key)]]}), encoding="utf-8")
    return {**table[reference_key], "grid_best_pq": table[best_key]["pq"], "grid_best": best_key,
            "grid_best_counts": table[best_key], "records": len(rows[reference]), "physical_images": len(by_file)}


def run_validation(model, dataset, device, document, train, report_path):
    """validate_grid() if train.validation_grid is configured, else validate() at the inference settings."""
    grid = train.get("validation_grid")
    if grid:
        return validate_grid(model, dataset, device, document.get("inference", {}), grid, report_path)
    return validate(model, dataset, device, document.get("inference", {}), train.get("validation_workers", 0),
                    report_path=report_path)


def selection_score(validation):
    """Checkpoint selection: best operating point when the grid ran, else the fixed one."""
    return validation.get("grid_best_pq", validation["pq"])


def main():
    """Train one stage from a YAML config (command line in the module docstring).

    Each epoch passes every labelled record, plus extra_per_epoch pseudo-labelled frames when
    extra_data_roots is set, through photometric jitter, an optional random backbone view size
    (context_sizes), the model (with mask pilots if configured) and FilamentCriterion. Each optimizer step
    clips the gradient norm, follows the warmup/cosine schedule and, unless the FP16 GradScaler skipped the
    step, updates the EMA. Rank 0 writes last.pt every epoch (training weights, optimizer, scaler and EMA
    state, for --resume) and either final.pt (EMA weights, when validation is off) or best.pt (EMA weights
    with the best validation selection score).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flash_large.yaml")
    parser.add_argument("--resume")
    parser.add_argument('--init-from', help='Warm-start model weights; start a new optimizer and schedule')
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Resume model/EMA/epoch from a checkpoint but start fresh AdamW moments")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, help="Smoke test: stop each epoch after this many batches")
    parser.add_argument("--val-limit", type=int, help="Smoke test: validate on the first N physical images")
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error('--resume and --init-from are mutually exclusive')
    document = read_config(args.config)
    config, train = model_config(document), document["train"]
    if args.output:
        train["output"] = args.output
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if train["epochs"] < 0:
        raise ValueError("epochs must be nonnegative; zero performs setup only")
    if train.get("accumulation", 1) != 1:
        raise ValueError("Gradient accumulation is disabled. Increase the actual per-GPU batch instead.")
    train["accumulation"] = 1
    if config.backbone == "large" and (not torch.cuda.is_available() or config.dcn_backend != "native" or native_function() is None):
        raise RuntimeError("Large-model training requires the official accelerated CUDA DCNv4 backend")
    # torchrun sets WORLD_SIZE, RANK and LOCAL_RANK; without them this is a single-process run.
    world, rank, local_rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                timeout=timedelta(minutes=train.get("distributed_timeout_minutes", 60)))
    configure_cuda()
    seed_everything(train["seed"] + rank)
    # train_split 'all' = final model on every labelled observation (no held-out validation).
    train_data = FilamentDataset(train["data_root"], train.get("train_split", "train"), train["fold"], train["folds"],
        train["seed"], augment=True, cache_dir=train.get("cache_dir"), require_cache=train.get("require_cache", True),
        image_uint8=True, rotate_probability=train.get("rotate_probability", 0.0),
        scale_jitter=train.get("scale_jitter", 0.0))
    val_data = FilamentDataset(train["data_root"], "val", train["fold"], train["folds"], train["seed"],
                               geometry=False, image_uint8=True, decode_masks=False)
    if args.val_limit:
        keep = set(sorted({r["file_name"] for r in val_data.records})[:args.val_limit])
        val_data.records = [r for r in val_data.records if r["file_name"] in keep]
    if train_data.require_cache:
        missing = sum(not train_data.cache_path(r).exists() if train_data.cache_path(r) else True for r in train_data.records)
        if missing:
            raise FileNotFoundError(f"{missing} uncached records. Run python scripts/prepare_data.py --config {args.config}")
    if train.get("extra_data_roots"):
        # Pseudo-labelled external observations (same directory layout as the competition data).
        from .data import MixedDataset
        extras = [FilamentDataset(root, "all", seed=train["seed"], augment=True, image_uint8=True,
                                  rotate_probability=train.get("rotate_probability", 0.0),
                                  scale_jitter=train.get("scale_jitter", 0.0))
                  for root in train["extra_data_roots"]]
        train_data = MixedDataset(train_data, extras, train.get("extra_per_epoch", len(train_data)), train["seed"])
        if rank == 0:
            print(json.dumps({"extra_pool": [len(e) for e in extras], "extra_per_epoch": train_data.extra_per_epoch}),
                  flush=True)
    model, load_report = build_model(config, load_backbone=not (args.resume or args.init_from))
    if args.init_from:
        initial = torch.load(args.init_from, map_location='cpu', weights_only=True, mmap=True)
        missing = load_warmstart(model, initial)
        load_report = {'warmstart': args.init_from, 'new_parameters': missing,
                       'source_epoch': initial.get('epoch'), 'source_validation': initial.get('validation')}
        del initial
    resume = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True) if args.resume else None
    if resume:
        # Operational fields (weights path, DCN backend, checkpointing, tiling) may differ; the architecture may not.
        operational = {"pretrained", "dcn_backend", "backbone_checkpointing", "detail_checkpointing", "detail_tile", "detail_halo"}
        if {k: v for k, v in config_dict(checkpoint_config(resume['model_config'])).items() if k not in operational} != {k: v for k, v in config_dict(config).items() if k not in operational}:
            raise ValueError("Resume architecture differs from the saved model")
        model.load_state_dict(resume["model"], strict=True)
    model.to(device)
    fused = train.get("fused_adamw", True) and device.type == "cuda"
    optimizer = torch.optim.AdamW(optimizer_groups(model, train), betas=tuple(train.get("betas", [.9, .999])),
                                  eps=train.get("adam_epsilon", 1e-8), fused=fused)
    amp = train.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=train.get("initial_scale", 4096.))
    ema = ModelEMA(model, train.get("ema_decay", .999), train.get('ema_warmup', 100)) if train.get("ema", True) else None
    start_epoch, global_step, best = 0, 0, -1.
    if resume:
        if not args.reset_optimizer:
            optimizer.load_state_dict(resume["optimizer"])
            scaler.load_state_dict(resume["scaler"])
        start_epoch, global_step, best = resume["epoch"] + 1, resume["global_step"], resume["best_pq"]
        if ema and "ema" in resume:
            ema.load_state_dict(resume["ema"])
        del resume
    loss_settings = document.get("loss", {})
    criterion = FilamentCriterion(train["point_samples"], **loss_settings).to(device)
    sampler = DistributedSampler(train_data, shuffle=True, seed=train["seed"]) if world > 1 else None
    generator = torch.Generator()
    workers = train["workers"]
    extra = {"persistent_workers": True, "prefetch_factor": train.get("prefetch_factor", 2)} if workers else {}
    loader = DataLoader(train_data, batch_size=train["batch_size"], shuffle=sampler is None,
                        sampler=sampler, num_workers=workers, pin_memory=device.type == "cuda", collate_fn=collate,
                        worker_init_fn=seed_worker, generator=generator, drop_last=train.get('drop_last', False), **extra)
    compiled = compile_training(model, train)
    # Keep eager validation with its different tile sizes outside the training
    # graph. The registered forward uses the same original parameter objects.
    eager_forward = model.forward
    if compiled["enabled"]:
        model.forward = compiled.pop("forward")
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                         broadcast_buffers=False, gradient_as_bucket_view=True)
    unwrapped = model.module if world > 1 else model
    folder = Path(train["output"])
    folder.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (folder / "configuration.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        (folder / "split.json").write_text(json.dumps({"train": sorted({r["file_name"] for r in train_data.records}),
            "val": sorted({r["file_name"] for r in val_data.records}), "seed": train["seed"], "fold": train["fold"]}, indent=2))
        print(json.dumps({"backbone": load_report, "parameters": sum(p.numel() for p in model.parameters()),
                          "train_records": len(train_data), "val_records": len(val_data), "compile": compiled,
                          "fused_adamw": fused, "effective_batch": train["batch_size"] * train["accumulation"] * world,
                          "optimizer_groups": [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]}), flush=True)
    accumulation = train["accumulation"]
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = train["epochs"] * updates_per_epoch
    warmup = max(1, train["warmup_epochs"] * updates_per_epoch)
    # Score the starting weights first, so best.pt is replaced only if training improves on them.
    if rank == 0 and train.get('validate_before_training', False) and not args.resume:
        training_forward = unwrapped.forward
        unwrapped.forward = eager_forward
        try:
            initial_validation = run_validation(unwrapped, val_data, device, document, train,
                                                folder / 'validation_initial.json')
        finally:
            unwrapped.forward = training_forward
        best = selection_score(initial_validation)
        print(json.dumps({'initial_validation': initial_validation}), flush=True)
        initial_state = {'model': unwrapped.state_dict(), 'model_config': config_dict(config), 'epoch': -1,
                     'global_step': 0, 'task_trained': bool(args.init_from), 'validation': initial_validation,
                     'loss_settings': loss_settings, 'initialized_from': args.init_from}
        save_atomic(initial_state, folder / 'best.pt')
    if world > 1:
        dist.barrier()
    for epoch in range(start_epoch, train["epochs"]):
        seed_everything(train["seed"] + 10000 * epoch + rank)
        generator.manual_seed(train["seed"] + epoch + rank)
        train_data.set_epoch(epoch)
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums, keys, started, skipped = None, None, time.perf_counter(), 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, (images, targets) in enumerate(DevicePrefetcher(loader, device, train.get("cuda_prefetch", True))):
            if args.max_steps is not None and step >= args.max_steps:
                break
            images = photometric(images, train)
            criterion.set_progress(epoch + step / len(loader))
            # accumulation is fixed to 1 above, so every batch is one optimizer step.
            update = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            sync_context = model.no_sync() if world > 1 and not update else nullcontext()
            with sync_context:
                pilots = prepare_mask_pilots(targets, config.denoising_queries)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    # Multi-scale training: one backbone view size per batch (detail path stays native).
                    view = {"context_size": random.choice(train["context_sizes"])} if train.get("context_sizes") else {}
                    outputs = (model(images, denoising_masks=pilots, **view) if pilots is not None
                               else model(images, **view))
                losses = criterion(outputs, targets)
                loss = sum(losses.values()) / group_size
                # Device-side check: stops on a non-finite loss without a host synchronization every step.
                torch._assert_async(torch.isfinite(loss), "Nonfinite training loss")
                scaler.scale(loss).backward()
            keys = list(losses)
            vector = torch.stack([losses[key].detach() for key in keys])
            sums = vector if sums is None else sums + vector
            if update:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.get("clip_grad_norm", 1.), foreach=True)
                factor = lr_factor(global_step, warmup, total_steps, train.get("minimum_lr_ratio", .02))
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * factor
                # The GradScaler lowers its scale when it skips a step with non-finite gradients; a skipped
                # step advances neither the schedule nor the EMA.
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                success = scaler.get_scale() >= old_scale
                optimizer.zero_grad(set_to_none=True)
                if success:
                    global_step += 1
                    if ema:
                        ema.update(unwrapped)
                else:
                    skipped += 1
            if rank == 0 and (step + 1) % train.get("log_every", 20) == 0:
                print(json.dumps({"epoch": epoch, "batch": step + 1, "batches": len(loader),
                                  "loss": float(vector.sum()), "updates": global_step, "skipped": skipped,
                                  "pq_loss": float(losses["pq"].detach()),
                                  "auxiliary_loss": float(vector.sum() - losses["pq"].detach()),
                                  'pq_diagnostics': {k: float(v) for k, v in criterion.last_stats.items()},
                                  "scale": scaler.get_scale(), "gradient_norm": float(grad_norm),
                                  "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                                  "global_images_per_second": (step + 1) * train["batch_size"] * world / (time.perf_counter() - started),
                                  "images_per_second": (step + 1) * train["batch_size"] / (time.perf_counter() - started)}), flush=True)
            del outputs, losses, loss, images, targets
        # Epoch loss sums averaged over ranks, for the history log.
        if world > 1:
            dist.all_reduce(sums)
            sums /= world
            dist.barrier()
        training_seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        if rank == 0:
            validation = None
            # Called inside ema.apply() for final.pt and best.pt, whose 'model' then holds the EMA weights.
            def metadata():
                return {"model": unwrapped.state_dict(), "model_config": config_dict(config), "epoch": epoch,
                        "global_step": global_step, "task_trained": global_step > 0,
                        'loss_settings': loss_settings, 'initialized_from': args.init_from}
            if not train.get("validation", True):
                # Full-data training: no held-out images; keep the EMA weights of the latest epoch.
                with ema.apply(unwrapped) if ema else nullcontext():
                    save_atomic({**metadata(), "validation": None, "uses_ema": ema is not None}, folder / "final.pt")
                if train.get("keep_epoch_checkpoints", True):
                    # Continuations of a converged model can peak early: keep every epoch's EMA weights.
                    shutil.copyfile(folder / "final.pt", folder / f"final_ep{epoch}.pt")
            elif (epoch + 1) % train["validate_every"] == 0 or epoch + 1 == train["epochs"]:
                training_forward = unwrapped.forward
                unwrapped.forward = eager_forward
                try:
                    with ema.apply(unwrapped) if ema else nullcontext():
                        validation = run_validation(unwrapped, val_data, device, document, train,
                                                    folder / f'validation_epoch_{epoch:03d}.json')
                        event = {'epoch': epoch, 'validation': validation, 'updates': global_step}
                        with (folder / 'validation_history.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(event) + '\n')
                        print(json.dumps({'validation_completed': event}), flush=True)
                        if selection_score(validation) > best:
                            best = selection_score(validation)
                            save_atomic({**metadata(), "validation": validation, "uses_ema": ema is not None}, folder / "best.pt")
                finally:
                    unwrapped.forward = training_forward
            record = {"epoch": epoch, "training_seconds": training_seconds, "seconds": time.perf_counter() - started,
                      "losses": dict(zip(keys, (sums / len(loader)).cpu().tolist())), "validation": validation,
                      "updates": global_step, "skipped_updates": skipped, "peak_training_gib": peak,
                      "learning_rates": [g["lr"] for g in optimizer.param_groups]}
            with (folder / "history.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            # last.pt (for --resume): training weights with optimizer, scaler and EMA state.
            state = {**metadata(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "best_pq": best,
                     "train_settings": train, 'validation': validation, 'inference_settings': document.get('inference', {})}
            if ema:
                state["ema"] = ema.state_dict()
            save_atomic(state, folder / "last.pt")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()
    if train["epochs"] == 0 and rank == 0:
        print(json.dumps({"setup_only": True, "optimizer_steps": 0}), flush=True)


if __name__ == "__main__":
    main()
