#!/usr/bin/env python3
"""DINOv2 feature baseline for Real-IAD Variety style submissions.

The pipeline uses normal Train samples only:
1. Extract DINOv2 patch tokens for each class/view normal image.
2. Build per-class/per-view feature prototypes and standard deviations.
3. Score Test_A patches by normalized distance to the matching normal prototype.
4. Upsample patch anomaly maps to 448x448 masks and produce submission.csv.

This is a first GPU baseline, designed to be deterministic and format-correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
VIEWS = range(5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv2 ViT-S/14 anomaly baseline.")
    parser.add_argument("--train-dir", type=Path, default=Path("Train"))
    parser.add_argument("--test-dir", type=Path, default=Path("Test_A"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/001_dinov2_vits14_patchcore_test_a"))
    parser.add_argument("--package-dir", type=Path, default=Path("results/_packages"))
    parser.add_argument("--experiment-name", default="001_dinov2_vits14_patchcore_test_a")
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--predict-batch-size", type=int, default=0, help="Defaults to --batch-size when 0.")
    parser.add_argument("--auto-batch", action="store_true", help="Probe candidate batch sizes and keep the largest one that fits.")
    parser.add_argument("--batch-candidates", default="128,192,256,320,384,448,512,640,768")
    parser.add_argument("--predict-batch-candidates", default="128,192,256,320,384,448,512,640,768")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--mask-workers", type=int, default=16)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--percentile-mode", choices=("fast", "exact"), default="exact")
    parser.add_argument("--stats-cache", type=Path, default=Path("results/_cache/dinov2_vits14_448_stats.npz"))
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--mask-high-percentile", type=float, default=99.5)
    parser.add_argument("--score-reducer", choices=("max", "mean_top2", "mean"), default="max")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-stats", action="store_true", help="Save/load feature prototypes/std maps through --stats-cache.")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("candidate list is empty")
    return values


def class_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def sample_dirs(class_dir: Path) -> list[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def iter_image_paths(root: Path) -> Iterable[tuple[str, str, int, Path]]:
    for cls in class_dirs(root):
        for sample in sample_dirs(cls):
            for view in VIEWS:
                path = sample / f"{view}.png"
                if not path.exists():
                    raise FileNotFoundError(path)
                yield cls.name, sample.name, view, path


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_rgb(path: Path, image_size: int) -> torch.Tensor:
    tfm = image_transform(image_size)
    with Image.open(path) as img:
        return tfm(img.convert("RGB"))


class ImagePathDataset(Dataset):
    def __init__(self, items: list[tuple[str, str, int, Path]], image_size: int):
        self.items = items
        self.transform = image_transform(image_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        *_meta, path = self.items[index]
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return index, image


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    print(f"Loading {model_name} from torch.hub...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def amp_dtype(amp: str) -> torch.dtype | None:
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    return None


def make_loader(
    items: list[tuple[str, str, int, Path]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ImagePathDataset(items, image_size), **kwargs)


def extract_patch_tokens(model: torch.nn.Module, batch: torch.Tensor, amp: str) -> torch.Tensor:
    with torch.inference_mode():
        dtype = amp_dtype(amp)
        autocast = torch.autocast("cuda", dtype=dtype, enabled=dtype is not None)
        with autocast:
            features = model.forward_features(batch)
        tokens = features["x_norm_patchtokens"]
        return F.normalize(tokens.float(), dim=-1)


def log_stage_speed(stage: str, processed: int, start: float) -> dict[str, float | int]:
    elapsed = time.time() - start
    rate = processed / max(elapsed, 1e-6)
    print(f"{stage}: {processed} images in {elapsed:.1f}s ({rate:.2f} img/s)", flush=True)
    return {
        "processed": processed,
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(rate, 3),
    }


def group_keys_from_items(items: list[tuple[str, str, int, Path]]) -> list[tuple[str, int]]:
    return sorted({(cls_name, view) for cls_name, _sample, view, _path in items}, key=lambda x: (x[0].lower(), x[1]))


def group_ids_for_items(
    items: list[tuple[str, str, int, Path]],
    group_to_id: dict[tuple[str, int], int],
) -> torch.Tensor:
    return torch.tensor([group_to_id[(cls_name, view)] for cls_name, _sample, view, _path in items], dtype=torch.long)


def build_stats(model: torch.nn.Module, device: torch.device, args: argparse.Namespace) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    items = list(iter_image_paths(args.train_dir))
    if not items:
        raise FileNotFoundError(f"No train images under {args.train_dir}")
    print(f"Extracting train features: {len(items)} images", flush=True)

    group_keys = group_keys_from_items(items)
    group_to_id = {key: idx for idx, key in enumerate(group_keys)}
    group_ids_cpu = group_ids_for_items(items, group_to_id)
    sums = None
    sums2 = None
    counts = torch.zeros(len(group_keys), dtype=torch.float32, device=device)
    patch_tokens = None
    feat_dim = None

    loader = make_loader(
        items,
        args.image_size,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        device.type == "cuda",
    )
    stage_start = time.time()
    processed = 0

    for indices, images in tqdm(loader, desc="train_features"):
        images = images.to(device, non_blocking=True)
        tokens = extract_patch_tokens(model, images, args.amp)
        if patch_tokens is None:
            patch_tokens = tokens.shape[1]
            feat_dim = tokens.shape[2]
            sums = torch.zeros((len(group_keys), patch_tokens, feat_dim), dtype=torch.float32, device=device)
            sums2 = torch.zeros_like(sums)
            print(f"Patch tokens={patch_tokens}, feat_dim={feat_dim}", flush=True)
        batch_group_ids = group_ids_cpu[indices].to(device, non_blocking=True)
        assert sums is not None and sums2 is not None
        sums.index_add_(0, batch_group_ids, tokens)
        sums2.index_add_(0, batch_group_ids, tokens * tokens)
        counts.index_add_(0, batch_group_ids, torch.ones_like(batch_group_ids, dtype=torch.float32, device=device))
        processed += int(indices.numel())
    metrics = log_stage_speed("train_features", processed, stage_start)
    args.runtime_metrics["train_features"] = metrics

    stats: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    assert sums is not None and sums2 is not None
    safe_counts = torch.clamp(counts, min=1.0).view(-1, 1, 1)
    means = sums / safe_counts
    var = torch.clamp(sums2 / safe_counts - means * means, min=args.eps * args.eps)
    stds = torch.sqrt(var)
    counts_cpu = counts.to(torch.int32).cpu().numpy()
    means_cpu = means.cpu().numpy()
    stds_cpu = stds.cpu().numpy()
    for idx, (cls_name, view) in enumerate(group_keys):
        stats.setdefault(cls_name, {})[view] = {
            "mean": means_cpu[idx],
            "std": stds_cpu[idx],
            "count": np.array(counts_cpu[idx], dtype=np.int32),
        }
    args.runtime_metrics["train_peak_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
    return stats


def save_uint8_mask(mask: np.ndarray, path: Path, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path, compress_level=compress_level)


def save_mask(score_map: np.ndarray, path: Path, high_percentile: float, compress_level: int) -> None:
    lo = float(np.percentile(score_map, 50.0))
    hi = float(np.percentile(score_map, high_percentile))
    if not math.isfinite(hi) or hi <= lo + 1e-6:
        hi = float(score_map.max())
    if hi <= lo + 1e-6:
        mask = np.zeros(score_map.shape, dtype=np.uint8)
    else:
        mask = np.clip((score_map - lo) / (hi - lo), 0.0, 1.0)
        mask = (mask * 255.0).astype(np.uint8)
    save_uint8_mask(mask, path, compress_level)


def top_percent_mean(values: np.ndarray, percent: float) -> float:
    flat = values.reshape(-1)
    k = max(1, int(round(flat.size * percent / 100.0)))
    if k >= flat.size:
        return float(flat.mean())
    hottest = np.partition(flat, flat.size - k)[flat.size - k :]
    return float(hottest.mean())


def reduce_view_scores(scores: list[float], reducer: str) -> float:
    if reducer == "mean":
        return float(np.mean(scores))
    if reducer == "mean_top2":
        return float(np.mean(sorted(scores, reverse=True)[:2]))
    return float(max(scores))


def stats_group_keys(stats: dict[str, dict[int, dict[str, np.ndarray]]]) -> list[tuple[str, int]]:
    return sorted(
        [(cls_name, view) for cls_name, views in stats.items() for view in views],
        key=lambda x: (x[0].lower(), x[1]),
    )


def stats_to_packed_device(
    stats: dict[str, dict[int, dict[str, np.ndarray]]],
    device: torch.device,
) -> tuple[list[tuple[str, int]], dict[tuple[str, int], int], torch.Tensor, torch.Tensor]:
    group_keys = stats_group_keys(stats)
    group_to_id = {key: idx for idx, key in enumerate(group_keys)}
    means = torch.from_numpy(np.stack([stats[cls_name][view]["mean"] for cls_name, view in group_keys])).to(device, non_blocking=True)
    stds = torch.from_numpy(np.stack([stats[cls_name][view]["std"] for cls_name, view in group_keys])).to(device, non_blocking=True)
    return group_keys, group_to_id, means, stds


def top_percent_mean_tensor(score_maps: torch.Tensor, percent: float) -> torch.Tensor:
    flat = score_maps.flatten(1)
    k = max(1, int(round(flat.shape[1] * percent / 100.0)))
    if k >= flat.shape[1]:
        return flat.mean(dim=1)
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


def percentile_values(flat: torch.Tensor, percentile: float, mode: str) -> torch.Tensor:
    if mode == "exact":
        return torch.quantile(flat, percentile / 100.0, dim=1)
    k = max(1, min(flat.shape[1], int(math.ceil(flat.shape[1] * percentile / 100.0))))
    return torch.kthvalue(flat, k, dim=1).values


def score_maps_to_uint8_masks(score_maps: torch.Tensor, high_percentile: float, percentile_mode: str) -> np.ndarray:
    flat = score_maps.flatten(1).float()
    lo = percentile_values(flat, 50.0, percentile_mode).view(-1, 1, 1)
    hi = percentile_values(flat, high_percentile, percentile_mode).view(-1, 1, 1)
    max_values = flat.max(dim=1).values.view(-1, 1, 1)
    hi = torch.where(torch.isfinite(hi) & (hi > lo + 1e-6), hi, max_values)
    valid = hi > lo + 1e-6
    masks = torch.where(valid, torch.clamp((score_maps - lo) / (hi - lo), 0.0, 1.0), torch.zeros_like(score_maps))
    return (masks * 255.0).to(torch.uint8).cpu().numpy()


def predict(model: torch.nn.Module, device: torch.device, stats: dict[str, dict[int, dict[str, np.ndarray]]], args: argparse.Namespace) -> int:
    items = list(iter_image_paths(args.test_dir))
    items_by_sample: dict[tuple[str, str], list[int]] = {}
    for idx, (cls_name, sample, _view, _path) in enumerate(items):
        items_by_sample.setdefault((cls_name, sample), []).append(idx)
    ordered_samples = sorted(items_by_sample, key=lambda x: (x[0].lower(), x[1]))
    masks_root = args.output_dir / "predicted_masks"
    csv_path = args.output_dir / "submission.csv"
    patch_grid = args.image_size // 14
    predict_batch_size = args.predict_batch_size or args.batch_size
    view_scores_by_sample: dict[tuple[str, str], list[float]] = {}
    _stat_keys, stats_group_to_id, packed_means, packed_stds = stats_to_packed_device(stats, device)
    predict_group_ids = group_ids_for_items(items, stats_group_to_id)

    print(f"Predicting {len(ordered_samples)} samples / {len(items)} images", flush=True)
    loader = make_loader(
        items,
        args.image_size,
        predict_batch_size,
        args.num_workers,
        args.prefetch_factor,
        device.type == "cuda",
    )
    stage_start = time.time()
    processed = 0
    save_futures = []

    with ThreadPoolExecutor(max_workers=args.mask_workers) as executor:
        for indices, images in tqdm(loader, desc="test_features"):
            images = images.to(device, non_blocking=True)
            tokens = extract_patch_tokens(model, images, args.amp)
            batch_items = [items[int(index)] for index in indices]
            batch_group_ids = predict_group_ids[indices].to(device, non_blocking=True)
            mean_tensor = packed_means[batch_group_ids]
            std_tensor = packed_stds[batch_group_ids]
            z = (tokens - mean_tensor) / std_tensor
            patch_score = torch.sqrt(torch.mean(z * z, dim=-1)).reshape(-1, 1, patch_grid, patch_grid)
            score_maps = F.interpolate(
                patch_score,
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            view_scores = top_percent_mean_tensor(score_maps, args.top_percent).detach().cpu().numpy()
            masks = score_maps_to_uint8_masks(score_maps, args.mask_high_percentile, args.percentile_mode)
            for mask, score, (cls_name, sample, view, _path) in zip(masks, view_scores, batch_items):
                view_scores_by_sample.setdefault((cls_name, sample), []).append(float(score))
                save_futures.append(
                    executor.submit(
                        save_uint8_mask,
                        mask,
                        masks_root / cls_name / sample / f"{view}_mask.png",
                        args.png_compress_level,
                    )
                )
            while len(save_futures) > args.mask_workers * 8:
                save_futures.pop(0).result()
            processed += len(batch_items)
        for future in tqdm(save_futures, desc="save_masks"):
            future.result()
    metrics = log_stage_speed("test_features", processed, stage_start)
    args.runtime_metrics["test_features"] = metrics
    args.runtime_metrics["predict_peak_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_folder", "anomaly_score"])
        for cls_name, sample in tqdm(ordered_samples, desc="write_csv"):
            view_scores = view_scores_by_sample[(cls_name, sample)]
            writer.writerow([f"{cls_name}/{sample}", f"{reduce_view_scores(view_scores, args.score_reducer):.8f}"])
    return len(ordered_samples)


def package_submission(output_dir: Path, package_dir: Path, experiment_name: str) -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    local_zip = output_dir / "submission.zip"
    package_zip = package_dir / f"{experiment_name}.zip"
    for zip_path in [local_zip, package_zip]:
        if zip_path.exists():
            zip_path.unlink()
    with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(output_dir / "submission.csv", arcname="submission.csv")
        for path in sorted((output_dir / "predicted_masks").rglob("*.png")):
            zf.write(path, arcname=path.relative_to(output_dir).as_posix())
    shutil.copy2(local_zip, package_zip)
    return package_zip


def train_image_count(train_dir: Path) -> int:
    return sum(1 for _ in iter_image_paths(train_dir))


def stats_cache_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "version": 2,
        "model": args.model,
        "image_size": args.image_size,
        "eps": args.eps,
        "train_dir": str(args.train_dir),
        "train_image_count": train_image_count(args.train_dir),
    }


def load_stats_cache(args: argparse.Namespace) -> dict[str, dict[int, dict[str, np.ndarray]]] | None:
    path = args.stats_cache
    if not args.cache_stats or not path or not path.exists():
        return None
    try:
        expected = stats_cache_metadata(args)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            if meta != expected:
                print(f"Stats cache metadata mismatch, rebuilding: {path}", flush=True)
                print(f"expected={expected}", flush=True)
                print(f"found={meta}", flush=True)
                return None
            classes = data["classes"].astype(str)
            views = data["views"].astype(np.int32)
            means = data["means"]
            stds = data["stds"]
            counts = data["counts"].astype(np.int32)
            stats: dict[str, dict[int, dict[str, np.ndarray]]] = {}
            for idx, (cls_name, view) in enumerate(zip(classes, views)):
                stats.setdefault(str(cls_name), {})[int(view)] = {
                    "mean": means[idx].astype(np.float32, copy=False),
                    "std": stds[idx].astype(np.float32, copy=False),
                    "count": np.array(counts[idx], dtype=np.int32),
                }
        print(f"Loaded stats cache: {path}", flush=True)
        args.runtime_metrics["stats_cache_hit"] = True
        return stats
    except Exception as exc:
        print(f"Failed to load stats cache {path}, rebuilding: {exc}", flush=True)
        return None


def save_stats_cache(stats: dict[str, dict[int, dict[str, np.ndarray]]], args: argparse.Namespace) -> None:
    if not args.cache_stats or not args.stats_cache:
        return
    path = args.stats_cache
    path.parent.mkdir(parents=True, exist_ok=True)
    group_keys = stats_group_keys(stats)
    meta = stats_cache_metadata(args)
    np.savez(
        path,
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
        classes=np.array([cls_name for cls_name, _view in group_keys]),
        views=np.array([view for _cls_name, view in group_keys], dtype=np.int32),
        means=np.stack([stats[cls_name][view]["mean"] for cls_name, view in group_keys]).astype(np.float32, copy=False),
        stds=np.stack([stats[cls_name][view]["std"] for cls_name, view in group_keys]).astype(np.float32, copy=False),
        counts=np.array([stats[cls_name][view]["count"] for cls_name, view in group_keys], dtype=np.int32),
    )
    print(f"Saved stats cache: {path}", flush=True)
    args.runtime_metrics["stats_cache_path"] = str(path)


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def load_probe_images(args: argparse.Namespace, batch_size: int) -> torch.Tensor:
    items = list(iter_image_paths(args.train_dir))
    if len(items) < batch_size:
        repeats = math.ceil(batch_size / max(len(items), 1))
        items = (items * repeats)[:batch_size]
    else:
        items = items[:batch_size]
    probe_loader = make_loader(
        items,
        args.image_size,
        batch_size,
        min(args.num_workers, 8),
        1,
        torch.cuda.is_available(),
    )
    _indices, images = next(iter(probe_loader))
    return images


def score_probe(tokens: torch.Tensor, args: argparse.Namespace) -> None:
    patch_grid = args.image_size // 14
    mean = torch.zeros_like(tokens)
    std = torch.ones_like(tokens)
    z = (tokens - mean) / std
    patch_score = torch.sqrt(torch.mean(z * z, dim=-1)).reshape(-1, 1, patch_grid, patch_grid)
    score_maps = F.interpolate(patch_score, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)[:, 0]
    _scores = top_percent_mean_tensor(score_maps, args.top_percent)
    _masks = score_maps_to_uint8_masks(score_maps, args.mask_high_percentile, args.percentile_mode)


def probe_candidate(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    images_cpu: torch.Tensor,
    batch_size: int,
    mode: str,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        images = images_cpu[:batch_size].to(device, non_blocking=True)
        tokens = extract_patch_tokens(model, images, args.amp)
        if mode == "train":
            sums = torch.zeros((1, tokens.shape[1], tokens.shape[2]), dtype=torch.float32, device=device)
            sums2 = torch.zeros_like(sums)
            group_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            sums.index_add_(0, group_ids, tokens)
            sums2.index_add_(0, group_ids, tokens * tokens)
        else:
            score_probe(tokens, args)
        torch.cuda.synchronize(device)
        peak_mb = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
        return {"batch_size": batch_size, "status": "ok", "peak_memory_mb": peak_mb}
    except Exception as exc:
        torch.cuda.empty_cache()
        if is_cuda_oom(exc):
            return {"batch_size": batch_size, "status": "oom", "message": str(exc).splitlines()[0][:240]}
        raise
    finally:
        torch.cuda.empty_cache()


def select_largest_batch(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    images_cpu: torch.Tensor,
    candidates: list[int],
    mode: str,
    fallback: int,
) -> tuple[int, list[dict[str, object]]]:
    trials = []
    for batch_size in sorted(candidates, reverse=True):
        if batch_size > images_cpu.shape[0]:
            continue
        result = probe_candidate(model, device, args, images_cpu, batch_size, mode)
        trials.append(result)
        print(f"auto_batch {mode}: {result}", flush=True)
        if result["status"] == "ok":
            return batch_size, trials
    print(f"auto_batch {mode}: all candidates failed; falling back to {fallback}", flush=True)
    return fallback, trials


def maybe_select_auto_batches(model: torch.nn.Module, device: torch.device, args: argparse.Namespace) -> None:
    train_selected = args.batch_size
    predict_selected = args.predict_batch_size or args.batch_size
    if args.auto_batch:
        train_candidates = parse_int_list(args.batch_candidates)
        predict_candidates = parse_int_list(args.predict_batch_candidates)
        max_probe = max(max(train_candidates), max(predict_candidates))
        print(f"Loading {max_probe} real images for auto-batch probing", flush=True)
        probe_images = load_probe_images(args, max_probe)
        train_selected, train_trials = select_largest_batch(
            model, device, args, probe_images, train_candidates, "train", fallback=96
        )
        predict_selected, predict_trials = select_largest_batch(
            model, device, args, probe_images, predict_candidates, "predict", fallback=128
        )
        args.runtime_metrics["auto_batch_trials"] = {
            "train": train_trials,
            "predict": predict_trials,
        }
        del probe_images
        torch.cuda.empty_cache()
    args.batch_size = train_selected
    args.predict_batch_size = predict_selected
    args.runtime_metrics["selected_batch_size"] = train_selected
    args.runtime_metrics["selected_predict_batch_size"] = predict_selected
    print(f"Selected batch sizes: train={train_selected}, predict={predict_selected}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def run() -> int:
    args = parse_args()
    args.runtime_metrics = {"stats_cache_hit": False}
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.package_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    log_path = args.output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        wall_start = time.time()
        print(json.dumps(config, indent=2, default=str), flush=True)
        print("--- nvidia-smi before ---", flush=True)
        print(nvidia_smi(), flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this baseline is expected to use GPU.")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model = load_model(args.model, device)
        maybe_select_auto_batches(model, device, args)
        config.update({
            "selected_batch_size": args.runtime_metrics["selected_batch_size"],
            "selected_predict_batch_size": args.runtime_metrics["selected_predict_batch_size"],
        })
        print("--- selected runtime config ---", flush=True)
        print(json.dumps(config, indent=2, default=str), flush=True)
        stats = load_stats_cache(args)
        if stats is None:
            torch.cuda.reset_peak_memory_stats(device)
            stats = build_stats(model, device, args)
            save_stats_cache(stats, args)
        total = predict(model, device, stats, args)
        package_zip = package_submission(args.output_dir, args.package_dir, args.experiment_name)
        config.update({
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - wall_start, 3),
            "samples_predicted": total,
            "package_zip": str(package_zip),
            "runtime_metrics": args.runtime_metrics,
            "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
            "max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        })
        print("--- nvidia-smi after ---", flush=True)
        print(nvidia_smi(), flush=True)
        print(f"Packaged: {package_zip}", flush=True)
        print(f"Done: {total} samples", flush=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"Done. See {log_path} and {args.output_dir / 'run_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
