#!/usr/bin/env python3
"""DINOv2 MemoryBank / PatchCore style anomaly submission pipeline.

This script keeps the data format and packaging behavior of baseline_dinov2.py,
but scores each patch by nearest-neighbor cosine distance to normal Train patch
tokens instead of distance to per-group mean/std prototypes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from baseline_dinov2 import (
    GLOBAL_CLASS,
    VIEWS,
    extract_patch_tokens,
    git_commit,
    is_cuda_oom,
    iter_image_paths,
    load_model,
    make_loader,
    maybe_select_auto_batches,
    nvidia_smi,
    package_submission,
    parse_float_list,
    parse_int_list,
    percentile_values,
    reduce_view_scores,
    save_uint8_mask,
    top_percent_mean_tensor,
)

Image.MAX_IMAGE_PIXELS = None


@dataclass
class MemoryBank:
    image_size: int
    meta: dict[str, object]
    class_classes: np.ndarray
    class_views: np.ndarray
    class_offsets: np.ndarray
    class_banks: np.ndarray
    global_views: np.ndarray
    global_offsets: np.ndarray
    global_banks: np.ndarray

    def __post_init__(self) -> None:
        self.class_to_index = {
            (str(cls_name), int(view)): idx
            for idx, (cls_name, view) in enumerate(zip(self.class_classes, self.class_views))
        }
        self.global_to_index = {int(view): idx for idx, view in enumerate(self.global_views)}


class BankAccumulator:
    def __init__(self, max_tokens: int, seed: int):
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.parts: list[torch.Tensor] = []
        self.total_tokens = 0
        self.compactions = 0

    def add(self, tokens: torch.Tensor) -> None:
        if tokens.numel() == 0:
            return
        if tokens.dtype != torch.float16:
            tokens = tokens.to(torch.float16)
        self.parts.append(tokens.contiguous())
        self.total_tokens += int(tokens.shape[0])
        if self.total_tokens > self.max_tokens * 2:
            self.compact()

    def compact(self) -> None:
        if not self.parts:
            return
        merged = torch.cat(self.parts, dim=0)
        if merged.shape[0] > self.max_tokens:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + self.compactions)
            keep = torch.randperm(merged.shape[0], generator=generator)[: self.max_tokens]
            keep, _ = torch.sort(keep)
            merged = merged.index_select(0, keep)
        self.parts = [merged.contiguous()]
        self.total_tokens = int(merged.shape[0])
        self.compactions += 1

    def finalize(self) -> torch.Tensor:
        self.compact()
        if not self.parts:
            raise RuntimeError("Cannot finalize an empty MemoryBank accumulator.")
        return self.parts[0].contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv2 ViT-L MemoryBank anomaly baseline.")
    parser.add_argument("--train-dir", type=Path, default=Path("Train"))
    parser.add_argument("--test-dir", type=Path, default=Path("Test_A"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/005_dinov2_vitl14_memorybank_test_a"))
    parser.add_argument("--package-dir", type=Path, default=Path("results/_packages"))
    parser.add_argument("--experiment-name", default="005_dinov2_vitl14_memorybank_test_a")
    parser.add_argument("--model", default="dinov2_vitl14")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--image-sizes", default="448,518,672")
    parser.add_argument("--mask-size", type=int, default=448)
    parser.add_argument("--scale-weights", default="0.25,0.35,0.40")
    parser.add_argument("--global-fallback", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--predict-batch-size", type=int, default=0)
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--batch-candidates", default="8,12,16,24,32,40,48,64,80,96")
    parser.add_argument("--predict-batch-candidates", default="8,12,16,24,32,40,48,64,80,96")
    parser.add_argument("--auto-batch-fallback", type=int, default=8)
    parser.add_argument("--auto-predict-batch-fallback", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--mask-workers", type=int, default=16)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--percentile-mode", choices=("fast", "exact"), default="exact")
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--mask-low-percentile", type=float, default=50.0)
    parser.add_argument("--mask-high-percentile", type=float, default=99.5)
    parser.add_argument("--score-reducer", choices=("max", "mean_top2", "mean"), default="max")
    parser.add_argument("--bank-samples-per-group", type=int, default=4096)
    parser.add_argument("--global-bank-samples-per-view", type=int, default=32768)
    parser.add_argument("--knn-chunk-tokens", type=int, default=8192)
    parser.add_argument("--knn-neighbors", type=int, default=1)
    parser.add_argument("--knn-reducer", choices=("nearest", "mean_topk", "kth"), default="nearest")
    parser.add_argument("--memory-cache", type=Path, default=Path("results/_cache/dinov2_vitl14_memorybank.npz"))
    parser.add_argument("--cache-memory", action="store_true")
    parser.add_argument("--save-score-maps", action="store_true")
    parser.add_argument("--score-map-cache", type=Path, default=Path("results/_cache/dinov2_vitl14_memorybank_score_maps.npz"))
    parser.add_argument("--debug-train-classes", type=int, default=0)
    parser.add_argument("--debug-train-samples-per-class", type=int, default=0)
    parser.add_argument("--debug-test-classes", type=int, default=0)
    parser.add_argument("--debug-test-samples-per-class", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def effective_image_sizes(args: argparse.Namespace) -> list[int]:
    return parse_int_list(args.image_sizes) if args.image_sizes else [args.image_size]


def normalized_scale_weights(args: argparse.Namespace, image_sizes: list[int]) -> list[float]:
    if args.scale_weights:
        weights = parse_float_list(args.scale_weights)
        if len(weights) != len(image_sizes):
            raise ValueError("--scale-weights length must match --image-sizes")
    else:
        weights = [1.0] * len(image_sizes)
    total = sum(weights)
    if total <= 0:
        raise ValueError("--scale-weights must sum to a positive value")
    return [weight / total for weight in weights]


def stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def filter_items(
    items: Iterable[tuple[str, str, int, Path]],
    max_classes: int,
    max_samples_per_class: int,
) -> list[tuple[str, str, int, Path]]:
    all_items = list(items)
    if max_classes <= 0 and max_samples_per_class <= 0:
        return all_items

    classes = sorted({cls_name for cls_name, _sample, _view, _path in all_items}, key=str.lower)
    allowed_classes = set(classes[:max_classes]) if max_classes > 0 else set(classes)
    allowed_samples: dict[str, set[str]] = {cls_name: set() for cls_name in allowed_classes}
    result: list[tuple[str, str, int, Path]] = []
    for item in all_items:
        cls_name, sample, _view, _path = item
        if cls_name not in allowed_classes:
            continue
        samples = allowed_samples[cls_name]
        if max_samples_per_class > 0 and sample not in samples and len(samples) >= max_samples_per_class:
            continue
        samples.add(sample)
        result.append(item)
    return result


def train_items(args: argparse.Namespace) -> list[tuple[str, str, int, Path]]:
    return filter_items(
        iter_image_paths(args.train_dir),
        args.debug_train_classes,
        args.debug_train_samples_per_class,
    )


def test_items(args: argparse.Namespace) -> list[tuple[str, str, int, Path]]:
    return filter_items(
        iter_image_paths(args.test_dir),
        args.debug_test_classes,
        args.debug_test_samples_per_class,
    )


def memory_cache_path(args: argparse.Namespace, image_size: int) -> Path:
    path = args.memory_cache
    sizes = effective_image_sizes(args)
    if len(sizes) == 1:
        return path
    return path.with_name(f"{path.stem}_{image_size}{path.suffix}")


def memory_cache_metadata(args: argparse.Namespace, image_size: int, item_count: int) -> dict[str, object]:
    return {
        "version": 1,
        "model": args.model,
        "image_size": image_size,
        "train_dir": str(args.train_dir),
        "train_item_count": item_count,
        "bank_samples_per_group": args.bank_samples_per_group,
        "global_bank_samples_per_view": args.global_bank_samples_per_view,
        "debug_train_classes": args.debug_train_classes,
        "debug_train_samples_per_class": args.debug_train_samples_per_class,
    }


def pack_accumulators(
    accumulators: dict[tuple[str, int], BankAccumulator],
    feat_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(accumulators, key=lambda key: (key[0].lower(), key[1]))
    classes: list[str] = []
    views: list[int] = []
    offsets = [0]
    arrays: list[np.ndarray] = []
    for cls_name, view in keys:
        bank = accumulators[(cls_name, view)].finalize().numpy()
        classes.append(cls_name)
        views.append(view)
        arrays.append(bank)
        offsets.append(offsets[-1] + bank.shape[0])
    banks = np.concatenate(arrays, axis=0).astype(np.float16, copy=False) if arrays else np.empty((0, feat_dim), dtype=np.float16)
    return (
        np.array(classes),
        np.array(views, dtype=np.int32),
        np.array(offsets, dtype=np.int64),
        banks,
    )


def pack_global_accumulators(
    accumulators: dict[int, BankAccumulator],
    feat_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    views = sorted(accumulators)
    offsets = [0]
    arrays: list[np.ndarray] = []
    for view in views:
        bank = accumulators[view].finalize().numpy()
        arrays.append(bank)
        offsets.append(offsets[-1] + bank.shape[0])
    banks = np.concatenate(arrays, axis=0).astype(np.float16, copy=False) if arrays else np.empty((0, feat_dim), dtype=np.float16)
    return np.array(views, dtype=np.int32), np.array(offsets, dtype=np.int64), banks


def save_memory_cache(bank: MemoryBank, args: argparse.Namespace) -> None:
    if not args.cache_memory:
        return
    path = memory_cache_path(args, bank.image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta_json=np.array(json.dumps(bank.meta, sort_keys=True)),
        class_classes=bank.class_classes,
        class_views=bank.class_views,
        class_offsets=bank.class_offsets,
        class_banks=bank.class_banks,
        global_views=bank.global_views,
        global_offsets=bank.global_offsets,
        global_banks=bank.global_banks,
    )
    print(f"Saved memory cache: {path}", flush=True)


def load_memory_cache(args: argparse.Namespace, image_size: int, item_count: int) -> MemoryBank | None:
    path = memory_cache_path(args, image_size)
    if not args.cache_memory or not path.exists():
        return None
    expected = memory_cache_metadata(args, image_size, item_count)
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            if meta != expected:
                print(f"Memory cache metadata mismatch, rebuilding: {path}", flush=True)
                print(f"expected={expected}", flush=True)
                print(f"found={meta}", flush=True)
                return None
            bank = MemoryBank(
                image_size=image_size,
                meta=meta,
                class_classes=data["class_classes"].astype(str),
                class_views=data["class_views"].astype(np.int32),
                class_offsets=data["class_offsets"].astype(np.int64),
                class_banks=np.ascontiguousarray(data["class_banks"].astype(np.float16, copy=False)),
                global_views=data["global_views"].astype(np.int32),
                global_offsets=data["global_offsets"].astype(np.int64),
                global_banks=np.ascontiguousarray(data["global_banks"].astype(np.float16, copy=False)),
            )
        print(f"Loaded memory cache: {path}", flush=True)
        return bank
    except Exception as exc:
        print(f"Failed to load memory cache {path}, rebuilding: {exc}", flush=True)
        return None


def build_memory_bank(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    image_size: int,
    items: list[tuple[str, str, int, Path]],
) -> MemoryBank:
    if not items:
        raise FileNotFoundError(f"No train images under {args.train_dir}")

    print(f"Building memory bank for scale {image_size}: {len(items)} train images", flush=True)
    class_accumulators: dict[tuple[str, int], BankAccumulator] = {}
    global_accumulators: dict[int, BankAccumulator] = {
        view: BankAccumulator(args.global_bank_samples_per_view, stable_seed(f"global/{view}/{image_size}"))
        for view in VIEWS
    }
    loader = make_loader(
        items,
        image_size,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        device.type == "cuda",
    )
    stage_start = time.time()
    processed = 0
    feat_dim = 0
    patch_tokens = 0

    for indices, images in tqdm(loader, desc=f"bank_train_{image_size}"):
        images = images.to(device, non_blocking=True)
        tokens = extract_patch_tokens(model, images, args.amp)
        patch_tokens = int(tokens.shape[1])
        feat_dim = int(tokens.shape[2])
        tokens_cpu = tokens.detach().to("cpu", dtype=torch.float16)
        for row, index in enumerate(indices.tolist()):
            cls_name, _sample, view, _path = items[int(index)]
            key = (cls_name, int(view))
            if key not in class_accumulators:
                class_accumulators[key] = BankAccumulator(
                    args.bank_samples_per_group,
                    stable_seed(f"{cls_name}/{view}/{image_size}"),
                )
            image_tokens = tokens_cpu[row].clone()
            class_accumulators[key].add(image_tokens)
            global_accumulators[int(view)].add(image_tokens)
        processed += int(indices.numel())

    elapsed = time.time() - stage_start
    print(
        f"bank_train_{image_size}: {processed} images in {elapsed:.1f}s "
        f"({processed / max(elapsed, 1e-6):.2f} img/s), patch_tokens={patch_tokens}, feat_dim={feat_dim}",
        flush=True,
    )
    class_classes, class_views, class_offsets, class_banks = pack_accumulators(class_accumulators, feat_dim)
    global_views, global_offsets, global_banks = pack_global_accumulators(global_accumulators, feat_dim)
    meta = memory_cache_metadata(args, image_size, len(items))
    meta.update(
        {
            "class_group_count": int(len(class_classes)),
            "global_group_count": int(len(global_views)),
            "patch_tokens": patch_tokens,
            "feat_dim": feat_dim,
            "class_bank_tokens": int(class_banks.shape[0]),
            "global_bank_tokens": int(global_banks.shape[0]),
        }
    )
    bank = MemoryBank(
        image_size=image_size,
        meta=meta,
        class_classes=class_classes,
        class_views=class_views,
        class_offsets=class_offsets,
        class_banks=class_banks,
        global_views=global_views,
        global_offsets=global_offsets,
        global_banks=global_banks,
    )
    save_memory_cache(bank, args)
    return bank


def load_or_build_memory_bank(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    image_size: int,
    items: list[tuple[str, str, int, Path]],
) -> MemoryBank:
    cached = load_memory_cache(args, image_size, len(items))
    if cached is not None:
        args.runtime_metrics.setdefault("memory_cache_hit", {})[str(image_size)] = True
        return cached
    torch.cuda.reset_peak_memory_stats(device)
    bank = build_memory_bank(model, device, args, image_size, items)
    args.runtime_metrics.setdefault("memory_cache_hit", {})[str(image_size)] = False
    args.runtime_metrics.setdefault("bank_peak_memory_mb", {})[str(image_size)] = round(
        torch.cuda.max_memory_allocated(device) / 1024**2,
        1,
    )
    return bank


def nearest_neighbor_distance(
    queries: torch.Tensor,
    bank: torch.Tensor,
    chunk_tokens: int,
    neighbors: int,
    reducer: str,
) -> torch.Tensor:
    if bank.numel() == 0:
        raise RuntimeError("Selected MemoryBank is empty.")
    if neighbors <= 0:
        raise ValueError("--knn-neighbors must be positive.")
    distances = torch.empty(queries.shape[0], dtype=torch.float32, device=queries.device)
    bank_t = bank.transpose(0, 1).contiguous()
    topk = max(1, min(int(neighbors), int(bank.shape[0])))
    for start in range(0, queries.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, queries.shape[0])
        sims = queries[start:end].to(bank.dtype) @ bank_t
        if reducer == "nearest" or topk == 1:
            selected_sim = sims.max(dim=1).values.float()
        else:
            top_values = torch.topk(sims, topk, dim=1).values.float()
            if reducer == "mean_topk":
                selected_sim = top_values.mean(dim=1)
            elif reducer == "kth":
                selected_sim = top_values[:, -1]
            else:
                raise ValueError(f"Unsupported --knn-reducer: {reducer}")
        distances[start:end] = torch.clamp(1.0 - selected_sim, min=0.0, max=2.0)
    return distances


def group_slices(offsets: np.ndarray, group_index: int) -> tuple[int, int]:
    return int(offsets[group_index]), int(offsets[group_index + 1])


def selected_bank_for_item(
    bank: MemoryBank,
    item: tuple[str, str, int, Path],
    global_fallback: bool,
) -> tuple[str, int]:
    cls_name, _sample, view, _path = item
    key = (cls_name, int(view))
    if key in bank.class_to_index:
        return "class", bank.class_to_index[key]
    if global_fallback and int(view) in bank.global_to_index:
        return "global", bank.global_to_index[int(view)]
    raise KeyError(f"Missing MemoryBank for {cls_name} view {view}")


def score_batch_with_bank(
    tokens: torch.Tensor,
    batch_items: list[tuple[str, str, int, Path]],
    bank: MemoryBank,
    class_banks_gpu: torch.Tensor,
    global_banks_gpu: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    batch_size, patch_tokens, _feat_dim = tokens.shape
    patch_scores = torch.empty((batch_size, patch_tokens), dtype=torch.float32, device=tokens.device)
    selected: dict[tuple[str, int], list[int]] = {}
    for row, item in enumerate(batch_items):
        selected.setdefault(selected_bank_for_item(bank, item, args.global_fallback), []).append(row)

    for (source, group_index), rows in selected.items():
        row_tensor = torch.tensor(rows, dtype=torch.long, device=tokens.device)
        query = tokens.index_select(0, row_tensor).reshape(-1, tokens.shape[-1])
        if source == "class":
            start, end = group_slices(bank.class_offsets, group_index)
            selected_bank = class_banks_gpu[start:end]
        else:
            start, end = group_slices(bank.global_offsets, group_index)
            selected_bank = global_banks_gpu[start:end]
        distances = nearest_neighbor_distance(
            query,
            selected_bank,
            args.knn_chunk_tokens,
            args.knn_neighbors,
            args.knn_reducer,
        )
        patch_scores.index_copy_(0, row_tensor, distances.reshape(len(rows), patch_tokens))
    return patch_scores


def score_maps_to_uint8_masks_custom(
    score_maps: torch.Tensor,
    low_percentile: float,
    high_percentile: float,
    percentile_mode: str,
) -> np.ndarray:
    flat = score_maps.flatten(1).float()
    lo = percentile_values(flat, low_percentile, percentile_mode).view(-1, 1, 1)
    hi = percentile_values(flat, high_percentile, percentile_mode).view(-1, 1, 1)
    max_values = flat.max(dim=1).values.view(-1, 1, 1)
    hi = torch.where(torch.isfinite(hi) & (hi > lo + 1e-6), hi, max_values)
    valid = hi > lo + 1e-6
    masks = torch.where(valid, torch.clamp((score_maps - lo) / (hi - lo), 0.0, 1.0), torch.zeros_like(score_maps))
    return (masks * 255.0).to(torch.uint8).cpu().numpy()


def save_score_map_cache(
    score_accum: np.ndarray,
    items: list[tuple[str, str, int, Path]],
    args: argparse.Namespace,
    used_scales: list[dict[str, object]],
) -> None:
    if not args.save_score_maps:
        return
    path = args.score_map_cache
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": 1,
        "experiment_name": args.experiment_name,
        "model": args.model,
        "mask_size": args.mask_size,
        "image_sizes": effective_image_sizes(args),
        "used_scales": used_scales,
        "score_dtype": "float16",
        "item_count": len(items),
    }
    np.savez(
        path,
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
        classes=np.array([cls_name for cls_name, _sample, _view, _path in items]),
        samples=np.array([sample for _cls_name, sample, _view, _path in items]),
        views=np.array([view for _cls_name, _sample, view, _path in items], dtype=np.int32),
        score_maps=score_accum.astype(np.float16, copy=False),
    )
    args.runtime_metrics["score_map_cache_path"] = str(path)
    args.runtime_metrics["score_map_cache_mb"] = round(path.stat().st_size / 1024**2, 1)
    print(f"Saved score map cache: {path}", flush=True)


def predict_scale_scores(
    model: torch.nn.Module,
    device: torch.device,
    bank: MemoryBank,
    args: argparse.Namespace,
    image_size: int,
    items: list[tuple[str, str, int, Path]],
    predict_batch_size: int,
) -> np.ndarray:
    patch_grid = image_size // 14
    scale_scores = np.zeros((len(items), args.mask_size, args.mask_size), dtype=np.float32)
    class_banks_gpu = torch.from_numpy(bank.class_banks).to(device, non_blocking=True)
    global_banks_gpu = torch.from_numpy(bank.global_banks).to(device, non_blocking=True)
    loader = make_loader(
        items,
        image_size,
        predict_batch_size,
        args.num_workers,
        args.prefetch_factor,
        device.type == "cuda",
    )
    stage_start = time.time()
    processed = 0
    for indices, images in tqdm(loader, desc=f"bank_test_{image_size}"):
        images = images.to(device, non_blocking=True)
        tokens = extract_patch_tokens(model, images, args.amp)
        batch_items = [items[int(index)] for index in indices]
        patch_scores = score_batch_with_bank(tokens, batch_items, bank, class_banks_gpu, global_banks_gpu, args)
        patch_score_maps = patch_scores.reshape(-1, 1, patch_grid, patch_grid)
        score_maps = F.interpolate(
            patch_score_maps,
            size=(args.mask_size, args.mask_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        scale_scores[indices.numpy()] = score_maps.detach().cpu().numpy()
        processed += int(indices.numel())

    elapsed = time.time() - stage_start
    print(
        f"bank_test_{image_size}: {processed} images in {elapsed:.1f}s "
        f"({processed / max(elapsed, 1e-6):.2f} img/s)",
        flush=True,
    )
    args.runtime_metrics.setdefault("test_features", {})[str(image_size)] = {
        "processed": processed,
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(processed / max(elapsed, 1e-6), 3),
    }
    args.runtime_metrics.setdefault("predict_peak_memory_mb", {})[str(image_size)] = round(
        torch.cuda.max_memory_allocated(device) / 1024**2,
        1,
    )
    del class_banks_gpu, global_banks_gpu
    torch.cuda.empty_cache()
    return scale_scores


def postprocess_submission(
    score_accum: np.ndarray,
    items: list[tuple[str, str, int, Path]],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    items_by_sample: dict[tuple[str, str], list[int]] = {}
    for idx, (cls_name, sample, _view, _path) in enumerate(items):
        items_by_sample.setdefault((cls_name, sample), []).append(idx)
    ordered_samples = sorted(items_by_sample, key=lambda x: (x[0].lower(), x[1]))
    masks_root = args.output_dir / "predicted_masks"
    csv_path = args.output_dir / "submission.csv"
    view_scores_by_sample: dict[tuple[str, str], list[float]] = {}
    save_futures = []
    post_start = time.time()
    post_batch = max(1, min(args.predict_batch_size or args.batch_size, 128))

    with ThreadPoolExecutor(max_workers=args.mask_workers) as executor:
        for start in tqdm(range(0, len(items), post_batch), desc="postprocess_masks"):
            end = min(start + post_batch, len(items))
            score_tensor = torch.from_numpy(score_accum[start:end]).to(device, non_blocking=True)
            view_scores = top_percent_mean_tensor(score_tensor, args.top_percent).detach().cpu().numpy()
            masks = score_maps_to_uint8_masks_custom(
                score_tensor,
                args.mask_low_percentile,
                args.mask_high_percentile,
                args.percentile_mode,
            )
            for offset, (mask, score) in enumerate(zip(masks, view_scores)):
                cls_name, sample, view, _path = items[start + offset]
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
        for future in tqdm(save_futures, desc="save_masks"):
            future.result()

    elapsed = time.time() - post_start
    args.runtime_metrics["postprocess_masks"] = {
        "processed": len(items),
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(len(items) / max(elapsed, 1e-6), 3),
    }

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_folder", "anomaly_score"])
        for cls_name, sample in tqdm(ordered_samples, desc="write_csv"):
            view_scores = view_scores_by_sample[(cls_name, sample)]
            writer.writerow([f"{cls_name}/{sample}", f"{reduce_view_scores(view_scores, args.score_reducer):.8f}"])
    return len(ordered_samples)


def run() -> int:
    args = parse_args()
    args.runtime_metrics = {}
    image_sizes = effective_image_sizes(args)
    scale_weights = normalized_scale_weights(args, image_sizes)
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.package_dir.mkdir(parents=True, exist_ok=True)

    train = train_items(args)
    test = test_items(args)
    if not train:
        raise FileNotFoundError(f"No train images under {args.train_dir}")
    if not test:
        raise FileNotFoundError(f"No test images under {args.test_dir}")

    config = vars(args).copy()
    config.update(
        {
            "effective_image_sizes": image_sizes,
            "normalized_scale_weights": scale_weights,
            "train_images": len(train),
            "test_images": len(test),
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    log_path = args.output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        wall_start = time.time()
        print(json.dumps(config, indent=2, default=str), flush=True)
        print("--- nvidia-smi before ---", flush=True)
        print(nvidia_smi(), flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this pipeline is expected to use GPU.")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model = load_model(args.model, device)

        original_batch_size = args.batch_size
        original_predict_batch_size = args.predict_batch_size
        score_accum = np.zeros((len(test), args.mask_size, args.mask_size), dtype=np.float32)
        used_scales: list[dict[str, object]] = []
        skipped_scales: list[dict[str, object]] = []
        used_weight_sum = 0.0

        for image_size, scale_weight in zip(image_sizes, scale_weights):
            print(f"--- preparing memory scale {image_size} weight={scale_weight:.4f} ---", flush=True)
            args.image_size = image_size
            args.batch_size = original_batch_size
            args.predict_batch_size = original_predict_batch_size
            maybe_select_auto_batches(model, device, args)
            try:
                bank = load_or_build_memory_bank(model, device, args, image_size, train)
                scale_scores = predict_scale_scores(
                    model,
                    device,
                    bank,
                    args,
                    image_size,
                    test,
                    args.predict_batch_size or args.batch_size,
                )
                score_accum += float(scale_weight) * scale_scores
                used_weight_sum += float(scale_weight)
                used_scales.append(
                    {
                        "image_size": image_size,
                        "weight": scale_weight,
                        "batch_size": args.batch_size,
                        "predict_batch_size": args.predict_batch_size or args.batch_size,
                        "class_groups": int(len(bank.class_classes)),
                        "class_bank_tokens": int(bank.class_banks.shape[0]),
                        "global_bank_tokens": int(bank.global_banks.shape[0]),
                    }
                )
                del bank, scale_scores
                torch.cuda.empty_cache()
            except Exception as exc:
                torch.cuda.empty_cache()
                if not is_cuda_oom(exc) or len(image_sizes) <= 1:
                    raise
                msg = f"Skipping scale {image_size} after CUDA OOM: {str(exc).splitlines()[0][:240]}"
                print(msg, flush=True)
                skipped_scales.append({"image_size": image_size, "reason": msg})

        if used_weight_sum <= 0:
            raise RuntimeError("No usable MemoryBank scales were completed.")
        if not math.isclose(used_weight_sum, 1.0):
            score_accum /= used_weight_sum
            for scale_run in used_scales:
                scale_run["weight"] = float(scale_run["weight"]) / used_weight_sum

        save_score_map_cache(score_accum, test, args, used_scales)
        total = postprocess_submission(score_accum, test, args, device)
        package_zip = package_submission(args.output_dir, args.package_dir, args.experiment_name)
        config.update(
            {
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(time.time() - wall_start, 3),
                "samples_predicted": total,
                "used_scales": used_scales,
                "skipped_scales": skipped_scales,
                "selected_batch_size": args.runtime_metrics.get("selected_batch_size", {}),
                "selected_predict_batch_size": args.runtime_metrics.get("selected_predict_batch_size", {}),
                "package_zip": str(package_zip),
                "runtime_metrics": args.runtime_metrics,
                "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
                "max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
            }
        )
        print("--- nvidia-smi after ---", flush=True)
        print(nvidia_smi(), flush=True)
        print(json.dumps(config, indent=2, default=str, ensure_ascii=False), flush=True)
        print(f"Packaged: {package_zip}", flush=True)
        print(f"Done: {total} samples", flush=True)

    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"Done. See {log_path} and {args.output_dir / 'run_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
