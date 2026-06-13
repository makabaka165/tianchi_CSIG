#!/usr/bin/env python3
"""DINOv2 multilayer MemoryBank anomaly submission pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from baseline_dinov2 import (
    VIEWS,
    amp_dtype,
    git_commit,
    is_cuda_oom,
    iter_image_paths,
    load_probe_images,
    make_loader,
    nvidia_smi,
    package_submission,
    parse_float_list,
    parse_int_list,
    percentile_values,
    reduce_view_scores,
    save_uint8_mask,
    top_percent_mean_tensor,
)
from baseline_dinov2_memorybank import (
    BankAccumulator,
    MemoryBank,
    filter_items,
    group_slices,
    nearest_neighbor_distance,
    pack_accumulators,
    pack_global_accumulators,
    selected_bank_for_item,
    stable_seed,
)

Image.MAX_IMAGE_PIXELS = None


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    cached_repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
    if cached_repo.exists():
        print(f"Loading {model_name} from local torch.hub cache: {cached_repo}", flush=True)
        model = torch.hub.load(str(cached_repo), model_name, source="local")
    else:
        print(f"Loading {model_name} from torch.hub...", flush=True)
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multilayer DINOv2 ViT-L MemoryBank anomaly baseline.")
    parser.add_argument("--train-dir", type=Path, default=Path("Train"))
    parser.add_argument("--test-dir", type=Path, default=Path("Test_A"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/009_dinov2_vitl14_multilayer_memorybank_test_a"))
    parser.add_argument("--package-dir", type=Path, default=Path("results/_packages"))
    parser.add_argument("--experiment-name", default="009_dinov2_vitl14_multilayer_memorybank_test_a")
    parser.add_argument("--model", default="dinov2_vitl14")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--image-sizes", default="518,672,784")
    parser.add_argument("--mask-size", type=int, default=448)
    parser.add_argument("--scale-weights", default="0.25,0.35,0.40")
    parser.add_argument("--feature-layers", default="8,16,24")
    parser.add_argument("--layer-weights", default="0.30,0.35,0.35")
    parser.add_argument("--tta", choices=("none", "hflip", "vflip", "hvflip"), default="none")
    parser.add_argument("--global-fallback", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--predict-batch-size", type=int, default=0)
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--batch-candidates", default="16,24,32,40,48,64")
    parser.add_argument("--predict-batch-candidates", default="16,24,32,40,48,64")
    parser.add_argument("--auto-batch-fallback", type=int, default=16)
    parser.add_argument("--auto-predict-batch-fallback", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--mask-workers", type=int, default=16)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--percentile-mode", choices=("fast", "exact"), default="exact")
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--mask-low-percentile", type=float, default=70.0)
    parser.add_argument("--mask-high-percentile", type=float, default=99.7)
    parser.add_argument("--score-reducer", choices=("max", "mean_top2", "mean"), default="max")
    parser.add_argument("--bank-samples-per-group", type=int, default=4096)
    parser.add_argument("--global-bank-samples-per-view", type=int, default=32768)
    parser.add_argument("--knn-chunk-tokens", type=int, default=2048)
    parser.add_argument("--knn-neighbors", type=int, default=3)
    parser.add_argument("--knn-reducer", choices=("nearest", "mean_topk", "kth"), default="mean_topk")
    parser.add_argument("--memory-cache", type=Path, default=Path("results/_cache/dinov2_vitl14_multilayer_memorybank.npz"))
    parser.add_argument("--cache-memory", action="store_true")
    parser.add_argument("--save-score-maps", action="store_true")
    parser.add_argument("--score-map-cache", type=Path, default=Path("results/_cache/009_dinov2_vitl14_multilayer_memorybank_score_maps.npz"))
    parser.add_argument("--debug-train-classes", type=int, default=0)
    parser.add_argument("--debug-train-samples-per-class", type=int, default=0)
    parser.add_argument("--debug-test-classes", type=int, default=0)
    parser.add_argument("--debug-test-samples-per-class", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def effective_image_sizes(args: argparse.Namespace) -> list[int]:
    return parse_int_list(args.image_sizes) if args.image_sizes else [args.image_size]


def requested_feature_layers(args: argparse.Namespace) -> list[int]:
    layers = parse_int_list(args.feature_layers)
    if not layers:
        raise ValueError("--feature-layers must not be empty")
    if any(layer <= 0 for layer in layers):
        raise ValueError("--feature-layers are 1-based block numbers and must be positive")
    if len(set(layers)) != len(layers):
        raise ValueError("--feature-layers contains duplicates")
    return layers


def normalized_named_weights(raw: str, expected_len: int, name: str) -> list[float]:
    weights = parse_float_list(raw)
    if len(weights) != expected_len:
        raise ValueError(f"{name} length must match its target list")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"{name} must sum to a positive value")
    return [float(weight) / total for weight in weights]


def normalized_scale_weights(args: argparse.Namespace, image_sizes: list[int]) -> list[float]:
    return normalized_named_weights(args.scale_weights, len(image_sizes), "--scale-weights")


def normalized_layer_weights(args: argparse.Namespace, feature_layers: list[int]) -> list[float]:
    return normalized_named_weights(args.layer_weights, len(feature_layers), "--layer-weights")


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


def layer_to_index(feature_layer: int) -> int:
    return int(feature_layer) - 1


def extract_intermediate_patch_tokens(
    model: torch.nn.Module,
    batch: torch.Tensor,
    amp: str,
    feature_layers: list[int],
) -> dict[int, torch.Tensor]:
    block_indices = [layer_to_index(layer) for layer in feature_layers]
    with torch.inference_mode():
        dtype = amp_dtype(amp)
        autocast = torch.autocast("cuda", dtype=dtype, enabled=dtype is not None)
        with autocast:
            outputs = model.get_intermediate_layers(
                batch,
                n=block_indices,
                reshape=False,
                return_class_token=False,
                norm=True,
            )
    if len(outputs) != len(feature_layers):
        raise RuntimeError(f"Expected {len(feature_layers)} intermediate layers, got {len(outputs)}")
    return {
        feature_layer: F.normalize(tokens.float(), dim=-1)
        for feature_layer, tokens in zip(feature_layers, outputs)
    }


def memory_cache_path(args: argparse.Namespace, image_size: int, feature_layer: int) -> Path:
    path = args.memory_cache
    return path.with_name(f"{path.stem}_{image_size}_L{feature_layer}{path.suffix}")


def memory_cache_metadata(
    args: argparse.Namespace,
    image_size: int,
    feature_layer: int,
    item_count: int,
) -> dict[str, object]:
    return {
        "version": 1,
        "model": args.model,
        "image_size": int(image_size),
        "feature_layer": int(feature_layer),
        "feature_layer_index": layer_to_index(feature_layer),
        "train_dir": str(args.train_dir),
        "train_item_count": int(item_count),
        "bank_samples_per_group": int(args.bank_samples_per_group),
        "global_bank_samples_per_view": int(args.global_bank_samples_per_view),
        "debug_train_classes": int(args.debug_train_classes),
        "debug_train_samples_per_class": int(args.debug_train_samples_per_class),
    }


def memory_cache_metadata_matches(meta: dict[str, object], expected: dict[str, object]) -> bool:
    return all(meta.get(key) == value for key, value in expected.items())


def save_memory_cache(bank: MemoryBank, args: argparse.Namespace, feature_layer: int) -> None:
    if not args.cache_memory:
        return
    path = memory_cache_path(args, bank.image_size, feature_layer)
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


def load_memory_cache(
    args: argparse.Namespace,
    image_size: int,
    feature_layer: int,
    item_count: int,
) -> MemoryBank | None:
    path = memory_cache_path(args, image_size, feature_layer)
    if not args.cache_memory or not path.exists():
        return None
    expected = memory_cache_metadata(args, image_size, feature_layer, item_count)
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            if not memory_cache_metadata_matches(meta, expected):
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


def make_layer_accumulators(args: argparse.Namespace, image_size: int, feature_layers: list[int]) -> tuple[
    dict[int, dict[tuple[str, int], BankAccumulator]],
    dict[int, dict[int, BankAccumulator]],
]:
    class_accumulators = {
        layer: {}
        for layer in feature_layers
    }
    global_accumulators = {
        layer: {
            view: BankAccumulator(
                args.global_bank_samples_per_view,
                stable_seed(f"global/{view}/{image_size}/L{layer}"),
            )
            for view in VIEWS
        }
        for layer in feature_layers
    }
    return class_accumulators, global_accumulators


def build_scale_memory_banks(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    image_size: int,
    feature_layers: list[int],
    items: list[tuple[str, str, int, Path]],
) -> dict[int, MemoryBank]:
    if not items:
        raise FileNotFoundError(f"No train images under {args.train_dir}")

    banks: dict[int, MemoryBank] = {}
    missing_layers: list[int] = []
    for layer in feature_layers:
        cached = load_memory_cache(args, image_size, layer, len(items))
        if cached is None:
            missing_layers.append(layer)
        else:
            banks[layer] = cached
            args.runtime_metrics.setdefault("memory_cache_hit", {}).setdefault(str(image_size), {})[str(layer)] = True

    if not missing_layers:
        return banks

    print(
        f"Building multilayer memory bank for scale {image_size}, layers {missing_layers}: {len(items)} train images",
        flush=True,
    )
    class_accumulators, global_accumulators = make_layer_accumulators(args, image_size, missing_layers)
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
    feat_dims: dict[int, int] = {}
    patch_tokens_by_layer: dict[int, int] = {}

    for indices, images in tqdm(loader, desc=f"bank_train_{image_size}_multilayer"):
        images = images.to(device, non_blocking=True)
        layer_tokens = extract_intermediate_patch_tokens(model, images, args.amp, missing_layers)
        for layer, tokens in layer_tokens.items():
            patch_tokens_by_layer[layer] = int(tokens.shape[1])
            feat_dims[layer] = int(tokens.shape[2])
            tokens_cpu = tokens.detach().to("cpu", dtype=torch.float16)
            for row, index in enumerate(indices.tolist()):
                cls_name, _sample, view, _path = items[int(index)]
                key = (cls_name, int(view))
                if key not in class_accumulators[layer]:
                    class_accumulators[layer][key] = BankAccumulator(
                        args.bank_samples_per_group,
                        stable_seed(f"{cls_name}/{view}/{image_size}/L{layer}"),
                    )
                image_tokens = tokens_cpu[row].clone()
                class_accumulators[layer][key].add(image_tokens)
                global_accumulators[layer][int(view)].add(image_tokens)
        processed += int(indices.numel())
        del layer_tokens

    elapsed = time.time() - stage_start
    print(
        f"bank_train_{image_size}_multilayer: {processed} images in {elapsed:.1f}s "
        f"({processed / max(elapsed, 1e-6):.2f} img/s)",
        flush=True,
    )

    for layer in missing_layers:
        feat_dim = feat_dims[layer]
        class_classes, class_views, class_offsets, class_banks = pack_accumulators(class_accumulators[layer], feat_dim)
        global_views, global_offsets, global_banks = pack_global_accumulators(global_accumulators[layer], feat_dim)
        meta = memory_cache_metadata(args, image_size, layer, len(items))
        meta.update(
            {
                "class_group_count": int(len(class_classes)),
                "global_group_count": int(len(global_views)),
                "patch_tokens": int(patch_tokens_by_layer[layer]),
                "feat_dim": int(feat_dim),
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
        save_memory_cache(bank, args, layer)
        banks[layer] = bank
        args.runtime_metrics.setdefault("memory_cache_hit", {}).setdefault(str(image_size), {})[str(layer)] = False

    return banks


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


def prepare_gpu_banks(
    banks: dict[int, MemoryBank],
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    gpu_banks = {}
    for layer, bank in banks.items():
        gpu_banks[layer] = (
            torch.from_numpy(bank.class_banks).to(device, non_blocking=True),
            torch.from_numpy(bank.global_banks).to(device, non_blocking=True),
        )
    return gpu_banks


def normalize_subset_weights(
    feature_layers: list[int],
    all_layers: list[int],
    all_weights: list[float],
) -> list[float]:
    by_layer = dict(zip(all_layers, all_weights))
    weights = [by_layer[layer] for layer in feature_layers]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(feature_layers)] * len(feature_layers)
    return [weight / total for weight in weights]


def score_images_for_layers(
    model: torch.nn.Module,
    images: torch.Tensor,
    batch_items: list[tuple[str, str, int, Path]],
    banks: dict[int, MemoryBank],
    gpu_banks: dict[int, tuple[torch.Tensor, torch.Tensor]],
    feature_layers: list[int],
    layer_weights: list[float],
    args: argparse.Namespace,
    image_size: int,
) -> torch.Tensor:
    patch_grid = image_size // 14
    combined = torch.zeros((images.shape[0], args.mask_size, args.mask_size), dtype=torch.float32, device=images.device)
    layer_tokens = extract_intermediate_patch_tokens(model, images, args.amp, feature_layers)
    for layer, layer_weight in zip(feature_layers, layer_weights):
        tokens = layer_tokens[layer]
        class_banks_gpu, global_banks_gpu = gpu_banks[layer]
        patch_scores = score_batch_with_bank(tokens, batch_items, banks[layer], class_banks_gpu, global_banks_gpu, args)
        patch_score_maps = patch_scores.reshape(-1, 1, patch_grid, patch_grid)
        score_maps = F.interpolate(
            patch_score_maps,
            size=(args.mask_size, args.mask_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        combined += float(layer_weight) * score_maps
    return combined


def predict_scale_scores(
    model: torch.nn.Module,
    device: torch.device,
    banks: dict[int, MemoryBank],
    args: argparse.Namespace,
    image_size: int,
    items: list[tuple[str, str, int, Path]],
    predict_batch_size: int,
    feature_layers: list[int],
    layer_weights: list[float],
) -> np.ndarray:
    scale_scores = np.zeros((len(items), args.mask_size, args.mask_size), dtype=np.float32)
    gpu_banks = prepare_gpu_banks(banks, device)
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
    for indices, images in tqdm(loader, desc=f"bank_test_{image_size}_multilayer_{args.tta}"):
        images = images.to(device, non_blocking=True)
        batch_items = [items[int(index)] for index in indices]

        def predict_augmented(image_batch: torch.Tensor) -> torch.Tensor:
            return score_images_for_layers(
                model,
                image_batch,
                batch_items,
                banks,
                gpu_banks,
                feature_layers,
                layer_weights,
                args,
                image_size,
            )

        score_maps = predict_augmented(images)
        if args.tta in {"hflip", "hvflip"}:
            h_scores = predict_augmented(torch.flip(images, dims=[3]))
            score_maps = score_maps + torch.flip(h_scores, dims=[2])
        if args.tta in {"vflip", "hvflip"}:
            v_scores = predict_augmented(torch.flip(images, dims=[2]))
            score_maps = score_maps + torch.flip(v_scores, dims=[1])
        if args.tta == "hvflip":
            hv_scores = predict_augmented(torch.flip(images, dims=[2, 3]))
            score_maps = score_maps + torch.flip(hv_scores, dims=[1, 2])
        if args.tta == "hflip" or args.tta == "vflip":
            score_maps = score_maps * 0.5
        elif args.tta == "hvflip":
            score_maps = score_maps * 0.25
        scale_scores[indices.numpy()] = score_maps.detach().cpu().numpy()
        processed += int(indices.numel())

    elapsed = time.time() - stage_start
    print(
        f"bank_test_{image_size}_multilayer_{args.tta}: {processed} images in {elapsed:.1f}s "
        f"({processed / max(elapsed, 1e-6):.2f} img/s)",
        flush=True,
    )
    args.runtime_metrics.setdefault("test_features", {})[str(image_size)] = {
        "processed": processed,
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(processed / max(elapsed, 1e-6), 3),
        "layers": feature_layers,
        "tta": args.tta,
    }
    args.runtime_metrics.setdefault("predict_peak_memory_mb", {})[str(image_size)] = round(
        torch.cuda.max_memory_allocated(device) / 1024**2,
        1,
    )
    del gpu_banks
    torch.cuda.empty_cache()
    return scale_scores


def save_score_map_cache(
    score_accum: np.ndarray,
    items: list[tuple[str, str, int, Path]],
    args: argparse.Namespace,
    used_runs: list[dict[str, object]],
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
        "feature_layers": requested_feature_layers(args),
        "tta": args.tta,
        "used_runs": used_runs,
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


def score_probe(tokens_by_layer: dict[int, torch.Tensor], args: argparse.Namespace) -> None:
    for tokens in tokens_by_layer.values():
        flat = tokens.reshape(-1, tokens.shape[-1])
        bank = flat[: min(flat.shape[0], max(args.knn_neighbors, 2048))]
        _dist = nearest_neighbor_distance(flat, bank, args.knn_chunk_tokens, args.knn_neighbors, args.knn_reducer)


def probe_candidate(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    images_cpu: torch.Tensor,
    batch_size: int,
    mode: str,
    feature_layers: list[int],
) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        images = images_cpu[:batch_size].to(device, non_blocking=True)
        tokens_by_layer = extract_intermediate_patch_tokens(model, images, args.amp, feature_layers)
        if mode == "train":
            for tokens in tokens_by_layer.values():
                _compact = tokens.detach().to("cpu", dtype=torch.float16)
        else:
            score_probe(tokens_by_layer, args)
        torch.cuda.synchronize(device)
        peak_mb = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
        return {"batch_size": batch_size, "status": "ok", "peak_memory_mb": peak_mb, "layers": feature_layers}
    except Exception as exc:
        torch.cuda.empty_cache()
        if is_cuda_oom(exc):
            return {"batch_size": batch_size, "status": "oom", "message": str(exc).splitlines()[0][:240], "layers": feature_layers}
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
    feature_layers: list[int],
) -> tuple[int, list[dict[str, object]]]:
    trials = []
    for batch_size in sorted(candidates, reverse=True):
        if batch_size > images_cpu.shape[0]:
            continue
        result = probe_candidate(model, device, args, images_cpu, batch_size, mode, feature_layers)
        trials.append(result)
        print(f"auto_batch {mode}: {result}", flush=True)
        if result["status"] == "ok":
            return batch_size, trials
    print(f"auto_batch {mode}: all candidates failed; falling back to {fallback}", flush=True)
    return fallback, trials


def maybe_select_multilayer_auto_batches(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    feature_layers: list[int],
) -> None:
    train_selected = args.batch_size
    predict_selected = args.predict_batch_size or args.batch_size
    if args.auto_batch:
        train_candidates = parse_int_list(args.batch_candidates)
        predict_candidates = parse_int_list(args.predict_batch_candidates)
        max_probe = max(max(train_candidates), max(predict_candidates))
        print(f"Loading {max_probe} real images for multilayer auto-batch probing", flush=True)
        probe_images = load_probe_images(args, max_probe)
        train_selected, train_trials = select_largest_batch(
            model, device, args, probe_images, train_candidates, "train", args.auto_batch_fallback, feature_layers
        )
        predict_selected, predict_trials = select_largest_batch(
            model, device, args, probe_images, predict_candidates, "predict", args.auto_predict_batch_fallback, feature_layers
        )
        args.runtime_metrics.setdefault("auto_batch_trials", {})[str(args.image_size)] = {
            "layers": feature_layers,
            "train": train_trials,
            "predict": predict_trials,
        }
        del probe_images
        torch.cuda.empty_cache()
    args.batch_size = train_selected
    args.predict_batch_size = predict_selected
    args.runtime_metrics.setdefault("selected_batch_size", {})[str(args.image_size)] = train_selected
    args.runtime_metrics.setdefault("selected_predict_batch_size", {})[str(args.image_size)] = predict_selected
    print(f"Selected batch sizes: train={train_selected}, predict={predict_selected}", flush=True)


def layer_fallback(feature_layers: list[int]) -> list[int]:
    if len(feature_layers) <= 2:
        return feature_layers
    return feature_layers[1:]


def run_scale_once(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    image_size: int,
    train: list[tuple[str, str, int, Path]],
    test: list[tuple[str, str, int, Path]],
    active_layers: list[int],
    all_layers: list[int],
    all_layer_weights: list[float],
) -> tuple[np.ndarray, list[int], list[float], dict[str, object]]:
    args.image_size = image_size
    maybe_select_multilayer_auto_batches(model, device, args, active_layers)
    torch.cuda.reset_peak_memory_stats(device)
    banks = build_scale_memory_banks(model, device, args, image_size, active_layers, train)
    args.runtime_metrics.setdefault("bank_peak_memory_mb", {})[str(image_size)] = round(
        torch.cuda.max_memory_allocated(device) / 1024**2,
        1,
    )
    active_layer_weights = normalize_subset_weights(active_layers, all_layers, all_layer_weights)
    scale_scores = predict_scale_scores(
        model,
        device,
        banks,
        args,
        image_size,
        test,
        args.predict_batch_size or args.batch_size,
        active_layers,
        active_layer_weights,
    )
    run_meta = {
        "image_size": int(image_size),
        "layers": active_layers,
        "layer_weights": active_layer_weights,
        "batch_size": int(args.batch_size),
        "predict_batch_size": int(args.predict_batch_size or args.batch_size),
        "class_groups": {str(layer): int(len(banks[layer].class_classes)) for layer in active_layers},
        "class_bank_tokens": {str(layer): int(banks[layer].class_banks.shape[0]) for layer in active_layers},
        "global_bank_tokens": {str(layer): int(banks[layer].global_banks.shape[0]) for layer in active_layers},
    }
    del banks
    torch.cuda.empty_cache()
    return scale_scores, active_layers, active_layer_weights, run_meta


def run() -> int:
    args = parse_args()
    args.runtime_metrics = {}
    image_sizes = effective_image_sizes(args)
    scale_weights = normalized_scale_weights(args, image_sizes)
    feature_layers = requested_feature_layers(args)
    layer_weights = normalized_layer_weights(args, feature_layers)
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
            "effective_feature_layers": feature_layers,
            "internal_layer_indices": [layer_to_index(layer) for layer in feature_layers],
            "normalized_layer_weights": layer_weights,
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
        used_runs: list[dict[str, object]] = []
        skipped_runs: list[dict[str, object]] = []
        used_weight_sum = 0.0

        for image_size, scale_weight in zip(image_sizes, scale_weights):
            print(f"--- preparing multilayer memory scale {image_size} weight={scale_weight:.4f} ---", flush=True)
            active_layers = list(feature_layers)
            attempted_fallback = False
            while True:
                args.batch_size = original_batch_size
                args.predict_batch_size = original_predict_batch_size
                try:
                    scale_scores, used_layers, used_layer_weights, run_meta = run_scale_once(
                        model,
                        device,
                        args,
                        image_size,
                        train,
                        test,
                        active_layers,
                        feature_layers,
                        layer_weights,
                    )
                    score_accum += float(scale_weight) * scale_scores
                    used_weight_sum += float(scale_weight)
                    run_meta.update(
                        {
                            "weight": float(scale_weight),
                            "requested_layers": feature_layers,
                            "used_layers": used_layers,
                            "used_layer_weights": used_layer_weights,
                        }
                    )
                    used_runs.append(run_meta)
                    del scale_scores
                    torch.cuda.empty_cache()
                    break
                except Exception as exc:
                    torch.cuda.empty_cache()
                    if not is_cuda_oom(exc):
                        raise
                    msg = str(exc).splitlines()[0][:240]
                    fallback_layers = layer_fallback(active_layers)
                    if not attempted_fallback and fallback_layers != active_layers:
                        print(
                            f"CUDA OOM at scale {image_size} with layers {active_layers}; "
                            f"retrying with layers {fallback_layers}: {msg}",
                            flush=True,
                        )
                        skipped_runs.append(
                            {
                                "image_size": int(image_size),
                                "layers": active_layers,
                                "reason": f"retry_after_oom: {msg}",
                            }
                        )
                        active_layers = fallback_layers
                        attempted_fallback = True
                        continue
                    if len(image_sizes) > 1:
                        print(f"Skipping scale {image_size} after CUDA OOM: {msg}", flush=True)
                        skipped_runs.append(
                            {
                                "image_size": int(image_size),
                                "layers": active_layers,
                                "reason": f"skipped_after_oom: {msg}",
                            }
                        )
                        break
                    raise

        if used_weight_sum <= 0:
            raise RuntimeError("No usable multilayer MemoryBank scales were completed.")
        if not math.isclose(used_weight_sum, 1.0):
            score_accum /= used_weight_sum
            for run_meta in used_runs:
                run_meta["weight"] = float(run_meta["weight"]) / used_weight_sum

        save_score_map_cache(score_accum, test, args, used_runs)
        total = postprocess_submission(score_accum, test, args, device)
        package_zip = package_submission(args.output_dir, args.package_dir, args.experiment_name)
        config.update(
            {
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(time.time() - wall_start, 3),
                "samples_predicted": total,
                "used_runs": used_runs,
                "skipped_runs": skipped_runs,
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
