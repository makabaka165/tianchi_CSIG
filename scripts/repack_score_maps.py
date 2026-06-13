#!/usr/bin/env python3
"""Rebuild a submission from cached score maps without rerunning DINOv2."""

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
from tqdm import tqdm

from baseline_dinov2 import (
    package_submission,
    percentile_values,
    reduce_view_scores,
    save_uint8_mask,
    top_percent_mean_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack cached anomaly score maps into a submission zip.")
    parser.add_argument("--score-map-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=Path("results/_packages"))
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--mask-low-percentile", type=float, default=70.0)
    parser.add_argument("--mask-high-percentile", type=float, default=99.7)
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--score-reducer", choices=("max", "mean_top2", "mean"), default="max")
    parser.add_argument("--percentile-mode", choices=("fast", "exact"), default="exact")
    parser.add_argument("--mask-workers", type=int, default=16)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_score_map_cache(path: Path) -> tuple[dict[str, object], list[tuple[str, str, int]], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
        classes = data["classes"].astype(str)
        samples = data["samples"].astype(str)
        views = data["views"].astype(np.int32)
        score_maps = np.asarray(data["score_maps"], dtype=np.float32)
    if not (len(classes) == len(samples) == len(views) == score_maps.shape[0]):
        raise ValueError("Score-map cache item metadata length does not match score_maps.")
    items = [(str(cls_name), str(sample), int(view)) for cls_name, sample, view in zip(classes, samples, views)]
    return meta, items, score_maps


def score_maps_to_uint8_masks(
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


def grouped_sample_order(items: list[tuple[str, str, int]]) -> list[tuple[str, str]]:
    return sorted({(cls_name, sample) for cls_name, sample, _view in items}, key=lambda x: (x[0].lower(), x[1]))


def repack_submission(
    score_maps: np.ndarray,
    items: list[tuple[str, str, int]],
    args: argparse.Namespace,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks_root = args.output_dir / "predicted_masks"
    csv_path = args.output_dir / "submission.csv"
    view_scores_by_sample: dict[tuple[str, str], list[float]] = {}
    save_futures = []
    post_batch = 128

    with ThreadPoolExecutor(max_workers=args.mask_workers) as executor:
        for start in tqdm(range(0, len(items), post_batch), desc="repack_masks"):
            end = min(start + post_batch, len(items))
            score_tensor = torch.from_numpy(score_maps[start:end]).to(device, non_blocking=True)
            view_scores = top_percent_mean_tensor(score_tensor, args.top_percent).detach().cpu().numpy()
            masks = score_maps_to_uint8_masks(
                score_tensor,
                args.mask_low_percentile,
                args.mask_high_percentile,
                args.percentile_mode,
            )
            for offset, (mask, score) in enumerate(zip(masks, view_scores)):
                cls_name, sample, view = items[start + offset]
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

    ordered_samples = grouped_sample_order(items)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_folder", "anomaly_score"])
        for cls_name, sample in tqdm(ordered_samples, desc="write_csv"):
            view_scores = view_scores_by_sample[(cls_name, sample)]
            writer.writerow([f"{cls_name}/{sample}", f"{reduce_view_scores(view_scores, args.score_reducer):.8f}"])
    return len(ordered_samples)


def score_stats(score_maps: np.ndarray) -> dict[str, float | int]:
    values = score_maps.reshape(score_maps.shape[0], -1)
    image_scores = values.mean(axis=1)
    finite = np.isfinite(score_maps)
    return {
        "score_map_count": int(score_maps.shape[0]),
        "finite_values": int(finite.sum()),
        "nan_or_inf": int((~finite).sum()),
        "min": float(np.nanmin(score_maps)),
        "max": float(np.nanmax(score_maps)),
        "mean": float(np.nanmean(score_maps)),
        "std": float(np.nanstd(score_maps)),
        "image_score_mean": float(np.nanmean(image_scores)),
        "image_score_std": float(np.nanstd(image_scores)),
    }


def run() -> int:
    args = parse_args()
    experiment_name = args.experiment_name or args.output_dir.name
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.package_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"

    config: dict[str, object] = {
        "experiment_name": experiment_name,
        "score_map_cache": str(args.score_map_cache),
        "output_dir": str(args.output_dir),
        "package_dir": str(args.package_dir),
        "mask_low_percentile": args.mask_low_percentile,
        "mask_high_percentile": args.mask_high_percentile,
        "top_percent": args.top_percent,
        "score_reducer": args.score_reducer,
        "percentile_mode": args.percentile_mode,
        "mask_workers": args.mask_workers,
        "png_compress_level": args.png_compress_level,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        start = time.time()
        cache_meta, items, score_maps = load_score_map_cache(args.score_map_cache)
        if not np.isfinite(score_maps).all():
            raise ValueError("Score-map cache contains NaN or Inf values.")
        total = repack_submission(score_maps, items, args)
        package_zip = package_submission(args.output_dir, args.package_dir, experiment_name)
        config.update(
            {
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(time.time() - start, 3),
                "samples_predicted": total,
                "score_map_cache_meta": cache_meta,
                "score_map_stats": score_stats(score_maps),
                "package_zip": str(package_zip),
            }
        )
        print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Done. See {log_path} and {args.output_dir / 'run_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
