#!/usr/bin/env python3
"""Fuse two submission directories without using labels or external data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse two Real-IAD submission directories.")
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=Path("results/_packages"))
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--primary-weight", type=float, default=0.6)
    parser.add_argument("--reference-weight", type=float, default=0.4)
    parser.add_argument("--score-primary-weight", type=float, default=None)
    parser.add_argument("--score-reference-weight", type=float, default=None)
    parser.add_argument("--mask-primary-weight", type=float, default=None)
    parser.add_argument("--mask-reference-weight", type=float, default=None)
    parser.add_argument("--score-mode", choices=("class_rank", "global_rank", "raw"), default="class_rank")
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalized_weights(primary_weight: float, reference_weight: float) -> tuple[float, float]:
    total = primary_weight + reference_weight
    if total <= 0:
        raise ValueError("Fusion weights must sum to a positive value.")
    return primary_weight / total, reference_weight / total


def read_submission(path: Path) -> tuple[list[str], dict[str, float]]:
    rows: list[str] = []
    scores: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["group_folder", "anomaly_score"]:
            raise ValueError(f"Unexpected submission header in {path}: {reader.fieldnames}")
        for row in reader:
            group = row["group_folder"]
            rows.append(group)
            scores[group] = float(row["anomaly_score"])
    return rows, scores


def class_name(group_folder: str) -> str:
    return group_folder.split("/", 1)[0]


def average_ranks(groups: list[str], scores: dict[str, float]) -> dict[str, float]:
    n = len(groups)
    if n == 1:
        return {groups[0]: 0.5}
    ordered = sorted(groups, key=lambda group: (scores[group], group))
    ranks: dict[str, float] = {}
    start = 0
    while start < n:
        end = start + 1
        score = scores[ordered[start]]
        while end < n and math.isclose(scores[ordered[end]], score, rel_tol=0.0, abs_tol=1e-12):
            end += 1
        avg_rank = ((start + end - 1) / 2.0) / (n - 1)
        for idx in range(start, end):
            ranks[ordered[idx]] = avg_rank
        start = end
    return ranks


def transform_scores(order: list[str], scores: dict[str, float], mode: str) -> dict[str, float]:
    if mode == "raw":
        return dict(scores)
    if mode == "global_rank":
        return average_ranks(order, scores)
    by_class: dict[str, list[str]] = {}
    for group in order:
        by_class.setdefault(class_name(group), []).append(group)
    transformed: dict[str, float] = {}
    for groups in by_class.values():
        transformed.update(average_ranks(groups, scores))
    return transformed


def write_submission(path: Path, order: list[str], scores: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_folder", "anomaly_score"])
        for group in order:
            writer.writerow([group, f"{scores[group]:.8f}"])


def score_stats(scores: dict[str, float]) -> dict[str, float]:
    values = np.array(list(scores.values()), dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.percentile(values, 50)),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std()),
        "nan_or_inf": int((~np.isfinite(values)).sum()),
    }


def mask_map(root: Path) -> dict[str, Path]:
    mask_root = root / "predicted_masks"
    return {path.relative_to(mask_root).as_posix(): path for path in sorted(mask_root.rglob("*.png"))}


def read_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.array(image, dtype=np.float32)


def save_mask(mask: np.ndarray, path: Path, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path, compress_level=compress_level)


def fuse_masks(
    primary_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    primary_weight: float,
    reference_weight: float,
    compress_level: int,
) -> dict[str, float | int]:
    primary_masks = mask_map(primary_dir)
    reference_masks = mask_map(reference_dir)
    if set(primary_masks) != set(reference_masks):
        missing_ref = sorted(set(primary_masks) - set(reference_masks))[:5]
        missing_primary = sorted(set(reference_masks) - set(primary_masks))[:5]
        raise ValueError(f"Mask sets differ. missing_ref={missing_ref}, missing_primary={missing_primary}")

    means: list[float] = []
    all_black = 0
    max_value = 0
    out_root = output_dir / "predicted_masks"
    for rel_path in tqdm(sorted(primary_masks), desc="fuse_masks"):
        primary_image = Image.open(primary_masks[rel_path]).convert("L")
        size = primary_image.size
        primary = np.array(primary_image, dtype=np.float32)
        reference = read_mask(reference_masks[rel_path], size=size)
        fused = np.clip(primary_weight * primary + reference_weight * reference, 0.0, 255.0)
        fused = np.rint(fused).astype(np.uint8)
        means.append(float(fused.mean()))
        max_value = max(max_value, int(fused.max()))
        if fused.max() == 0:
            all_black += 1
        save_mask(fused, out_root / rel_path, compress_level)
    return {
        "count": len(means),
        "all_black": all_black,
        "max_value": max_value,
        "mean_avg": float(np.mean(means)) if means else 0.0,
        "mean_p50": float(np.percentile(means, 50)) if means else 0.0,
        "mean_p95": float(np.percentile(means, 95)) if means else 0.0,
    }


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


def run() -> int:
    args = parse_args()
    experiment_name = args.experiment_name or args.output_dir.name
    score_primary_weight, score_reference_weight = normalized_weights(
        args.score_primary_weight if args.score_primary_weight is not None else args.primary_weight,
        args.score_reference_weight if args.score_reference_weight is not None else args.reference_weight,
    )
    mask_primary_weight, mask_reference_weight = normalized_weights(
        args.mask_primary_weight if args.mask_primary_weight is not None else args.primary_weight,
        args.mask_reference_weight if args.mask_reference_weight is not None else args.reference_weight,
    )
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / "run.log"
    config: dict[str, object] = {
        "experiment_name": experiment_name,
        "primary_dir": str(args.primary_dir),
        "reference_dir": str(args.reference_dir),
        "output_dir": str(args.output_dir),
        "package_dir": str(args.package_dir),
        "primary_weight": args.primary_weight,
        "reference_weight": args.reference_weight,
        "score_primary_weight": score_primary_weight,
        "score_reference_weight": score_reference_weight,
        "mask_primary_weight": mask_primary_weight,
        "mask_reference_weight": mask_reference_weight,
        "score_mode": args.score_mode,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        start = time.time()
        primary_order, primary_scores = read_submission(args.primary_dir / "submission.csv")
        reference_order, reference_scores = read_submission(args.reference_dir / "submission.csv")
        if primary_order != reference_order:
            raise ValueError("Submission row order differs between primary and reference directories.")

        primary_transformed = transform_scores(primary_order, primary_scores, args.score_mode)
        reference_transformed = transform_scores(reference_order, reference_scores, args.score_mode)
        fused_scores = {
            group: score_primary_weight * primary_transformed[group] + score_reference_weight * reference_transformed[group]
            for group in primary_order
        }
        write_submission(args.output_dir / "submission.csv", primary_order, fused_scores)
        mask_stats = fuse_masks(
            args.primary_dir,
            args.reference_dir,
            args.output_dir,
            mask_primary_weight,
            mask_reference_weight,
            args.png_compress_level,
        )
        package_zip = package_submission(args.output_dir, args.package_dir, experiment_name)
        config.update(
            {
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(time.time() - start, 3),
                "rows": len(primary_order),
                "score_stats": score_stats(fused_scores),
                "mask_stats": mask_stats,
                "package_zip": str(package_zip),
            }
        )
        print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. See {log_path} and {args.output_dir / 'run_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
