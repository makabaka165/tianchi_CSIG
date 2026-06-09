#!/usr/bin/env python3
"""Template-difference baseline for Real-IAD Variety style submissions.

This is a deliberately simple first-pass baseline:
1. Build per-class/per-view normal templates from Train, which contains normal samples.
2. Score Test_A samples by robust pixel deviation from the normal template.
3. Emit submission.csv and 448x448 grayscale anomaly masks.

It is meant to validate the full competition pipeline before moving to learned features.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VIEWS = range(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a template-difference anomaly baseline.")
    parser.add_argument("--train-dir", type=Path, default=Path("Train"))
    parser.add_argument("--test-dir", type=Path, default=Path("Test_A"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/template_baseline_test_a"))
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--std-floor", type=float, default=0.035, help="Minimum per-pixel std in [0,1] units.")
    parser.add_argument("--top-percent", type=float, default=1.0, help="Percent of hottest pixels used for image score.")
    parser.add_argument("--mask-high-percentile", type=float, default=99.5)
    parser.add_argument("--score-reducer", choices=("max", "mean_top2", "mean"), default="max")
    parser.add_argument("--make-zip", action="store_true", help="Package submission.csv and predicted_masks into a zip.")
    parser.add_argument("--force", action="store_true", help="Remove existing output directory before running.")
    return parser.parse_args()


def class_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def sample_dirs(class_dir: Path) -> list[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def load_gray(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def save_mask(score_map: np.ndarray, path: Path, high_percentile: float) -> None:
    # Robust per-view normalization. This keeps the first-pass masks inspectable
    # even when one sample has a very different raw score range.
    lo = float(np.percentile(score_map, 50.0))
    hi = float(np.percentile(score_map, high_percentile))
    if not math.isfinite(hi) or hi <= lo + 1e-6:
        hi = float(score_map.max())
    if hi <= lo + 1e-6:
        mask = np.zeros(score_map.shape, dtype=np.uint8)
    else:
        mask = np.clip((score_map - lo) / (hi - lo), 0.0, 1.0)
        mask = (mask * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


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


def build_stats(train_dir: Path, size: int, std_floor: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    classes = class_dirs(train_dir)
    if not classes:
        raise FileNotFoundError(f"No class folders found under {train_dir}")

    print(f"Building normal templates from {len(classes)} classes in {train_dir}", flush=True)
    start = time.time()
    for idx, cls in enumerate(classes, 1):
        samples = sample_dirs(cls)
        if not samples:
            raise FileNotFoundError(f"No sample folders found under {cls}")
        sums = np.zeros((5, size, size), dtype=np.float64)
        sums2 = np.zeros((5, size, size), dtype=np.float64)
        counts = np.zeros(5, dtype=np.int64)

        for sample in samples:
            for view in VIEWS:
                image_path = sample / f"{view}.png"
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                arr = load_gray(image_path, size)
                sums[view] += arr
                sums2[view] += arr * arr
                counts[view] += 1

        mean = sums / counts[:, None, None]
        var = np.maximum(sums2 / counts[:, None, None] - mean * mean, 1e-8)
        std = np.maximum(np.sqrt(var), std_floor)
        stats[cls.name] = (mean.astype(np.float32), std.astype(np.float32))

        elapsed = time.time() - start
        print(f"[{idx:02d}/{len(classes)}] stats {cls.name}: {len(samples)} samples, elapsed {elapsed:.1f}s", flush=True)
    return stats


def predict(test_dir: Path, output_dir: Path, stats: dict[str, tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> int:
    masks_root = output_dir / "predicted_masks"
    csv_path = output_dir / "submission.csv"
    classes = class_dirs(test_dir)
    rows: list[tuple[str, float]] = []
    total_samples = sum(len(sample_dirs(cls)) for cls in classes)
    done = 0
    start = time.time()

    print(f"Predicting {total_samples} samples from {test_dir}", flush=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_folder", "anomaly_score"])
        for cls in classes:
            if cls.name not in stats:
                raise KeyError(f"Missing train template for test class {cls.name}")
            mean, std = stats[cls.name]
            for sample in sample_dirs(cls):
                view_scores: list[float] = []
                for view in VIEWS:
                    image_path = sample / f"{view}.png"
                    if not image_path.exists():
                        raise FileNotFoundError(image_path)
                    arr = load_gray(image_path, args.image_size)
                    score_map = np.abs(arr - mean[view]) / std[view]
                    view_scores.append(top_percent_mean(score_map, args.top_percent))
                    save_mask(
                        score_map,
                        masks_root / cls.name / sample.name / f"{view}_mask.png",
                        args.mask_high_percentile,
                    )
                score = reduce_view_scores(view_scores, args.score_reducer)
                group_folder = f"{cls.name}/{sample.name}"
                writer.writerow([group_folder, f"{score:.8f}"])
                rows.append((group_folder, score))
                done += 1
                if done % 50 == 0 or done == total_samples:
                    elapsed = time.time() - start
                    print(f"Predicted {done}/{total_samples} samples, elapsed {elapsed:.1f}s", flush=True)

    if len(rows) != total_samples:
        raise RuntimeError(f"Expected {total_samples} rows, wrote {len(rows)}")
    return total_samples


def make_zip(output_dir: Path) -> Path:
    zip_path = output_dir / "submission_template_baseline_test_a.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(output_dir / "submission.csv", arcname="submission.csv")
        for path in sorted((output_dir / "predicted_masks").rglob("*.png")):
            zf.write(path, arcname=path.relative_to(output_dir).as_posix())
    return zip_path


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = build_stats(args.train_dir, args.image_size, args.std_floor)
    np.savez_compressed(
        args.output_dir / "template_stats.npz",
        classes=np.array(sorted(stats.keys()), dtype=object),
        # Keep stats outside the zip but useful for later analysis/debugging.
        **{f"{name}__mean": stats[name][0] for name in stats},
        **{f"{name}__std": stats[name][1] for name in stats},
    )
    total = predict(args.test_dir, args.output_dir, stats, args)
    if args.make_zip:
        zip_path = make_zip(args.output_dir)
        print(f"Packaged zip: {zip_path}", flush=True)
    print(f"Done. Wrote {total} rows to {args.output_dir / 'submission.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
