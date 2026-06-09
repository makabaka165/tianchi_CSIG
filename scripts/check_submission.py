#!/usr/bin/env python3
"""Validate Tianchi-style anomaly submission layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

VIEWS = range(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=Path("Test_A"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/template_baseline_test_a"))
    parser.add_argument("--image-size", type=int, default=448)
    return parser.parse_args()


def sample_keys(test_dir: Path) -> list[str]:
    keys = []
    for cls in sorted([p for p in test_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        for sample in sorted([p for p in cls.iterdir() if p.is_dir()], key=lambda p: p.name):
            keys.append(f"{cls.name}/{sample.name}")
    return keys


def main() -> int:
    args = parse_args()
    expected = sample_keys(args.test_dir)
    csv_path = args.output_dir / "submission.csv"
    masks_root = args.output_dir / "predicted_masks"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not masks_root.exists():
        raise FileNotFoundError(masks_root)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    got = [row["group_folder"] for row in rows]
    if got != expected:
        missing = sorted(set(expected) - set(got))[:10]
        extra = sorted(set(got) - set(expected))[:10]
        raise AssertionError(f"CSV group_folder mismatch. missing={missing}, extra={extra}")
    for row in rows:
        float(row["anomaly_score"])

    checked_masks = 0
    for key in expected:
        cls, sample = key.split("/", 1)
        for view in VIEWS:
            path = masks_root / cls / sample / f"{view}_mask.png"
            if not path.exists():
                raise FileNotFoundError(path)
            with Image.open(path) as img:
                if img.size != (args.image_size, args.image_size):
                    raise AssertionError(f"Bad mask size {img.size}: {path}")
                if img.mode not in {"L", "I;16", "I"}:
                    raise AssertionError(f"Mask is not grayscale-like ({img.mode}): {path}")
            checked_masks += 1
    print(f"OK: {len(rows)} submission rows, {checked_masks} masks checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
