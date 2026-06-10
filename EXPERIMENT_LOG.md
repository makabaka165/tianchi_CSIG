# Experiment Log

This file records the key ideas, commands, outputs, and leaderboard feedback for each effective submission. Keep entries short but complete enough to resume work quickly.

## 001_dinov2_vits14_patchcore_test_a

- Official score: `58.9440`
- Commit context: initial GPU DINOv2 baseline.
- Submission package: `results/_packages/001_dinov2_vits14_patchcore_test_a.zip`
- Core idea: use official DINOv2 ViT-S/14 public weights to extract Train normal patch tokens, build per-class/per-view normal mean and std prototypes, then score Test_A patches by normalized distance.
- Key parameters: single scale `448`, output mask `448x448`, image score by max over 5 views, mask from per-image score-map percentile normalization.
- Validation: `check_submission.py` passed with 750 CSV rows and 3750 masks.
- Lesson: the simple foundation-model prototype baseline is valid and gives the first usable score.

## 002_dinov2_vits14_autobatch_test_a

- Official score: `58.9442`
- Commit: `99fc165 Add optimized DINOv2 baseline pipeline`
- Submission package: `results/_packages/002_dinov2_vits14_autobatch_test_a.zip`
- Core idea: keep ViT-S/14 but remove speed bottlenecks with auto batch probing, GPU vectorized stats accumulation, GPU postprocessing, and stats cache.
- Key parameters: single scale `448`, `--auto-batch`, selected train/predict batch `256`, bf16 AMP, cached stats, fast percentile mode.
- Runtime: about `76.539s`, peak allocated/reserved CUDA memory about `5901MB / 8354MB`.
- Validation: format and zip checks passed; score nearly identical to 001 but much faster.
- Lesson: speed/GPU utilization improved, but the model capacity and feature scale were still the limiting factors.

## 003_dinov2_vitb14_multiscale_test_a

- Official score: `59.1242`
- Commit: `ade5c51 Add ViT-B multiscale anomaly baseline`
- Submission package: `results/_packages/003_dinov2_vitb14_multiscale_test_a.zip`
- Core idea: switch to public DINOv2 ViT-B/14, add `448` and `518` feature scales, keep output masks at `448x448`, and add global per-view fallback stats for unseen B-list classes.
- Key parameters: `--model dinov2_vitb14`, `--image-sizes 448,518`, `--scale-weights 0.5,0.5`, `--global-fallback`, bf16 AMP, exact percentile mode, auto batch.
- Runtime: about `306.562s`; selected train/predict batch `160` for both scales; peak allocated/reserved CUDA memory about `8303.6MB / 11278.0MB`.
- Proxy checks: 750 finite scores, 3750 readable masks, no all-black masks, zip contained 3751 entries. Score correlation with 002 was about `0.978`, so the model changed ranking while staying stable.
- Lesson: larger DINOv2 capacity plus multiscale features improved the public score; next useful direction is a stronger backbone and safe fusion with 003.

## 004_dinov2_vitl14_vitb_fusion_test_a

- Official score: fused `59.7794`, raw `59.2690`.
- Commit: `86ed1c2 Add ViT-L fusion experiment pipeline`
- Core idea: use public DINOv2 ViT-L/14 with larger multiscale features, then fuse the raw ViT-L submission with the proven 003 ViT-B/14 submission.
- Raw package: `results/_packages/004_dinov2_vitl14_multiscale_raw_test_a.zip`
- Fused package: `results/_packages/004_dinov2_vitl14_vitb_fusion_test_a.zip`
- Key parameters: `--model dinov2_vitl14`, `--image-sizes 448,518,672`, `--scale-weights 0.35,0.35,0.30`, `--global-fallback`, bf16 AMP, exact percentile mode, auto batch candidates `8,12,16,24,32,40,48,64,80,96`.
- Fusion: class-rank fuse image scores and uint8-weighted fuse masks with weights `0.6 * ViT-L + 0.4 * 003`.
- Runtime: raw ViT-L took about `795.145s`; selected train/predict batch `96` for all three scales; peak allocated/reserved CUDA memory about `15059.1MB / 17296.0MB`.
- Validation: raw and fused both passed `check_submission.py`; both zips contain 3751 entries, 3750 masks, and local/remote CRC checks passed.
- Proxy checks: raw/fused both have 750 finite scores, no NaN/Inf, 3750 readable `448x448` masks, and no all-black masks. Fused mask mean sits between raw ViT-L and 003, as expected.
- Delivery: recommended submit file is `004_dinov2_vitl14_vitb_fusion_test_a.zip`; raw ViT-L zip is retained as backup.
- Lesson: ViT-L/14 three-scale features run comfortably on the 4090 with much higher memory use than 003, but output scores are highly correlated with 003 (`~0.989` raw vs 003), so fusion is the safer first submission.

## 005_dinov2_vitl14_memorybank_fusion_test_a

- Official score: pending platform submission.
- Commit: `0b92b63 Add ViT-L memory bank anomaly pipeline`
- Core idea: add a ViT-L/14 multiscale MemoryBank/PatchCore branch using nearest normal patch distance, then fuse it with the strongest 004 fused package.
- Raw package: `results/_packages/005_dinov2_vitl14_memorybank_test_a.zip`
- Fused package: `results/_packages/005_dinov2_vitl14_memorybank_fusion_test_a.zip`
- Key parameters: `--model dinov2_vitl14`, `--image-sizes 448,518,672`, `--scale-weights 0.25,0.35,0.40`, class/view bank `4096`, global/view bank `32768`, bf16 AMP, GPU chunked cosine nearest-neighbor scoring.
- Fusion: class-rank score weights `0.35 * memorybank + 0.65 * 004_fused`; mask weights `0.60 * memorybank + 0.40 * 004_fused`.
- Runtime: raw MemoryBank took about `1351.260s`; selected train/predict batch `96` for all three scales; no scale fallback; peak allocated/reserved CUDA memory about `11545.4MB / 15260.0MB`, with live `nvidia-smi` sampling up to about `15.8GB` and `100%` GPU utilization during 672 KNN.
- Validation: raw and fused both passed `check_submission.py`; both zips contain 3751 entries, 3750 masks, and remote/local CRC checks passed.
- Proxy checks: 750 finite scores, no NaN/Inf, 3750 readable `448x448` masks, and no all-black masks. Raw MemoryBank score correlation with 004 fused is about `0.631`, so it is more complementary than prior ViT-L prototype scores; fused score correlation with 004 fused is about `0.958`.
- Delivery: recommended submit file is `005_dinov2_vitl14_memorybank_fusion_test_a.zip`; raw MemoryBank zip is retained as backup.
- Lesson: nearest-neighbor patch banks create a genuinely different ranking signal and sharper mask prior, but the branch is much slower and cache-heavy; future iterations can tune fusion weights or reduce cache overhead without changing the generated 005 package.
