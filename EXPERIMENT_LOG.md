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

- Official score: fused `69.1309`; raw not submitted.
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

## 006_dinov2_vitl14_memorybank_v2_fusion_test_a

- Official score: fused `69.8682`; 006a `68.9152`; raw fallback not submitted.
- Commit: `8e76a41 Update experiment log for 006 MemoryBank v2 run`.
- Core idea: keep the high-scoring 005 fused package as the anchor, then add a MemoryBank v2 branch with larger banks, top-3 mean nearest-neighbor scoring, and more aggressive mask normalization.
- Low-cost backup package: `results/_packages/006a_existing_memorybank_mask75_fusion_test_a.zip`.
- Raw fallback package: `results/_packages/006_dinov2_vitl14_memorybank_v2_fallback_518_672_test_a.zip`.
- Recommended fused package: `results/_packages/006_dinov2_vitl14_memorybank_v2_fusion_test_a.zip`.
- Full raw attempt: planned scales `518,672,784` with weights `0.25,0.35,0.40`; `518` and `672` completed, but the process exited at the start of the `784` bank stage without producing a full raw zip.
- Final raw used for fusion: fallback scales `518,672` with normalized weights `0.42,0.58`, class/view bank `8192`, global/view bank `65536`, `--knn-neighbors 3`, `--knn-reducer mean_topk`, `--mask-low-percentile 70`, `--mask-high-percentile 99.7`.
- Fusion: recommended 006 uses score weights `0.25 * 006_raw_fallback + 0.75 * 005_fused` and mask weights `0.65 * 006_raw_fallback + 0.35 * 005_fused`; 006a reuses 005 raw with mask weights `0.75 * memorybank + 0.25 * 004_fused`.
- Runtime: raw fallback took about `1243.660s`; selected train/predict batch `96` for both scales; peak allocated/reserved CUDA memory about `13865.4MB / 15962.0MB`.
- Validation: 006a, raw fallback, and recommended fused all passed `check_submission.py`; each zip contains 3751 entries, 3750 masks, and remote/local CRC checks passed.
- Proxy checks: recommended fused has 750 finite scores, no NaN/Inf, 3750 readable `448x448` masks, and no all-black masks. Score correlation with 005 fused is about `0.985`; mask-mean correlation with 005 fused is about `0.940`, so this is a conservative fusion with a changed mask prior.
- Delivery: recommended submit file is `006_dinov2_vitl14_memorybank_v2_fusion_test_a.zip`; backup zips and logs are in `提交结果/006_dinov2_vitl14_memorybank_v2_fusion_test_a/` on both the server and local workspace.
- Lesson: top-k MemoryBank scoring gives a new signal but higher resolution `784` is fragile in this environment; cache metadata matching was also relaxed so future runs can reuse valid bank caches instead of rebuilding when extra derived metadata fields are present.
- Feedback: 006a only changed mask weighting and dropped below both 006 fused and 005 fused, so the next direction should improve score ranking or add a new robust 784-safe branch instead of simply increasing old MemoryBank mask weight.

## 007_dinov2_vitl14_memorybank_v2_784safe_fusion_test_a

- Official score: recommended fused `70.0345`; `007s02_score35_mask50_fusion_test_a.zip` scored `69.7332`.
- Core idea: first sweep conservative fusions that reuse existing 006 raw fallback and 006 fused, then run a safer `784` MemoryBank v2 branch with lower workers, smaller KNN chunks, and lower batch candidates.
- Sweep packages: `results/_packages/007s01_score30_mask55_fusion_test_a.zip`, `results/_packages/007s02_score35_mask50_fusion_test_a.zip`, `results/_packages/007s03_score20_mask65_fusion_test_a.zip`.
- Raw package: `results/_packages/007_dinov2_vitl14_memorybank_v2_784safe_test_a.zip`.
- Recommended fused package: `results/_packages/007_dinov2_vitl14_memorybank_v2_784safe_fusion_test_a.zip`.
- Key parameters: `--image-sizes 518,672,784`, `--scale-weights 0.25,0.35,0.40`, class/view bank `8192`, global/view bank `65536`, `--knn-neighbors 3`, `--knn-reducer mean_topk`, `--knn-chunk-tokens 2048`, `--num-workers 8`, `--prefetch-factor 1`, batch candidates `16,24,32,40,48,64`.
- Runtime: raw 784-safe completed all three scales in about `1019.419s`; `518/672` loaded existing bank caches, `784` built a new cache; selected train/predict batch `64` for every scale; peak allocated/reserved CUDA memory about `13154.6MB / 15228.0MB`.
- Fusion: recommended 007 uses score weights `0.20 * 007_raw + 0.80 * 006_fused` and mask weights `0.50 * 007_raw + 0.50 * 006_fused`.
- Validation: raw, recommended fused, and all three sweep packages passed `check_submission.py`; each zip contains 3751 entries, 3750 masks, and remote/local CRC checks passed.
- Proxy checks: 007 raw score correlation with 006 fused is about `0.797`, so 784-safe adds a new ranking signal; recommended fused score correlation with 006 fused is about `0.995`, so it is intentionally conservative. All fused masks are readable `448x448` and non-black.
- Delivery: recommended submit file is `007_dinov2_vitl14_memorybank_v2_784safe_fusion_test_a.zip`; raw and sweep zips are retained as backups in `????/007_dinov2_vitl14_memorybank_v2_784safe_fusion_test_a/` on both the server and local workspace.
- Lesson: lowering workers, batch candidates, and KNN chunk size made `784` stable without sacrificing all GPU utilization; the next score step should compare official feedback from conservative 007 fused against a slightly more aggressive 007 sweep/raw submission before designing 008.
- Feedback: 007 recommended fused is the current best; 007s02's more aggressive blend dropped below best, so 008 should keep 007 as the anchor and only add new raw MemoryBank signal conservatively.

## 008_dinov2_vitl14_memorybank_dense_k5_fusion_test_a

- Official score: pending platform submission.
- Core idea: keep `007` recommended fused as the anchor, first run low-cost 007 raw/007 best fusion sweeps, then add a denser ViT-L/14 MemoryBank branch with larger banks and top-5 mean KNN at `518,672,784`.
- Low-cost packages: `008a01_score10_mask30_fusion_test_a.zip`, `008a02_score15_mask40_fusion_test_a.zip`, `008a03_score20_mask45_fusion_test_a.zip`, `008a04_score25_mask50_fusion_test_a.zip`.
- Raw dense package: `results/_packages/008_dinov2_vitl14_memorybank_dense_k5_test_a.zip`.
- Recommended fused package: `results/_packages/008b02_score20_mask50_fusion_test_a.zip`; backups are `008b01_score15_mask40_fusion_test_a.zip` and `008b03_score25_mask55_fusion_test_a.zip`.
- Key parameters: `--image-sizes 518,672,784`, `--scale-weights 0.20,0.35,0.45`, class/view bank `12288`, global/view bank `98304`, `--knn-neighbors 5`, `--knn-reducer mean_topk`, `--knn-chunk-tokens 2048`, `--mask-low-percentile 70`, `--mask-high-percentile 99.7`.
- Runtime: raw dense elapsed `2282.687` seconds; selected train batch `{'518': 64, '672': 64, '784': 64}` and predict batch `{'518': 64, '672': 64, '784': 64}`; peak allocated/reserved CUDA memory `15474.6MB / 18014.0MB`.
- Fusion: default recommendation uses score weights `0.20 * 008_dense + 0.80 * 007_best` and mask weights `0.50 * 008_dense + 0.50 * 007_best`.
- Proxy checks: see `results/008_quality_summary.json`; all generated 008 zips passed format and CRC checks before delivery.
- Delivery: recommended submit file is `008b02_score20_mask50_fusion_test_a.zip` under `????/008_dinov2_vitl14_memorybank_dense_k5_fusion_test_a/`.
- Lesson: 008 tests whether more bank density and top-5 smoothing improve ranking beyond the stable 007 anchor; if official score falls, keep 007 as the current best and use 008 raw only as a future fusion component.

