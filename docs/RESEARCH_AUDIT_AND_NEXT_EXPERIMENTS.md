# GaugeAnything Research Audit & Next Experiments

Date: 2026-06-11  
Scope: objective review after E-mm-1/E-cnt-1/E-mm-3, EDA report, and project-page update.  
Execution environment for serious experiments: DGX Spark at `/home/hwoo_joo/github/GaugeAnything`.

## Executive Judgment

GaugeAnything has a strong and coherent research direction, but the defensible claim is narrower than the
marketing story:

> **Defensible today:** SAM3 can provide useful promptable masks for concrete-noun defects, and our geometry/scale
> core can turn masks/profiles into measurement atoms. We have real-photo known-object consistency and profile-level
> physical crack-width validation.

> **Not yet defensible:** a full field claim of "prompt any crack photo and get calibrated physical width" without
> either ArUco/caliper capture or krkCMd image-level extraction.

The immediate research priority is therefore not adding more demos. It is closing the gap between:

1. image-level promptable segmentation, and
2. physical-unit measurement ground truth.

## Results Reconfirmed

| Track | Current result | What it proves | What it does not prove |
|---|---:|---|---|
| Crack segmentation | SAM3 0.442±0.011 crack-only mIoU, 2.44× adaptive | SAM3 is a justified zero-shot backbone | Not SOTA vs supervised crack nets |
| Measurement fidelity | SAM3 width rel.err 62.9%, adaptive 43.5% | mIoU and measurement quality diverge | Physical mm accuracy |
| M2 refiner v1 | width rel.err 0.730→0.564 on held-out sources | measurement-aware refinement can help | Domain-independent calibration |
| PlaneScale | 50° tilt 19.3%→0.7% | homography scale is necessary | Real field marker robustness |
| Prompt ensemble | fracture/pit 0.0→0.374/0.352 | prompt mapping fixes synonym collapse | Semantic robustness for all industrial terms |
| Matting v2 | real fray IoU 0.949 vs guided 0.860 | directional synthesis can transfer | Real alpha accuracy |
| Coin E-mm-1 | LOO mean error 1.74%, pass@5% 100% | real-photo segmentation→diameter consistency | absolute marker/caliper chain |
| Rebar E-cnt-1 | best prompt MAE 13.2, acc@10% 0% | zero-shot counting gap is real | counting solution |
| Rebar E-cnt-2 | SAHI tiled SAM3 MAE 7.35, acc@10% 20% | tiling partially fixes scale/crowding | dense touching count solved |
| krkCMd E-mm-3 | GaugeProfile+cal 25.9μm on one group split | profile-level physical crack-width MAE | image-level promptable width |

## New Split Robustness Audit: krkCMd

Why: The 25.9μm headline could be accidental if the deterministic group split is unusually easy.

Script:

```bash
cd /home/hwoo_joo/github/GaugeAnything
.venv/bin/python experiments/krkcmd_split_audit.py
```

Output:

- `experiments/results/krkcmd_split_audit.json`

### Robustness Summary

| Split audit | DLM author | AED author | GaugeProfile uncal | GaugeProfile+cal |
|---|---:|---:|---:|---:|
| group 5-fold MAE mean±std | 14.0±2.4μm | 34.4±5.4μm | 35.9±3.1μm | **27.8±2.5μm** |
| group 5-fold min..max | 11.1..18.5 | 26.5..40.1 | 31.3..40.2 | **24.7..31.3** |
| leave-one-stage mean±std | 13.9±1.9 | 34.1±2.3 | 35.6±1.4 | **27.7±2.1** |
| leave-one-series mean±std | 16.7±9.2 | 38.6±13.1 | 39.0±10.0 | **30.7±9.9** |
| leave-one-series worst | 32.2 | 56.8 | 54.2 | **46.7** |

### Interpretation

The original 25.9μm is not a cherry-picked miracle: the 5-fold group average is 27.8±2.5μm.
However, leave-one-series reveals real domain shift. A camera-ready table should report:

- primary: `GaugeProfile+cal 27.8±2.5μm` over group 5-fold
- single split: `25.9μm` as the fold used in E-mm-3
- robustness warning: leave-one-series can degrade to `46.7μm`

This is more credible than using only the best-looking split.

## Main Weaknesses / Possible Cherry-Picking Risks

### 1. krkCMd is profile-level, not promptable image-level

Risk: The project could overstate "promptable crack width in μm."  
Reality: krkCMd table gives cross-crack brightness profiles, not the image-level prompt→mask path.

Mitigation:
- Always label krkCMd as **profile-level physical GT**.
- Next: extract image subset from `krkCMd_images.zip` or use ArUco/caliper capture to close the full chain.

### 2. Coin E-mm-1 validates consistency, not absolute field calibration

Risk: same-denomination leave-one-out can hide global scale errors.  
Reality: It validates SAM3 coin segmentation and equivalent diameter consistency on real photos.

Mitigation:
- Keep the note: marker absolute chain is synthetic/PlaneScale; coin LOO is real-photo consistency.
- Next: SmartDoc or printed board/Aruco capture for absolute-scale real photos.

### 3. M2 refiner v1 has domain-dependent bias

Risk: A single global correction may look good on average but fail per source.  
Reality: bias sign and magnitude vary across sources; CrackTree200 remains very low IoU.

Mitigation:
- M2 v2 should be domain-conditioned, scale-aware, or uncertainty-aware.
- Report per-source, not just overall.

### 4. Soft inspection results use small real subsets

Risk: fray/uneven conclusions could be dataset-specific.  
Reality: the regime idea is strong, but real eval sizes are small.

Mitigation:
- Keep "regime evidence" rather than broad industrial generality.
- Add more texture/fuzzy datasets or collect small in-house validation.

### 5. Rebar counting is a negative result, but sample size is small

Risk: n=20 prompt sweep can miss a workable inference strategy.  
Reality: prompt wording alone failed; SAHI-style tiling improves MAE 13.2→7.35 but dense touching objects
still undercount badly.

Mitigation:
- Report global and tiled results together.
- Train/fine-tune a small detector/density fallback for dense scenes.

### 6. Project-page story asset is AI-generated

Risk: viewers may mistake it for an experiment.  
Mitigation:
- Caption already says it is a visual concept.
- Future: replace with real capture or use only audited experiment panels.

## Model Roadmap

### A. M2 v2: Domain/Scale-Conditioned Measurement Refiner

Goal: make width correction robust across source/domain, not just lower average error.

Inputs:
- image crop
- raw SAM3 mask/logit if available
- skeleton/EDT width profile
- scale token: px/mm or profile μm/px
- optional domain token: source/statistical style embedding

Loss:
- soft Dice or BCE for mask preservation
- width profile loss: `|pred_width - gt_width|`
- bias penalty per mini-batch/source
- uncertainty head: predict measurement interval, calibrated by residuals

Evaluation:
- source-held-out CrackSeg9k
- krkCMd profile-level physical GT (profile head)
- future ArUco/caliper real captures

Success criterion:
- overall width rel.err improves without worsening source worst-case
- per-source bias closer to zero
- uncertainty covers high-error domains

### B. Counting Head / Density Fallback

Goal: turn E-cnt-1 from negative result into an honest improvement track.

Stages:
1. SAHI tiled SAM3 inference on ROI-1555.
2. If still weak, train a small density/centroid head from ROI-1555 masks.
3. Compare: zero-shot, SAHI, density, supervised mask detector.

Evaluation:
- scene-level split, not random image-only if scene leakage is possible
- metrics: MAE, rel.err, acc@10%, exact, under/over-count bias by density bin

Success criterion:
- reduce MAE from 13.2 to below 5 on held-out scenes, or document failure.

### C. Soft Regime Router v2

Goal: make binary/matting/field selection measurable and automatic.

Needed:
- regime classifier from mask/texture statistics
- hold-out threshold selection
- per-regime confidence/uncertainty

Evaluation:
- Magnetic Tile per class
- additional fuzzy/field anomaly datasets if license allows

### D. SAM3 Metadata / Confidence Audit

Goal: know whether confidence/logits are usable for uncertainty.

Existing:
- `experiments/eda_probe.py`

Next:
- run on Spark and record actual SAM3 output schema
- if only instance scores are stable, use score/calibration curves
- if logits/soft masks are exposed, use them for uncertainty and M2 input

## Data Roadmap

### Priority 1: Real image-level metric GT

Minimum viable data:
- 20-50 photos with ArUco/printed board
- 1-3 defect types: crack, fastener spacing, coin/known object
- caliper readings or known physical dimensions

Why:
- This closes the biggest claim gap.

### Priority 2: krkCMd image subset

Use the table to choose:
- thin, median, wide, and failure profiles
- several held-out series

Then download/extract only corresponding images if the zip structure allows partial extraction.

Goal:
- connect scanner image ROI -> profile -> width
- evaluate image segmentation/profile extraction, not only table profile inference

### Priority 3: T-LESS CAD+pose

Use as industrial metric geometry benchmark.

Plan:
- derive visible object extents/diameters from CAD+pose/intrinsics
- evaluate measurement core with GT mask first (upper bound)
- then SAM3 segmentation with generic/object prompts (actual promptable test)

Risk:
- T-LESS objects are textureless and category names may be unnatural for SAM3 text prompts.

### Priority 4: Rebar/fastener counting

ROI-1555 is available but license is not clean enough for broad claims. Use it for internal evaluation.
Need either clean public alternative or self-captured fastener board for publishable demo.

## Immediate Next Experiments

### Experiment N1: SAHI rebar counting

Question: Is E-cnt-1 failure due to global image scale/crowding rather than concept failure?

Implementation:
- tile image into overlapping windows
- run SAM3 per tile
- stitch masks/boxes with IoU/center-distance dedup
- compare to existing zero-shot global counts

Result:
- Global SAM3 `metal rod`: MAE 13.2, acc@10% 0%.
- SAHI SAM3 threshold 0.35: MAE 7.35, acc@10% 20%, exact 5%.
- High-density scenes remain undercounted (e.g. GT 81→40).

Interpretation:
- Tiling helps materially, so E-cnt-1 was partly a scale/crowding failure.
- Tiling is not enough; next step should be density/centroid fallback or supervised instance head.

### Experiment N2: T-LESS GT-mask measurement upper bound

Question: If segmentation is perfect, can GaugeAnything derive stable mm dimensions from CAD/pose/masks?

Implementation:
- parse BOP scene camera/GT
- use GT visible masks if available or render CAD silhouette
- measure diameter/width in pixels and convert to mm via pose/CAD geometry

Expected outcomes:
- separates measurement-core validity from SAM3 segmentation difficulty.

### Experiment N3: M2 v2 calibration-only baseline

Question: Can source-conditioned scalar calibration beat neural refiner on width bias?

Implementation:
- train per-source or style-cluster linear/quantile calibration using existing mask-derived width data
- evaluate source-held-out

Expected outcomes:
- if a simple calibrator works, neural refiner should not be overclaimed.

## Claim Language To Use Going Forward

Safe:
- "promptable segmentation plus metrology core"
- "profile-level physical crack-width validation"
- "real-photo known-object measurement consistency"
- "zero-shot counting gap identified"
- "continuous representations recover signal for fuzzy/field regimes"

Avoid until more evidence:
- "field-ready crack-width measurement"
- "caliper-grade promptable mm output"
- "general industrial counting"
- "real alpha matting accuracy"
- "SAM3 understands industrial defects broadly"
