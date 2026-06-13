# GaugeAnything Owned Model Roadmap

날짜: 2026-06-12

목표: GaugeAnything을 단순한 `SAM3 + geometry` 파이프라인에서, 여러 물리량을 직접 학습하는
작은 자체 모델군으로 발전시킨다. 단, 지금 단계에서 "새 foundation model"을 주장하지 않는다.
우리의 강점은 거대한 perception pretraining이 아니라 `WHERE -> HOW MUCH -> UNCERTAINTY`를 잇는
measurement grammar다.

## 결론

자체 모델은 가능하다. 다만 첫 성공은 거창한 backbone fine-tuning이 아니라 작은 measurement head에서
나왔다.

| crack width held-out test | rel err | bias | note |
|---|---:|---:|---|
| raw SAM3 mask width | 0.7302 | +0.6800 | mask geometry only |
| M2 neural refiner v1 | 0.5640 | +0.5033 | 1.9M UNet, superseded |
| 5-bin quantile calibration | 0.4804 | +0.4106 | strongest simple baseline |
| **GaugeHead-Tiny width specialist** | **0.4724** | +0.4557 | ExtraTrees over 19 image/mask features, train+val refit |

판정:

- `GaugeHead-Tiny`는 처음으로 quantile calibration bar를 넘었다.
- 개선 폭은 작다. headline이 아니라 "owned model path opened"로 써야 한다.
- worst-source는 아직 크다: CrackTree200 `0.7201` rel err. 다음 모델은 평균 개선보다
  domain worst-case와 uncertainty를 겨냥해야 한다.

결과물:

- script: `experiments/m2_specialist_tabular.py`
- result: `experiments/results/m2_specialist_tabular.json`
- checkpoint: `checkpoints/gaugehead_tiny_width.pkl`
- Spark feature cache: `/home/hwoo_joo/github/GaugeAnything/datasets/m2_cache/tabular_features_v1.npz`

## 왜 Measurement Specialist인가

2026년 현재 외부 foundation model 축은 이미 강하다.

- SAM 2 / SAM 계열: promptable image/video segmentation backbone.
  - Meta SAM 2 page: https://ai.meta.com/research/sam2/
  - arXiv: https://arxiv.org/abs/2408.00714
- DINOv2: frozen visual features for many downstream tasks.
  - Meta demo/research: https://dinov2.metademolab.com/
  - arXiv: https://arxiv.org/abs/2304.07193
- Depth Anything V2: monocular/metric depth family and metric-depth fine-tuning code.
  - project: https://depth-anything-v2.github.io/
  - GitHub metric depth: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
- CountGD / open-world counting: open-vocabulary counting reference line.
  - project: https://www.robots.ox.ac.uk/~vgg/research/countgd/
  - arXiv: https://arxiv.org/abs/2407.04619

따라서 정면 승부 지점은 "더 큰 segmentation foundation model"이 아니다. GaugeAnything의 빈칸은:

```text
prompt / mask / depth / known scale / sensor
  -> measurement regime
  -> physical quantity head
  -> calibrated uncertainty and coverage
```

즉, 작은 모델이라도 물리량을 직접 예측하고 실패 조건을 말할 수 있으면 연구적으로 가치가 있다.

## Model Families

| family | rough size | role | first training targets | promotion rule |
|---|---:|---|---|---|
| `GaugeHead-Tiny` | 1e3-2M | frozen masks/profiles/statistics 위 task head | crack width, profile CNN, fray alpha, uneven severity, density count | strongest simple baseline을 source-held-out에서 이겨야 함 |
| `GaugeSpecialist-Base` | 25M-90M | frozen DINO/SAM/depth feature + prompt/regime/scale tokens | document/card scale, BOP dimensions, dynamic object dimensions | 최소 3개 physical family에서 개선, worst-domain 악화 금지 |
| `GaugeSpecialist-Mid` | 100M-300M | encoder LoRA/adapters + multi-task heads | thin/fuzzy/field/count/scale/dynamic 통합 | dataset-family holdout에서 scaling law 확인 |
| `GaugeSpecialist-Large` | 300M+ | prompt-conditioned physical measurement model | unified mask+scale+quantity+uncertainty | Base/Mid가 먼저 성공하지 않으면 보류 |

## Benchmark Rules

1. Random split 금지. source/dataset-family holdout을 기본으로 한다.
2. 모든 모델은 `raw`, `classical/calibration`, `previous neural`, `new model`을 같은 표에 둔다.
3. simple baseline에 지면 negative/reproducibility로 보관한다.
4. physical unit GT와 pixel-derived GT를 절대 섞어 headline으로 쓰지 않는다.
5. 평균뿐 아니라 coverage, bias, worst-source, uncertainty calibration을 함께 보고한다.

## Training Atoms

| atom | readiness | current result | model opportunity |
|---|---|---|---|
| CrackSeg9k M2 cache | ready on Spark | GaugeHead-Tiny `0.4724` rel err vs quantile `0.4804` | uncertainty + worst-source correction |
| krkCMd profile width | ready on Spark, clean license | profile CNN table MAE ~18um; localization-gated e2e 39.9um / 23.2um median | scale-normalized profile model + uncertainty |
| Magnetic Tile fray/uneven | checkpoints ready | fray IoU 0.949; uneven learned AUC 0.636 vs classical 0.669 | regime router; don't overclaim field head |
| ROI-1555 rebar | ready internal | SAHI SAM3 MAE 8.9, rel err 44.9% | density/centroid head, target MAE < 5 |
| SmartDoc A4 scale | ready on Spark | naive scale 10-17% error; GT quad upper-bound 0 | document quad detector/SAM prompt |
| T-LESS CAD dimensions | ready on Spark | SAM3 mask median 2.5%; perfect-mask ceiling 2.83% | BOP family expansion + geometry mode classifier |
| TUM/ADT dynamic | partial | TUM gated 1.06%; ADT oracle 8.7%; ROI-only 316% | replace oracle gate with prompt/seg gate |

## GaugeHead-Tiny Experiment

Protocol:

- Data: `datasets/m2_cache/{train,val,test}.npz`
- Test sources are the original M2 held-out sources: `cfd`, `cracktree200`, `deepcrack`
- Features: 19 grayscale/mask geometry statistics
- Target: mask-derived crack width, not physical mm
- Selection: model family selected on train-source validation only
- Final checkpoint: selected family refit on train+val

Selected model:

- validation winner: `extra_trees_logwidth`
- checkpoint model: `extra_trees_logwidth_refit_trainval`
- checkpoint: `checkpoints/gaugehead_tiny_width.pkl`

Per-source test:

| source | rel err | bias |
|---|---:|---:|
| cfd | 0.5497 | +0.5432 |
| cracktree200 | 0.7201 | +0.7201 |
| deepcrack | 0.3480 | +0.3180 |

Interpretation:

This result is useful, but it is not enough. It proves that learned measurement heads can beat
hand calibration, but the persistent positive bias means the model is still mostly correcting a
thin-crack over-mask failure rather than understanding physical width. The next step is an uncertainty
head and a source/style-conditioned model that can abstain or widen intervals on CrackTree-like domains.

## Next Experiments

### M2 v2-b — Uncertainty-Aware Width Head — DONE 2026-06-13

Add interval prediction around `GaugeHead-Tiny`:

- conformal residual intervals by validation residuals
- per-source calibration audit
- report `coverage@90%`, interval width, and abstention rate

Success:

- keep rel err <= 0.472
- 90% interval coverage on every held-out source
- CrackTree200 should be flagged high-uncertainty instead of silently overconfident

Result (`experiments/results/m2_uncertainty_conformal.json`,
`docs/progress/2026-06-13_m2-v2b-uncertainty-conformal.md`):

| criterion | outcome |
|---|---|
| rel err <= 0.472 | met — `conformal_log_cv_trainval` keeps 0.4724 |
| 90% coverage per held-out source | met — cfd 0.91 / cracktree200 1.00 / deepcrack 0.95 |
| CrackTree200 high-uncertainty flag | **failed (honest negative)** — learned σ, ensemble std, kNN distance all miss it |

핵심 발견: val에서 가장 효율적인 adaptive interval(normalized conformal, CQR)은 CrackTree200에서
coverage 0.21/0.11로 붕괴. CrackTree200 실패는 covariate shift가 아니라 width-label 관계가 다른
**concept shift**라 feature 기반 OOD 신호로는 원리적으로 잡히지 않는다. 배포 checkpoint는
non-adaptive `conformal_log_cv_trainval` (`checkpoints/gaugehead_tiny_width_conformal.pkl`).
interval은 calibrated이지만 tight하지 않음(median relative width 1.394 ≈ ±70%).

### Count v1 — Rebar Density/Centroid Head

The rebar failure is no longer a prompt problem. Train a small density or centroid heatmap head from
ROI-1555 masks.

Success:

- held-out MAE < 5
- dense-bin undercount bias cut by at least 50%

### Base v0 — Frozen Encoder Measurement Tokens

Use a frozen visual encoder only after Tiny establishes targets:

```text
image crop + optional mask/logit + scale token + regime token
  -> shared frozen features
  -> task heads: width / alpha / count / scale / dimension / uncertainty
```

Start with DINOv2 or a locally usable encoder; keep SAM/depth as teachers or inputs, not as a claim.

