# 2026-06-12 — Owned Model Track: GaugeHead-Tiny

## 목적

foundation model을 새로 만드는 대신, GaugeAnything이 축적한 measurement 노하우를 작은 자체 모델로
학습할 수 있는지 확인했다. 핵심 질문:

> 기존 M2 neural refiner와 5-number quantile calibration을 같은 held-out source split에서
> 이기는 작은 measurement specialist를 만들 수 있는가?

## 추가 산출물

- `data/model_training_manifest.json`
- `docs/MODEL_RESEARCH_ROADMAP.md`
- `experiments/m2_specialist_tabular.py`
- `experiments/results/m2_specialist_tabular.json`
- `checkpoints/gaugehead_tiny_width.pkl`

Spark 산출물:

- repo: `/home/hwoo_joo/github/GaugeAnything`
- feature cache: `/home/hwoo_joo/github/GaugeAnything/datasets/m2_cache/tabular_features_v1.npz`
- checkpoint: `/home/hwoo_joo/github/GaugeAnything/checkpoints/gaugehead_tiny_width.pkl`
- result: `/home/hwoo_joo/github/GaugeAnything/experiments/results/m2_specialist_tabular.json`

## 프로토콜

- 데이터: 기존 `datasets/m2_cache/{train,val,test}.npz`
- 입력: SAM3 mask + grayscale image/mask statistics 19개
- target: mask-derived crack width, physical mm/um 아님
- split: 기존 M2 source-held-out 유지
- model selection: train-source validation에서 family 선택
- final: 선택된 family를 train+val로 refit 후 test 평가

주의: 이 실험은 physical GT가 아니라 mask-derived px width 실험이다. 논문에서는 "measurement-head
learning feasibility" 또는 "M2 v2-a calibration/model ladder"로 써야 한다.

## 결과

| method | test rel err | bias | note |
|---|---:|---:|---|
| raw SAM3 mask width | 0.7302 | +0.6800 | baseline |
| M2 neural refiner v1 | 0.5640 | +0.5033 | old 1.9M UNet |
| 5-bin quantile calibration | 0.4804 | +0.4106 | strongest simple baseline |
| HGB refit train+val | 0.4732 | +0.4395 | close |
| **GaugeHead-Tiny ExtraTrees refit train+val** | **0.4724** | +0.4557 | selected by val |

Per-source for selected model:

| source | rel err | bias |
|---|---:|---:|
| cfd | 0.5497 | +0.5432 |
| cracktree200 | 0.7201 | +0.7201 |
| deepcrack | 0.3480 | +0.3180 |

## 판정

작지만 의미 있는 성공:

- 기존 neural M2보다 확실히 좋다: `0.564 -> 0.4724`
- 5-bin quantile calibration도 처음으로 넘었다: `0.4804 -> 0.4724`
- checkpoint가 작고 재현성이 좋다: `gaugehead_tiny_width.pkl` 약 1.1MB

하지만 breakthrough로 과장하면 안 된다:

- 개선폭은 0.8%p 수준이다.
- bias는 여전히 +0.456으로 크다.
- CrackTree200 worst-source는 0.7201로 아직 매우 나쁘다.
- target은 physical GT가 아니라 mask-derived width다.

## 다음 실험

1. `M2 v2-b`: uncertainty/conformal interval head
   - 목표: rel err를 유지하면서 source별 90% coverage 보장
   - CrackTree200은 낮은 신뢰도 또는 넓은 interval로 표시되어야 함
2. `M2 v2-c`: SAM3 raw logits/soft masks를 feature에 추가
   - 2026-06-12 audit에서 raw `pred_masks` logits 접근 가능 확인됨
   - binary mask 통계만으로는 thin over-mask failure를 충분히 판별하지 못함
3. `Count v1`: ROI-1555 density/centroid head
   - SAHI가 MAE 8.9까지 줄였으나 dense undercount 지속
   - target: held-out MAE < 5
4. `Base v0`: frozen visual encoder + regime/scale tokens
   - Tiny 결과가 충분히 축적된 뒤 DINO/SAM/depth features로 확장

