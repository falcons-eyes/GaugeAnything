# 2026-06-13 — Codex to Claude Code Handoff

이 파일은 Claude Code가 바로 재개할 수 있도록 만든 최신 인계 인덱스다. 상세 handoff는 repo 바깥
Claude 설정 폴더에도 저장했다:

```text
/Users/jamesjoo/work/falconoon.com/.claude/GaugeAnything_HANDOFF_2026-06-13.md
```

## 환경

- local repo: `/Users/jamesjoo/work/falconoon.com/GaugeAnything`
- Spark repo: `/home/hwoo_joo/github/GaugeAnything`
- Spark 접속: `ssh hwoo_joo-Edgexpert-Spark`
- Spark Python: `.venv/bin/python`
- project page: `http://localhost:8848/index.html`

## Codex가 최근 완료한 핵심 작업

### 1. Physical AI coverage P0-P2

목표: ADT 한 트랙이 아니라 GaugeAnything의 이름값에 맞게 여러 데이터셋/물리량 coverage를 보여주는
구조로 전환.

추가/수정:

- `data/physical_coverage_matrix.json`
- `experiments/physical_coverage_report.py`
- `docs/PHYSICAL_COVERAGE_MATRIX.md`
- `docs/assets/physical_coverage_matrix.png`
- `docs/index.html` — `Physical AI Coverage` 12-card gallery
- `experiments/physical_adapter_sprint.py`
- `experiments/results/physical_adapter_sprint.json`
- `docs/progress/2026-06-12_physical-coverage-p0-p2.md`

현재 coverage counts:

| official | partial | negative | next | candidate |
|---:|---:|---:|---:|---:|
| 7 | 4 | 1 | 2 | 1 |

### 2. SmartDoc known-document scale result

Spark에서 실제 데이터 다운로드와 smoke eval 완료:

- Spark data: `/home/hwoo_joo/github/GaugeAnything/datasets/smartdoc/frames.tar.gz` (973M)
- result: `experiments/results/smartdoc_scale_eval.json`
- script: `experiments/smartdoc_scale_eval.py`
- downloader: `data/scripts/download_metric.sh smartdoc`

결과:

| metric | value |
|---|---:|
| n frames | 5,000 |
| naive height rel err median / p90 | 10.3% / 14.9% |
| naive width rel err median / p90 | 11.4% / 17.5% |
| GT quad homography upper bound | 0% by construction |

주의: GT quadrilateral 사용 결과다. detector/SAM prompt 결과로 과장 금지.

### 3. Owned model track — GaugeHead-Tiny

사용자 요청: foundation model까지는 아니더라도 자체 fine-tuned/specialist 모델을 만들 수 있는지 검토.
Codex는 첫 tiny measurement head를 실제로 학습/평가했다.

추가:

- `data/model_training_manifest.json`
- `docs/MODEL_RESEARCH_ROADMAP.md`
- `experiments/m2_specialist_tabular.py`
- `experiments/results/m2_specialist_tabular.json`
- `checkpoints/gaugehead_tiny_width.pkl`
- `docs/progress/2026-06-12_owned-model-gaugehead-tiny.md`

Spark 산출물:

- feature cache: `/home/hwoo_joo/github/GaugeAnything/datasets/m2_cache/tabular_features_v1.npz`
- checkpoint: `/home/hwoo_joo/github/GaugeAnything/checkpoints/gaugehead_tiny_width.pkl`
- result: `/home/hwoo_joo/github/GaugeAnything/experiments/results/m2_specialist_tabular.json`

결과:

| method | test rel err | bias |
|---|---:|---:|
| raw SAM3 mask width | 0.7302 | +0.6800 |
| old neural M2 refiner | 0.5640 | +0.5033 |
| 5-bin quantile calibration | 0.4804 | +0.4106 |
| **GaugeHead-Tiny ExtraTrees refit train+val** | **0.4724** | +0.4557 |

한계:

- mask-derived px width 실험이지 physical μm/mm GT가 아니다.
- quantile 대비 개선폭은 작다.
- CrackTree200 worst-source rel err가 0.7201로 여전히 나쁘다.

다음 모델 작업 추천:

- `M2 v2-b`: GaugeHead-Tiny에 uncertainty/conformal interval 추가
- `M2 v2-c`: SAM3 raw logits/soft masks feature 추가
- `Count v1`: ROI-1555 density/centroid head

## 검증

로컬:

```bash
python -m py_compile experiments/physical_coverage_report.py experiments/physical_adapter_sprint.py experiments/smartdoc_scale_eval.py experiments/m2_specialist_tabular.py
python -m json.tool data/physical_coverage_matrix.json >/dev/null
python -m json.tool data/model_training_manifest.json >/dev/null
python -m json.tool experiments/results/smartdoc_scale_eval.json >/dev/null
python -m json.tool experiments/results/m2_specialist_tabular.json >/dev/null
git diff --check
```

Spark:

```bash
cd /home/hwoo_joo/github/GaugeAnything
.venv/bin/python experiments/physical_adapter_sprint.py
.venv/bin/python experiments/smartdoc_scale_eval.py --data-root datasets/smartdoc --max-frames 5000
.venv/bin/python -u experiments/m2_specialist_tabular.py
```

Spark 재실행에서:

- SmartDoc 수치 재현
- adapter ready: `["smartdoc_midv_known_quad_scale"]`
- GaugeHead-Tiny rel err: `0.4724`

## Claude에게 권장하는 다음 순서

1. `docs/MODEL_RESEARCH_ROADMAP.md`와 `docs/progress/2026-06-12_owned-model-gaugehead-tiny.md`를 먼저 읽기.
2. `M2 v2-b` uncertainty/conformal interval 구현.
3. 또는 `P2-1b` SmartDoc detected/SAM document quad 구현.
4. Rebar는 SAHI로도 MAE 8.9라, 다음은 density/centroid head가 맞음.
5. ADT는 oracle gate 상태이므로 segmentation/promptable gate 대체 전까지 과장 금지.

