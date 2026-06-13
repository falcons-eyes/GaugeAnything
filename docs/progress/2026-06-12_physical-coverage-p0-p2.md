# Physical AI Coverage P0-P2 — GaugeAnything as Measurement Layer

날짜: 2026-06-12

## 목적

ADT 한 트랙을 더 깊게 파는 대신, GaugeAnything의 이름값에 맞게 "여러 데이터셋, 여러 현장,
여러 물리량"을 덮는 coverage-first 로드맵을 구축한다.

핵심 프레이밍:

```text
image/video/sensor + prompt
  -> object/defect gate
  -> physical quantity
  -> uncertainty / coverage / failure reason
  -> inspection decision
```

## P0 — Physical Coverage Matrix

추가:

- `data/physical_coverage_matrix.json`
- `experiments/physical_coverage_report.py`
- `docs/PHYSICAL_COVERAGE_MATRIX.md`
- `docs/assets/physical_coverage_matrix.png`

현재 coverage atoms:

| status | count | 의미 |
|---|---:|---|
| official | 7 | headline으로 쓸 수 있는 감사/재현 결과 |
| partial | 4 | 작동하지만 oracle/gating/coverage 한계가 있음 |
| negative | 1 | 병목을 규정하는 음성 결과 |
| next | 2 | 바로 adapter sprint로 옮길 후보 |
| candidate | 1 | 유용하지만 검증이 더 필요 |

핵심 atoms:

- crack mask/width px — CrackSeg9k/VT LCW
- physical crack width μm — krkCMd
- hole diameter / fray alpha / uneven severity — Magnetic Tile
- coin known-object mm scale — kaa coins
- CAD part dimensions — T-LESS
- dense count — ROI-1555 rebar
- dynamic scale/object dimensions — TUM, ADT
- partial/new adapters — SmartDoc A4 scale
- next adapters — TimberSeg, DeepFish, HB/YCB/ITODD, KITTI signs

## P1 — Project Page Coverage Gallery

수정:

- `docs/index.html`

새 섹션:

- `Physical AI Coverage`
- 12-card gallery:
  - crack width
  - physical μm crack GT
  - sharp defect diameter
  - fuzzy fray alpha
  - boundaryless mura severity
  - coin mm scale
  - CAD part dimension
  - rebar count bottleneck
  - dynamic metric stability
  - ADT RGB-D dimensions
  - document/card next adapter
  - logs/fish next adapter

의도: Segment Anything류처럼 "딱 봐도 무엇을 하는지" 보이게 하되, GaugeAnything의 차별점인
물리량/단위/실패 병목을 카드별로 명시한다.

## P2 — Adapter Sprint Queue

추가:

- `experiments/physical_adapter_sprint.py`
- `experiments/results/physical_adapter_sprint.json`
- `experiments/smartdoc_scale_eval.py`
- `experiments/results/smartdoc_scale_eval.json`
- `data/scripts/download_metric.sh` SmartDoc release asset 다운로드 보강 (`frames.tar.gz`, optional `models.tar.gz`)

Spark에는 SmartDoc helper repo shell만 있었으나, 이번 P2에서 실제 `frames.tar.gz`를 다운로드했다.
adapter 실행 여부는 "repo shell 존재"가 아니라 실제 metadata/label 파일 존재를 뜻하는 `ready_evidence`를
기준으로 판정한다.

현재 `experiments/results/physical_adapter_sprint.json`:

- Spark ready: `["smartdoc_midv_known_quad_scale"]`
- Spark data root: `/home/hwoo_joo/github/GaugeAnything/datasets`
- Spark artifact: `/home/hwoo_joo/github/GaugeAnything/datasets/smartdoc/frames.tar.gz` (973M)
- SmartDoc result: `/home/hwoo_joo/github/GaugeAnything/experiments/results/smartdoc_scale_eval.json`

Sprint queue:

| 우선순위 | adapter | 데이터셋 | target |
|---:|---|---|---|
| 1 | `smartdoc_midv_known_quad_scale` | SmartDoc15-CH1 / MIDV-500 | document/card edge mm |
| 2 | `timberseg_log_count` | TimberSeg 1.0 | log count / diameter distribution |
| 3 | `deepfish_tray_length` | DeepFish / AutoFish | fish length |
| 4 | `bop_family_cad_dimensions` | HB / YCB-V / ITODD | CAD object dimensions |
| 5 | `kitti_round_sign_diameter` | KITTI signs | standard traffic-sign diameter |
| 6 | `arkitscenes_furniture_dimensions` | ARKitScenes 3DOD | furniture dimensions |

P2-1 SmartDoc adapter result:

- 입력: `datasets/smartdoc/frames.tar.gz` 내부 `metadata.csv.gz`
- n: 5,000 frames
- 목표: A4 GT quadrilateral 기반 marker-free scale stress test
- baseline: apparent width/height에서 얻은 naive global mm/px
- naive height relative error: median 10.3%, p90 14.9%
- naive width relative error: median 11.4%, p90 17.5%
- perspective ratio: median 1.19, p90 1.22
- upper bound: GT quad homography는 0-error by construction
- 다음 실행:

```bash
DATA_ROOT=./datasets bash data/scripts/download_metric.sh smartdoc
python experiments/smartdoc_scale_eval.py --data-root datasets/smartdoc
```

주의: 이 결과는 GT quadrilateral을 사용하므로 detector/SAM prompt 성능이 아니다. 논문/페이지에서는
"known-size document scale diagnostic" 또는 "plane geometry upper bound"로 표현해야 한다.

## 판정

이제 연구의 다음 메시지는 "ADT에서 한 결과"가 아니라:

> GaugeAnything is a promptable physical measurement interface over visual foundation models,
> evaluated across cracks, defects, parts, counts, known objects, dynamic scenes, and physical state.

다음 실제 실행은 P2-1b `SmartDoc/MIDV detected quad scale`이 가장 좋다. 이유:

1. 현장 설득력이 높다: marker 없이 문서/카드 규격으로 mm scale을 얻는 장면은 비전문가도 바로 이해한다.
2. PlaneScale/tilt 결과와 연결된다.
3. GT quad smoke test가 이미 끝나서 detector/prompt gate만 추가하면 된다.
4. ADT처럼 oracle 논쟁이 적다.
