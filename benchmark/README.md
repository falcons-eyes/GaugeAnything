# GaugeBench v1.0 — Promptable Physical Measurement Benchmark

이미지 + 프롬프트에서 **물리량**(px·μm·mm)을 측정하는 모델/파이프라인을 위한 벤치마크.
분할 IoU가 아니라 **측정 오차가 1급 지표**다. 모든 수치는 감사된 프로토콜
([docs/RIGOR_AUDIT.md](../docs/RIGOR_AUDIT.md))로 산출되고,
[expected_v1.json](expected_v1.json)에 핀(pin)되며,
`python benchmark/collect_results.py`가 드리프트를 검증한다.

현재 리더보드: [LEADERBOARD.md](LEADERBOARD.md)

## 공통 규칙

1. **random split 금지** — source/dataset-family holdout이 기본.
2. config/모델 선택은 train/val에서만 — test label은 선택에 사용 불가.
3. 분할 지표는 crack-only IoU (2-class mIoU 금지 — [metric trap](../paper/RELATED_BASELINES.md)).
4. px-파생 GT와 물리 단위 GT를 섞어 headline으로 쓰지 않는다.
5. 평균과 함께 worst-source·bias·coverage를 보고한다.
6. gating을 쓰면 coverage(측정 거부율)를 정확도와 함께 보고한다.

## Tracks

### Track S — CrackSeg9k promptable crack segmentation

- 데이터: CrackSeg9k 9,159쌍 (14개 원본 소스, 파일 접두사로 식별). CC0 (DeepCrack/GAPs 서브셋 NC).
- 프로토콜: crack-only IoU, empty-GT 이미지 제외(별도 clean-rate), 3 seeds, n=150/seed.
- 재현: `python experiments/gauge_bench.py --n 150 --segmenters adaptive frangi sam3 --seeds 3`
- 공식 수치: SAM 3 zero-shot **0.442±0.011** (classical 최고 0.181의 2.44×), clean rate 0.68.

### Track W — crack width calibration ladder (px GT)

- 데이터: CrackSeg9k 파생 M2 cache. test는 source-held-out(cfd/cracktree200/deepcrack, n=219).
- GT: mask-derived width(px) — **물리 단위 아님을 명시**. 측정 보정 방법론 트랙.
- 재현:
  - `python experiments/m2v2_logit_threshold.py` (θ* ladder)
  - `python experiments/m2_specialist_tabular.py` (GaugeHead-Tiny)
  - `python experiments/m2_uncertainty_conformal.py` (90% conformal audit)
- 공식 ladder: raw 0.730 → neural 0.564 → quantile 0.480 → **GaugeHead-Tiny 0.472** → θ*+qcal **0.437**.
- uncertainty: log cross-conformal이 rel err 0.4724 유지 + 전 source coverage ≥0.90
  (adaptive 방법은 worst source에서 붕괴 — [상세](../docs/progress/2026-06-13_m2-v2b-uncertainty-conformal.md)).

### Track P — krkCMd physical crack width (μm GT)

- 데이터: krkCMd 19,098 profiles, 수동 측정 `MANwidth`(μm). **CC BY 4.0** — 분야 유일의
  공개 물리 폭 GT. group split (test 4,674).
- 재현:
  - profile-level: `python experiments/krkcmd_profile_eval.py`
  - signal CNN: `python experiments/krkcmd_signal_width.py`
  - e2e promptable(멀티-인스턴스): `python experiments/krkcmd_multiinstance_eval.py --stages 3`
- 공식 수치: 저자 DLM anchor 11.1μm · signal CNN(table) 18.6μm · GaugeProfile+cal 25.9μm ·
  e2e promptable recall 0.926 / MAE 29.8μm / median 15.7μm.

### Track D — T-LESS industrial part dimensions (CAD mm GT)

- 데이터: T-LESS (BOP, CC BY 4.0). CAD 정점 투영 최대 chord = 정확한 mm GT. visib≥0.95.
- 재현: `python experiments/tless_upper_bound.py` (perfect-mask 상한) →
  `python experiments/tless_sam3_eval.py` (SAM 3 교체).
- 공식 수치: perfect-mask 상한 median **2.83%** · SAM 3 최고 프롬프트 **2.51%** (match 100%).

### Track K — known-object scale consistency (실사진)

- 데이터: kaa euro coins (MIT), 22 scenes. 법정 직경 = known size, leave-one-out.
- 재현: `python experiments/coins_mm_eval.py --per-denom 4`
- 공식 수치: LOO mean **1.74%**, pass@5% **100%**.

## 향후 트랙 (v1.x 예정)

- **Track C — dense counting** (ROI-1555 rebar): 현재 음성 결과 상태(SAHI MAE 8.9) —
  density/centroid head가 entry 기준(MAE<5)을 만들면 정식 트랙 승격.
- **Track F — GaugeBench-Field**: ArUco+caliper 실측 mm GT 수집
  ([캡처 프로토콜](../docs/CAPTURE_PROTOCOL.md)) — 분야 첫 "사진+실측 mm" 결함 트랙.
- **Track Y — dynamic scenes**: TUM/ADT — promptable gate가 oracle gate를 대체하면 승격.

## Submissions

외부 제출 환영. 절차:

1. 해당 트랙의 재현 스크립트와 동일 split·동일 지표로 평가.
2. 결과 JSON + 실행 환경 + 방법 설명을 GitHub issue
   ([falcons-eyes/GaugeAnything](https://github.com/falcons-eyes/GaugeAnything/issues),
   `benchmark-submission` 라벨)로 제출.
3. 우리가 재현 확인 후 LEADERBOARD에 등재 (재현 불가 시 사유와 함께 보류 기록).

## Release gate

```bash
python benchmark/collect_results.py --check   # 21개 핀 지표 드리프트 검증
```

결과 JSON을 재생성하는 PR은 이 검사를 통과하거나, expected_v1.json 갱신 사유를
커밋 메시지에 명시해야 한다.
