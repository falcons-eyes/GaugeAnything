# 2026-06-13 — GaugeBench v1.0 패키징 + P2-1b SmartDoc detected quad

## 1. 무엇을 왜 했나

[연구 포지션 평가](../RESEARCH_POSITION_AND_ROADMAP.md)의 H1 실행 1·3번:

1. **GaugeBench v1.0** — 흩어진 audited 결과를 외부가 재현·제출할 수 있는 벤치마크로 패키징.
   전략 근거: 빅랩 후속 모델이 나와도 벤치마크·물리 GT 선점은 소멸하지 않는다.
2. **P2-1b** — SmartDoc 문서 스케일의 "GT quad 상한(0%)" 꼬리표를 실제 promptable
   성능으로 교체.

## 2. GaugeBench v1.0

위치: [`benchmark/`](../../benchmark/README.md)

- 5개 트랙: S(CrackSeg9k 분할) · W(폭 ladder, px) · P(krkCMd 물리 μm) ·
  D(T-LESS CAD mm) · K(known-object 일관성)
- `expected_v1.json`에 **21개 공식 지표 핀** + 허용오차
- `collect_results.py` — canonical JSON에서 지표 추출 → 핀 검증(release gate) →
  `LEADERBOARD.md` 렌더. 21/21 검증 통과 확인.
- 외부 제출 절차(issue 라벨 `benchmark-submission`) 명시
- v1.x 승격 대기 트랙: C(counting, MAE<5 기준) · F(Field 실측 mm) · Y(dynamic)

주의(핀 과정에서 발견한 드리프트): 논문 본문과 canonical JSON 사이 미세 차이 존재 —
multiinstance recall 0.926/29.8/15.7μm (논문 표기 93%/30.5/16.4),
signal CNN table MAE 18.6μm (논문 17.9). JSON 재생성 시점 차이로 보임.
**벤치마크는 JSON에 핀**했고, 논문 camera-ready에서 수치 동기화 필요 항목으로 기록.

## 3. P2-1b — SmartDoc detected quad (promptable 문서 스케일)

script: `experiments/smartdoc_detected_quad_eval.py`
result: `experiments/results/smartdoc_detected_quad_eval.json`

프로토콜: 150 프레임(배경 5 × 문서 30, 조합당 1장 stratified) → SAM3 프롬프트
3종 → mask→quad 게이트(4코너·볼록·면적·mask 정합) → detected-quad homography로
GT 코너를 metric 평면에 사상 → 변 길이 vs A4/모델 규격. 게이트 실패는 "측정 불가"로
별도 집계(추측 금지).

| prompt | gate pass | quad IoU | corner err px | edge rel err median | p90 |
|---|---:|---:|---:|---:|---:|
| **document** | **0.96** | 0.968 | 5.7 | **0.0154** | 0.317 |
| paper | 0.98 | 0.967 | 5.9 | 0.0161 | 0.268 |
| white paper sheet | 0.92 | 0.968 | 5.8 | 0.0158 | **0.140** |

앵커: naive global scale 10.3–17.5% (P2-1) · GT quad 상한 0% (정의상).

### 판정

1. **promptable 문서 스케일은 작동한다**: median 1.5% — naive 대비 ~10×, 게이트
   커버리지 92–98%. coverage matrix의 `document_card_scale` atom을
   partial → **official** 승격 (정직 caveat 포함).
2. **tail이 무겁다 (정직)**: p90이 14–32%. 게이트를 통과했지만 기하가 깨진 프레임
   (모션 블러·부분 가림 추정)이 남아 있다. median을 headline으로 쓰되 p90을 항상
   병기할 것. 프롬프트 간 coverage/tail 트레이드오프 존재('white paper sheet'가
   tail 최선·coverage 최저).
3. 다음: per-frame 품질 게이트(perspective/score 기반)로 p90 절단 → MIDV-500
   ID-card로 두 번째 known-size family.

## 4. 검증

```bash
# 로컬
python -m py_compile experiments/smartdoc_detected_quad_eval.py benchmark/collect_results.py
python benchmark/collect_results.py --check     # 21/21
python -m json.tool experiments/results/smartdoc_detected_quad_eval.json >/dev/null
# Spark 실행 로그: gate pass 0.96/0.98/0.92 재현
```
