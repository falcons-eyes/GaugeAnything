# GaugeAnything — 연구 포지션 냉정 평가와 다음 로드맵

날짜: 2026-06-13. 조사 시점 기준 외부 지형 재확인 포함.

> 궁극 목표(재확인): 현장의 physical vision 문제(폭·치수·개수·간격·등급·불확실성)를
> vertical하게 풀어주는 **generalized "anything" 측정 모델**을 오픈소스 + 연구 논문 경로로
> 만든다. 이 문서는 그 목표 대비 현재 위치를 의도적으로 박하게 평가한다.

## 1. 문제 정의 검토 — 유효한가?

**Promptable Quantitative Inspection** (image + prompt → Inspection Atoms
{mask, class, count, mm±σ, grade, confidence}) 정의 자체는 2026-06 시점에도 유효하고,
여전히 선점 상태다:

- Measure Anything(arXiv 2412.03472)은 여전히 와인병 1개 "±10%"가 정량 평가의 전부 —
  promptable 계측의 정량 비교표는 우리가 채우기 전까지 비어 있다.
- SAM 3 기반 계측의 peer-comparable 수치는 우리 보고가 처음이다(분할 IoU 포함).
- VLM 쪽에서도 MeasureBench(arXiv 2510.26865)가 "VLM은 계기판도 못 읽는다"를 보였다 —
  측정은 인식과 다른 능력이라는 우리 명제를 외부에서 보강.
- PAI-Bench(CVPR 2026)·Cosmos 3 등 "Physical AI" 벤치마크는 world-model/video 레이어에
  집중 — **측정(metrology) 레이어는 여전히 공백**이다. 포지셔닝 충돌 없음.

단, 약점이 하나 있다. 현재 정의는 **시스템(파이프라인)의 정의이지 모델의 정의가 아니다**.
"anything 모델"을 주장하려면 문제 정의를 모델 입출력 계약으로 다시 써야 한다:

```text
(image | video | profile | sensor) + prompt + scale evidence
  -> 단일 모델 -> {physical quantity, unit, interval, coverage, failure reason}
```

이 계약의 학습 가능 형태가 GaugeSpecialist-Base(§5 H2)다.

## 2. 데이터셋 검토 — 문제 정의의 시작점

냉정한 사실: **physical GT를 가진 우리 자산은 3개 family뿐이다.**

| 등급 | 데이터 | physical GT 메커니즘 | 한계 |
|---|---|---|---|
| 진짜 물리 GT | krkCMd (19,098 profiles) | 수동 측정 μm | 6400dpi 스캐너·콘크리트 단일 도메인 |
| 진짜 물리 GT | T-LESS/BOP | CAD mm + pose | 강체 부품, 실험실 촬영 |
| 유사 물리 GT | coins / SmartDoc / MIDV | 규격 치수(known size) | scale 검증용, 결함 측정 아님 |
| px-파생 GT | CrackSeg9k M2 cache | mask 기하 폭(px) | 물리 단위 아님 — headline 금지 유지 |
| 진단/상한 | ADT(oracle gate), SmartDoc(GT quad) | GT pose/quad 의존 | promptable 성능 아님 |

분야 전체로 봐도 "결함 사진 + 물리 측정값" 페어 공개 데이터셋은 krkCMd가 거의 유일하다.
이것이 의미하는 바는 두 가지다:

1. **모델보다 데이터가 병목이다.** GaugeSpecialist를 키워도 검증할 물리 GT가 없으면
   주장할 수 없다.
2. **데이터셋 공백 자체가 우리의 모트 기회다.** ArUco+caliper 캡처 프로토콜(이미 공개)은
   사업 자산이 아니라 **벤치마크 데이터셋 구축 도구**로 격상해야 한다. 우리가
   "GaugeBench-Field" (실측 mm GT 수백 장)를 만들어 공개하면, 이후 모든 후속 연구가
   우리 벤치마크 위에서 평가받는다 — CrackSeg9k가 그랬듯이.

## 3. 선행연구·baseline 지형 (2026-06 재확인)

기존 `paper/RELATED_BASELINES.md`의 비교표는 여전히 유효. 신규 인접 연구:

| 신규 (2026) | 내용 | 우리에의 함의 |
|---|---|---|
| ConceptSeg-R1 (arXiv 2605.20385) | meta-RL 기반 concept segmentation | SAM 3 대안/추가 backbone 후보 — prompt 취약성 비교 대상 |
| CAD-Prompted SAM3 (arXiv 2602.20551) | CAD 조건부 산업 객체 분할 | T-LESS 트랙의 직접 인접 — E-mm-2 후속에서 비교/인용 필요 |
| MeasureBench (arXiv 2510.26865) | VLM 계기 읽기 벤치마크 | related work 인용 — "측정은 별개 능력" 외부 근거 |
| PAI-Bench (CVPR 2026) | Physical AI 종합 벤치(world model 중심) | 측정 레이어 공백 입증 — 포지셔닝 인용 |

baseline 체계 판정: **비교 대상 선정은 건전하다.** 단 두 가지 보강 필요 —
(1) CrackMamba를 우리 split으로 재실행(코드 공개됨, camera-ready 체크리스트 기존 항목),
(2) 카운팅에서 CountGD/GeCo zero-shot을 ROI-1555에 직접 돌려 우리 표에 외부 모델 셀을
채울 것(현재 rebar 표는 우리 결과 + 문헌 수치 혼재).

벤치마크 판정: 우리의 Gauge-Bench는 아직 **자체 평가 도구이지 커뮤니티 벤치마크가 아니다.**
외부 채택이 없으면 "첫 보고" 우위는 시간이 지나면 소멸한다. 패키징(재현 스크립트 + 고정
split + leaderboard 표)이 H1의 핵심 과제다.

## 4. 현황 냉정 평가

### 자산 (paper-grade)

1. 정직성 인프라 — rigor audit, val/test 분리, 음성 결과 문서화. 리뷰 방어력이 높다.
2. 중심 발견 "mask=WHERE, signal=HOW WIDE" — 물리 GT로 검증된 명제 (23.2μm median).
3. calibration ladder + conformal audit — 단순 baseline이 학습 모델을 이기는 구간과
   학습 모델이 처음 이기는 구간(GaugeHead-Tiny 0.472)을 모두 정직하게 보유.
4. 동적 증거 — TUM 1.06% gated, ADT oracle 8.7% + ROI-only 316% 음성 대조.
5. 선점 — promptable 물리 계측의 정량표 첫 entry들.

### 부채 (모델 주장 차단 요인)

1. **generalized 모델이 없다.** 현재 실체는 frozen SAM 3 + 고전 기하 + 5개 독립 tiny
   head다. "GaugeAnything"이라는 이름이 약속하는 단일 promptable 측정 모델은 아직 0줄이다.
2. **headline 다수가 상한(oracle/GT-gated)이다.** ADT는 GT pose/volume gate, SmartDoc은
   GT quad, krkCMd e2e는 localization gate 통과분(46-66%)이다. 전부 문서에 명시했지만,
   "상한"을 "성능"으로 바꾸는 작업이 그대로 남아 있다.
3. **coverage 폭이 아직 좁다.** coverage matrix 15 atoms 중 official 7, 나머지는
   partial/진단/계획. "anything"을 말하기엔 physical family가 크랙·부품·동전에 편중.
4. **counting 미해결** — SAHI 후에도 MAE 8.9, dense touching 구조적 undercount.
5. **uncertainty는 calibrated이지만 넓다**(±70%) — 실용 계기로는 미달, concept shift
   플래깅은 미해결(정직 음성으로 보유).
6. **외부 검증 0** — 인용·외부 재현·리더보드 제출 모두 아직 없음. arXiv 엔도스먼트도
   미해결 상태.
7. 단일 저자·단일 GPU — 실행 속도가 경쟁 창 대비 위험 요인.

### 종합 판정

**연구 자산은 시스템/발견 논문 1편 기준 제출 가능 수준이다. 그러나 "generalized anything
모델" 기준으로는 pre-model 단계다.** 창은 열려 있으나(빅랩이 SAM 후속에 측정을 통합하면
닫힌다), 우리의 방어선은 모델이 아니라 (a) 물리 GT 벤치마크 선점, (b) 측정 문법(regime
router + scale resolver + uncertainty 계약)의 표준화다. 이 둘을 먼저 굳히고 모델을 그
위에 세우는 순서가 맞다.

## 5. 다음 로드맵

### H1 — 지금부터 4주: "벤치마크와 논문을 굳힌다"

| # | 과제 | 성공 기준 |
|---|---|---|
| 1 | paper v2 arXiv 제출 — 엔도스먼트 해결(공저자/추천인 확보 포함 옵션 검토) | arXiv ID 발급 |
| 2 | **GaugeBench v1.0 패키징** — 고정 split·재현 스크립트·결과표·제출 가이드를 단일 디렉토리로; krkCMd(μm)·T-LESS(mm)·coins(known-size)·CrackSeg9k(px+IoU) 4 트랙 | 외부인이 README만으로 전 수치 재현 |
| 3 | P2-1b: SmartDoc detected quad (SAM 3 프롬프트) — GT quad 상한을 promptable 성능으로 | detected-quad 오차 + gate 실패율 보고 |
| 4 | ADT promptable gate — oracle 대체, gate 실패율 표 | "상한" 꼬리표 제거 또는 정직한 실패 기록 |
| 5 | Count v1: ROI-1555 density/centroid head | held-out MAE < 5, dense-bin undercount 50%↓ |

### H2 — 1~3개월: "데이터 엔진과 첫 통합 모델"

| # | 과제 | 성공 기준 |
|---|---|---|
| 6 | **GaugeBench-Field 수집** — ArUco+caliper 프로토콜로 실측 mm GT 캡처(크랙·부품·파스너 ≥300장), CC BY 공개 | 분야 첫 "사진+실측 mm" 결함 데이터셋 |
| 7 | adapter sprint 소화 — MIDV 카드 scale → TimberSeg 통나무 → DeepFish 길이 → BOP family 확장 | coverage official ≥ 12 atoms |
| 8 | **GaugeSpecialist-Base v0** (25-90M) — frozen DINOv2/SAM3 feature + scale/regime token + multi-task head(width/diameter/count/scale) + conformal head | 최소 3개 physical family에서 per-task 파이프라인 동등 이상, worst-domain 악화 금지 (roadmap 승격 규칙 유지) |
| 9 | M2 v2-c: SAM3 raw logit/soft mask feature — concept shift 신호 탐색 | CrackTree200 플래깅 or 정직 음성 2차 기록 |

### H3 — 3~6개월: "모델 주장과 두 번째 논문"

| # | 과제 | 성공 기준 |
|---|---|---|
| 10 | GaugeSpecialist-Mid — LoRA/adapter + 통합 uncertainty, GaugeBench-Field에서 평가 | 단일 모델이 GaugeBench 전 트랙 entry |
| 11 | 논문 2(모델 페이퍼): "단일 promptable 측정 모델 + 물리 GT 벤치마크" | 1편은 시스템/발견, 2편은 모델/벤치 — 분리 유지 |
| 12 | 커뮤니티 운영 — leaderboard, 외부 제출 수용, 인접 연구(CAD-Prompted SAM3 등) 비교 초대 | 외부 entry ≥ 1 |

### 순서의 논리

데이터(벤치마크) → 모델 순서다. 역순이면: 모델을 먼저 키워도 물리 GT가 없어 주장이 안 되고,
빅랩 후속 모델이 나오는 순간 모델 우위는 소멸하지만 **벤치마크와 실측 데이터셋 선점은
소멸하지 않는다.** "generalized anything 모델"이라는 최종 목표는 H2의 Base가 3 family를
통과하는 순간부터 주장 가능 범위에 들어온다.

## 6. 리스크 (정직하게)

1. **경쟁 창**: SAM 3 출시 후 6-12개월 내 측정 통합 후속이 나올 확률 높음 → H1 벤치마크
   선점이 유일한 헤지.
2. **데이터 수집 실행력**: GaugeBench-Field는 손과 캘리퍼가 필요한 물리 노동 — 단일
   저자의 최대 병목. 수집 파트너(시공사/검사업체) 확보를 H2 초에 검토.
3. **엔도스먼트 지연**: arXiv가 늦어지면 "첫 보고" 타임스탬프가 위험 — 프로젝트 페이지
   PDF + GitHub 타임스탬프 + HF가 임시 증거이나 약함. OpenReview/워크숍 제출 병행 검토.
4. **벤치마크 무관심**: 공개해도 외부 채택이 없을 수 있음 — 인접 저자들(krkCMd,
   OmniCrack30k, Measure Anything)에게 직접 평가 초대가 필요.
