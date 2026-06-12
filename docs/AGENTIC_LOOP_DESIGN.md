# GaugeLoop — Agentic Metrology 설계 (SpatialClaw 검토 기반)

날짜: 2026-06-12 · 참조: SpatialClaw (NVIDIA, 2026 — code as the action interface, persistent
kernel + 5-stage loop, 20벤치 +11.2pt, 6개 VLM 백본 무수정).

## 1. 판정: 왜 우리에게 맞는가

SpatialClaw의 본질은 "중간 증거를 보고 조합을 수정할 수 있는 action interface". 우리의
잔여 병목(위치 커버리지 46~66%)은 정확히 그 종류의 문제다 — 실제로 폭 병목 해체 과정
자체가 사람이 돌린 luop였다: 전역 추론 → 1,090px 오프 진단 → 타일화 → snap → 조건부 분해.

**차별 각도 (후속 연구 포지션)**: SpatialClaw=QA 정답률, 우리=**계측 게이트 통과율** —
"스스로 신뢰할 때까지 재측정하는 계측기 (agentic metrology)". 루프의 종료 조건이
ReturnAnswer 검증이 아니라 **계측 신뢰 게이트**라는 점이 새롭다.

## 2. 원칙 (계측기로서의 제약 — SpatialClaw와 다른 지점)

1. **결정성 우선**: 에이전트는 결정적 도구의 *조합과 순서*만 탐색. 최종 수치는 채택된
   조합의 결정적 재실행으로 확정. 같은 입력 → 같은 출력.
2. **게이트가 곧 보상**: 프로파일 상관/명암/연속성 게이트 통과율이 루프의 목적 함수.
   통과 실패 지점은 '측정 불가'로 정직 보고 (커버리지 명시).
3. **에스컬레이션 설계**: 1차 결정적 파이프라인이 게이트를 통과하면 루프 미발동
   (T-LESS 2.5%, 동전 1.74%는 루프 불필요). 실패 구간에서만 단계 상승:
   결정적 복구 루프 → (그래도 실패 시) VLM 에이전트.
4. **감사 사다리**: LLM 에이전트의 기여는 결정적 루프 베이스라인 대비로만 주장
   (N3 교훈 — 단순한 것이 이기면 그렇게 보고).

## 3. GaugeKernel (persistent kernel 프리미티브 — 기존 자산 매핑)

| SpatialClaw | GaugeKernel (우리) |
|---|---|
| InputImages/Metadata | 이미지 + 스케일 참조(마커/기지물체) |
| tools.SAM3 | segment_sam3 / ensemble (이미 보유) |
| tools.Reconstruct (DA3) | PlaneScale homography / pose-depth (보유) |
| numpy/scipy 조합 | measure_thin/blob, skeleton, profile 추출 (보유) |
| show() 시각 피드백 | mask/프로파일/게이트 맵 렌더 (보유 갤러리 코드) |
| ReturnAnswer 검증 | **계측 게이트**: 상관·명암·연속성·σ (보유) |
| vlm.ask | (2단계) 프롬프트 후보 생성/장면 판단 |

## 4. 실험 사다리

| 단계 | 내용 | 성공 기준 |
|---|---|---|
| **E-loop-0** (결정적) | 실패 구간 active reacquisition: 인접 신뢰 열 행 보간 prior → 풀해상도 줌 크롭 SAM3 재분할 → 재스냅 → 게이트. 2회 반복(크롭 512→768) | 커버리지 46~66% → **80%+**, gated MAE ≤50μm 유지 |
| E-loop-1 (결정적+) | 능선 추적(다익스트라 on darkness) 연결 폴백 + 프롬프트 앙상블 재시도 | 커버리지 90%+ |
| E-loop-2 (VLM 에이전트) | GaugeKernel + 코드 셀 루프 (로컬 VLM 또는 API) — E-loop-1이 못 푸는 잔여(장면 모호성, 도구 선택)에서의 추가 기여 측정 | E-loop-1 대비 유의 개선 시에만 채택 |

## 5. 리스크 (정직)
- 루프가 오인 구조를 "자신 있게" 추적할 위험 → 게이트에 연속성(인접 행 jump 한계) 포함.
- VLM 단계 비용/지연 → 에스컬레이션로만. 제품 기본 경로는 결정적.
- E-loop-0/1으로 커버리지가 이미 90%+면 E-loop-2는 연구 가치(범용성 주장)로만.
