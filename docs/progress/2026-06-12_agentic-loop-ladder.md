# 2026-06-12 (4부) — SpatialClaw 검토 → E-loop 사다리 → 멀티-인스턴스 결론

## SpatialClaw 정독 (NVIDIA, 2026)
persistent Python kernel(SAM3·DA3·scipy preload) + VLM이 스텝당 코드 셀 1개 → 중간 증거
(stdout/변수/show() 이미지)를 보고 수정 → ReturnAnswer. 20벤치 +11.2pt, 6개 백본 무수정.
본질: **action interface** — 고정 API가 못 하는 "중간 증거 기반 조합 수정"이 코드에서 자연 발생.

**우리 적용 판정**: 위치 커버리지 병목은 정확히 그 유형의 문제. 단, 계측기 제약 2가지 —
결정성(에이전트는 조합 탐색만, 수치는 결정적 재실행) + 감사 사다리(LLM 전에 결정적 루프).
차별 각도: 루프 종료조건=계측 게이트인 **agentic metrology**. 설계: `docs/AGENTIC_LOOP_DESIGN.md`.

## E-loop 사다리 (결정적, LLM 없음)

| 단계 | 커버리지 | gated MAE | 한 줄 진단 |
|---|---:|---:|---|
| E-loop-0 연속성+줌복구 | 52% | 39.9μm | 오인 경로도 매끈 — 기하 무력, 실패=의미 |
| E-loop-1 크랙다움 열별 선택 | 44% | 26.2μm | 선택은 oracle급 해결, recall 부족 |
| E-loop-1b +valley 후보 | 29% | 24.8μm | 잡 valley 점수 우위 — precision 붕괴 |
| E-loop-1c Viterbi 결합 | 49% | 27.3μm | 결정적 단일-경로의 수렴점 |

## 시각 진단 → 문제 재정의 (돌파)

후보 경로 vs 주석 좌표 렌더(`docs/assets/loop_diag_paths.png`): **장면에 진짜 크랙이 2개 이상**.
"실패"의 다수는 *다른 진짜 크랙*을 측정한 것 — 계측기로서 정상. 단일 경로 강요가
벤치마크 아티팩트였다.

**멀티-인스턴스 평가** (모든 SAM3 경로 인스턴스 각각 측정, 주석 크랙과 인스턴스 매칭):

| 지표 | 값 |
|---|---|
| 주석 크랙 recall | **93% (200/216)** |
| 매칭 인스턴스 폭 MAE / 중앙 | **30.5μm / 16.4μm** |

## 결론 (정직)
1. "위치 커버리지 46~66%"는 능력 한계가 아니라 **문제 정의 오류** — Inspection Atoms
   원설계(인스턴스별 측정)로 돌아가면 93%/30.5μm.
2. 특정 크랙 지정은 SAM3 **포인트 프롬프트**로 자연 해소 (제품 UX: 사용자가 찍는다).
3. **VLM 에이전트는 현 단계 불필요** — 결정적 파이프라인+올바른 문제 정의가 이김.
   N3 교훈 재현. 에이전트의 미래 가치는 잔여 7%·장면급 모호성·도구 선택 자동화로 한정.
4. SpatialClaw 검토의 진짜 수확: 루프 그 자체보다 "**중간 증거를 보고 가설을 수정하는
   진단 규율**" — 이번 사다리(시각 진단이 문제 재정의를 끌어냄)가 그 규율의 실행이었다.

## 산출물
- AGENTIC_LOOP_DESIGN.md · krkcmd_loop_recovery.py · krkcmd_loop_pathselect.py
- results: krkcmd_loop_recovery.json · krkcmd_loop_pathselect.json
- 갤러리: loop_diag_paths.png · RESULTS.md E-loop 섹션
