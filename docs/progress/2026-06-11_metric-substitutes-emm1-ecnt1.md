# 2026-06-11 — mm GT 대체 데이터 확보 + E-mm-1·E-cnt-1 첫 실측

## 배경
현장 촬영 불가(장비/환경 부재 — 이것이 연구 본질이기도) → 공개 데이터로 mm GT를
대체/유도하는 전수조사([data/REAL_MM_SOURCES.md](../../data/REAL_MM_SOURCES.md)) 후
Tier S 확보 완료: coins 455M · ROI-1555 rebar 579M · **T-LESS 1.8G** · SmartDoc · krkCMd 목록.

## E-mm-1 — 동전 cross-coin mm 검증 ✅ ⭐ (실사진 계측 첫 수치)

**설계**: kaa src가 README와 달리 A4 아닌 테이블 위(정직 기록) → 동일 권종 scene에서
**leave-one-out known-object**: 동전 i를 나머지 평균 px직경+법정 직경으로 환산 → 법정 직경과 비교.
검증 범위: 분할(SAM3 "coin")→등가직경→known-object 리졸버 **체인의 일관성** (마커 절대 체인은 합성 0.38% 별도).

| 지표 | 값 |
|---|---|
| 평가 | 22장, 이미지당 동전 8~60개 (6개 권종) |
| **LOO 상대오차** | **평균 1.74% / 중앙값 1.68%** |
| ±5% / ±10% 합격 | **100% / 100%** |
| px-CV (동일 물리 크기 일관성) | 평균 ~1.9% |

부수 발견: SAM3가 이미지당 **45~60개 동전을 안정 검출** (동일 scene 반복촬영에서 59/59/60/60 —
분리된 객체의 카운팅 신뢰성 간접 증거). 갤러리: `docs/assets/coins_mm.png`.

**의미**: "1.53mm" 류 측정의 분할+직경 체인이 실사진에서 ~1.7% 정확 — Measure Anything의
"±10%(와인병 1개)"보다 강한 첫 promptable 계측 수치 (비교표 (d)의 우리 셀 시작).

## E-cnt-1 — rebar 카운팅 ⚠️ 정직한 부정적 결과 (능력 갭 확정)

ROI-1555 (labelme 폴리곤, GT=인스턴스 수), n=40, zero-shot:

| 프롬프트 | MAE (GT평균 22.5) | 상대오차 | ±10% 정확률 |
|---|---|---|---|
| rebar | 16.07 | 72.8% | 0% |
| **steel bar (best)** | 13.12 | **59.5%** | 5% |
| metal rod / circular cross section / rebar end / pipe (n=20 스윕) | 13~20 | 80~100% | 0% |

**판정**:
1. **어휘 문제 아님** — 6개 프롬프트 전부 실패 → SAM3의 능력 갭 (논문 §B 자인: "niche visual
   domain의 fine-grained 개념 일반화 약함"의 실증).
2. **밀도 자체가 원인 아님** — 동전 60개는 검출함. 원인은 **맞닿은 저대비 녹슨 단면**이라는
   시각 도메인. 지도학습 앵커(UAV rebar 86.27% acc)와의 갭이 카운팅에서의 fine-tune/밀도
   폴백 필요성을 정량화 — 아키텍처의 "SAHI/density fallback" 슬롯이 실증으로 정당화됨.
3. 비교표 (b)의 우리 셀: "zero-shot 한계 보고 + 분리객체(동전)에선 신뢰" 2면 보고.

## 논문 자료 (같은 날 완료)
- [paper/DATASETS.md](../../paper/DATASETS.md): 사용/확보/후보/불가 전수표
- [paper/RELATED_BASELINES.md](../../paper/RELATED_BASELINES.md): 비교표 4종 설계 + PDF-verified 앵커
  (핵심: 우리 zero-shot 0.442 ≈ SAC fine-tuned 44.13 / SAM3 크랙 IoU 첫 보고 자리 /
  Measure Anything 캘리퍼 MAE 부재 / MT per-class 보고 공백)

## 다음
- E-mm-3: krkCMd 크랙 폭 물리 GT (선별 다운로드 후) — 비교표 (d) 완성
- E-mm-2: T-LESS CAD+pose 치수 유도 (다운로드 완료)
- 카운팅 개선 트랙: SAHI 타일링 또는 rebar 소량 fine-tune (능력 갭 해소 시도)
