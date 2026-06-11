# 2026-06-11 — OSS 분리 + 방향성 matting v2 성공 + 라우터 통합(#3)

## OSS 마이그레이션 ✅

GaugeAnything이 독립 공개 repo로 분리됨: **github.com/falcons-eyes/GaugeAnything** (Apache-2.0).

- 구조: `src/gauge` → **`gaugeanything` 패키지** (pip install -e . 가능), experiments/(비전 트랙만),
  docs/(페이지+감사+진행로그), paper/, data/. import 전면 재작성.
- OSS 파일: 영문 README(감사 후 공식 수치 표 포함), pyproject, CITATION.cff, .gitignore 확장.
- Spark 작업본: `/home/hwoo_joo/github/GaugeAnything` — datasets/.venv는 구 repo 심볼릭 링크,
  checkpoints 복사. **selftest 14/14 + soft_selftest 11/11 통과로 재구조 검증.**
- 이후 모든 작업은 이 repo 기준. (모 repo `falconoon.com/Industrial_Anything`은 센서 트랙 보존.)

## #1 방향성 matting v2 — 실 전이 실패 해결 ✅ ⭐

v1(블롭 합성)의 실 fray 전이 실패(보존성 0.483)를 **방향성 합성**으로 교정:
연신 이방성(5~15×) + 가닥(strand) 경계 + 텍스처 교란 외형 + coarse 50% 무지터("재형성 말고 정제").

| 지표 | v1 블롭 | **v2 방향성** | guided_matte(고전) |
|---|---|---|---|
| 실제 MT_Fray 보존성 IoU (n=32) | 0.483 ⚠️ | **0.949** | 0.860 |
| 경계 softness | 0.640 | 0.590 | — |
| 합성 holdout α-MAE | 0.0100 | 0.0056 (binary 0.0083) | — |

**판정**: 학습 헤드가 실데이터에서 고전을 처음으로 능가(0.949 > 0.860), softness 유지.
보류 해제 조건 충족 — 라우터 fuzzy regime의 학습 헤드 채택 후보로 복귀.
주의: 합성 마진이 줄어든 것(0.0056 vs 0.0083)은 v2 coarse가 α에 더 가깝게 설계된 때문 —
의미 있는 지표는 실데이터 보존성. 체크포인트: `checkpoints/matte_fray_directional.pt`.
여전한 한계: α 정확도 자체는 합성 한정(실 α GT 부재), 단일 데이터셋(자성타일).

## #3 라우터·페이지 통합 ✅ (앞서 완료)

- `inspect_soft(marker_size_mm=)`: ArUco→**PlaneScale homography 국소 스케일** (인스턴스 중심별)
- `segmenter="sam3_ensemble"`: 동의어 붕괴 방지
- 페이지 "Metrology Rigor — Trust the Millimeters" 섹션 (틸트 19.3%→0.7%, 앙상블 0→0.374)

## #2 M2 측정 인식 refiner 🔄

캐시: train 1192 / val 149 / test 219 (홀드아웃 소스 cfd·cracktree200·deepcrack).
학습 진행 중 — ep5에서 val width bias −0.684→−0.220 교정 중. 완료 후 공식 test 수치 별도 기록.
