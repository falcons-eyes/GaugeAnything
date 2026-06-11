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

## #2 M2 측정 인식 refiner ✅ — 부분 성공 + 새 발견

**설정**: frozen SAM3 마스크 + 이미지 → refiner(1.9M), 손실 = soft-Dice + 0.3·면적보정.
train 1192 / val 149 (학습 소스) / **test 219 (홀드아웃 소스 cfd·cracktree200·deepcrack)**.
체크포인트 선택은 val만, 보고는 test만. `checkpoints/m2_refiner.pt`.

| 홀드아웃 test (n=219) | mIoU | width rel.err ↓ | width bias |
|---|---|---|---|
| raw SAM3 | 0.482 | 0.730 | +0.680 |
| **refined (M2 v1)** | 0.487 | **0.564** | **+0.503** |

**판정**:
1. **폭 상대오차 23% 상대 감소** (0.730→0.564), bias 0 방향으로 교정, mIoU 무손상(0.482→0.487).
2. **새 발견 — bias 부호가 도메인 의존**: val(학습 소스)에서는 음수(최종 −0.17, 과소),
   홀드아웃에서는 양수(+0.68, 과대 — thin 크랙에서 SAM3 마스크가 GT보다 두꺼움).
   기존 "−22% 과소추정"은 소스 혼합 평균이었던 것. **전역 단일 refiner로는 미지 도메인
   폭 보정이 구조적으로 불완전** → 도메인 적응/스케일 인식 보정이 M2 v2 과제.
3. val 곡선 요동(bias −0.68↔−0.02)도 같은 원인 — 폭 보정 목표가 소스마다 반대 방향.

→ 로드맵 갱신: M2 v1 완료(부분), M2 v2 = 도메인 조건부 폭 보정(per-source 통계 또는
스케일 토큰 주입). 폭 GT는 여전히 마스크 유래 — 실 mm GT가 근본 해결책.
