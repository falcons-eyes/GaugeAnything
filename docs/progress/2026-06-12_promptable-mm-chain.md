# 2026-06-12 (2부) — promptable mm 체인의 3대 진전 (E-mm-2b · M2v2-a · E-mm-3b)

감사(RESEARCH_AUDIT)가 지목한 최대 갭 — "image-level promptable 분할 ↔ 물리 단위 GT" —
을 정면 공략한 하루. 세 실험이 한 사슬로 연결된다.

## E-mm-2b — promptable 산업부품 계측 실증 ⭐⭐

N2(완벽 마스크 상한)와 동일 케이스, 마스크만 SAM3 zero-shot으로:

| prompt | 매칭 | IoU중앙 | 측정 rel_err 중앙 | ±10% |
|---|---:|---:|---:|---:|
| electrical component / plastic part / white object | **100%** | **0.94** | **2.5%** | 91~94% |
| industrial part | 0% | — | — | — |
| (N2 상한) | — | 1.0 | 2.83% | 94% |

**분할이 만드는 측정 비용 ≈ 0** — "prompt→산업부품 mm"가 CAD 유도 물리 GT로 입증.
추상 명사 0%는 동의어 붕괴의 제3 도메인 재현 (구체 명사 + 앙상블 원칙 재확인).

## M2v2-a — 로짓 iso-level 폭 보정: 합격선 돌파 ⭐

가설 사슬: N3("bias 원인=마스크 두께") + D("로짓 노출") → mask_threshold가 폭의 직접 노브.
1 forward → 9 threshold post_process, θ* 선택은 train+val만, 보고는 홀드아웃 test:

| 방법 (파라미터 수) | test rel_err | bias |
|---|---:|---:|
| default 0.5 | 0.730 | +0.680 |
| M2 v1 신경 (1.9M) | 0.564 | +0.503 |
| 분위 보정 (5) — 기존 합격선 | 0.480 | +0.411 |
| **θ\*=0.7 + 분위 보정 (6)** | **0.437** | +0.367 |

학습 0으로 −40%. 한계(정직): cracktree200 0.664 — 보정의 천장은 마스크 품질.

## E-mm-3b — prompt→mask→μm 체인 개통 (물리 GT, 정직한 첫 수치)

**기술 돌파**: ① 38.6GB zip에서 HTTP Range로 TIF 1장(1.07GB)만 추출 ② Gridline 컬럼이
ImageJ ROI 좌표(yyyy-xxxx, y=라인 중점)임을 해독 — 19,098 프로파일의 이미지 좌표 직접
복원 (상관 0.95 게이트, 복원율 73~88%).

| 단계 | MAE (μm) | 의미 |
|---|---:|---|
| rung1: 이미지 추출 프로파일+minrun5 | **35~43** (best stages) | profile-level 앵커(31.3)와 일치 → **체인 검증** |
| rung2: SAM3 ds4/th0.5 | 203~264 | zero-shot 마스크 폭의 현주소 |
| rung2: SAM3 ds2/**θ\*=0.7** | **144~186 (−30%)** | **M2v2-a 노브의 cross-dataset·물리단위 전이** |

남은 갭(144 vs 35μm)의 병목 = zero-shot 마스크 경계 — M2 v2 본 트랙(로짓 입력 마스크
정제)의 실증적 근거.

## 오늘의 사슬 (서사)
완벽 마스크 상한 2.83%(N2) → SAM3로도 2.5%(E-mm-2b, 부품) — **잘 분할되는 도메인에서
promptable 계측은 이미 작동**. 크랙(thin)에선 마스크가 병목(N3·M2v2-a·E-mm-3b 일관) —
로짓 노브로 −30~40% 회복했고, 다음은 마스크 자체 정제.

## 산출물
- experiments: tless_sam3_eval.py · m2v2_logit_threshold.py · krkcmd_image_eval.py
- results: tless_sam3_eval.json · m2v2_logit_threshold.json · krkcmd_image_eval(.json/_ds4th05.json)
- 데이터: krkCMd TIF 2장(Range 추출), RESULTS.md 3개 섹션 추가
