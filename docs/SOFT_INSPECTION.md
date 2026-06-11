# Soft Inspection — 경계 없는/애매한 결함을 위한 연속 표현

> 멀티도메인 검증에서 SAM3 zero-shot이 fray(0.03)·uneven(0.005)에 실패한 이유를 분석하고,
> binary segmentation을 넘어선 **연속(soft) 표현**으로 해결하는 방법론. 문헌 검증 완료.

## 1. 왜 binary가 실패하는가 — 결함의 3가지 regime

결함은 경계의 성질로 셋으로 갈린다. 도구가 regime과 맞아야 한다.

| Regime | 결함 예 | 경계 | 올바른 도구 | 출력 |
|---|---|---|---|---|
| **(a) 선명 경계** | 크랙, 홀, 스크래치, 덴트 | 뚜렷 | **binary segmentation** (SAM3) | 0/1 마스크 |
| **(a′) 애매 경계** | **fray**, 마모 가장자리, 미세 크랙 | 흐릿하지만 *존재* | **alpha matting** | α∈[0,1] (혼합 불투명도) |
| **(b) 경계 없는 장(field)** | **uneven/mura**, 확산 부식, 음영 드리프트 | *없음* (완만한 장) | **조명/intrinsic 모델 + soft regression** | severity field |

핵심 통찰: **matting은 `I = αF + (1−α)B`로 전경 F와 배경 B를 분리**한다.
- fray는 F(마모 섬유/공동)와 B(정상 표면)가 있고 경계만 흐림 → **matting의 정확한 영역**.
- uneven은 분리할 F가 없음(완만한 밝기장) → matting의 α가 무의미 → **장 모델링**이 답.

문헌 근거: 의료 Matting(MICCAI'21)이 "병변↔정상 혼합 계수 = α"로 fuzzy 경계를 binary보다 잘 표현함을 실증.
uneven은 디스플레이 검사의 **"mura"** 문제로, basis-image/RPCA/Retinex/saliency+intrinsic(IEEE 9229127)이 직격 선행연구.

## 2. (a′) Fray — SAM3 마스크 → soft alpha (matting)

**레시피 (license-clean)**: SAM3 binary 마스크 → auto-trimap → matting → α
1. `fg = erode(mask, k, it)`, `unknown = dilate(mask,k,it) − fg`, `bg = 나머지` (Matte-Anything 레시피, ~10줄 OpenCV)
2. matting head로 α 추정. **라이선스 주의**: ViTMatte/MAM/Matte-Anything 가중치는 Adobe Comp-1k(비상업) 오염.
   - **PoC(고전, 깨끗)**: guided filter / closed-form matting — 학습 불요, 라이선스 자유. 경계를 이미지 엣지로 feather.
   - **production**: MAM의 M2M 헤드(~2.7M)나 MODNet(Apache)/P3M(MIT)을 **자체 결함 α로 fine-tune** (포트레이트 도메인이라 필수).
3. unknown 밴드 폭 = 경계 흐림 예산. fray는 크게, 크랙은 작게.

`src/gauge/soft.py: guided_matte()` — guided filter 기반 PoC 구현.

## 3. (b) Uneven/mura — 조명장 모델 + 잔차 (matting 아님)

**핵심 재framing**: 불균일 *자체*가 매끄러운 장 → 매끄러운 모델을 적합하고 **잔차 = 결함**.
1. **매끄러운 배경/조명 적합**: 2차 2D 다항식 surface fit (또는 large-kernel 가우시안 / Retinex / 저랭크).
2. **잔차** `r = image − fit` (또는 image ÷ illumination). 이 잔차가 연속 severity field — 본질적으로 soft, 경계 불요.
3. **severity 정량 (ISO 25178 areal roughness)**: `Sa = mean|r|`, `Sq = rms(r)`, gradient percentile.
4. (학습형) DRAEM(MIT)의 Perlin 패치 합성을 **저주파 밝기 섭동**으로 교체해 점진 불균일을 학습.
5. **boundary loss 금지** — 없는 경계를 날카롭게 만들 뿐. soft-Dice/L1 사용.

`src/gauge/soft.py: illumination_residual(), mura_severity()` 구현.

## 4. Soft 마스크에서의 측정 — GaugeAnything 계측의 연속화

binary `>0.5` 카운트는 경계 partial-volume 정보를 버린다. soft α는 그것을 복원한다 (의료 partial-volume 문헌).

| 측정 | binary | **soft (구현)** |
|---|---|---|
| 면적 | 픽셀 수 | `area = Σα · mm_per_px²` (partial volume) |
| 경계 | 계단형 | **α=0.5 iso-contour** 선형보간 (marching squares) — sub-pixel |
| 폭(크랙) | 2·EDT(skeleton), burr 문제 | 수직 프로파일 **soft 적분** `width = Σ_perp α` (흐릿해도 graceful) |
| severity(확산결함) | — | **extent × intensity**: `Σα/ROI` × `mean/p90(α)`, 로그 전이(ASTM D610) → Good/Fair/Poor/Severe(AASHTO) |
| 불균일 severity | — | 조명보정 잔차의 **Sa/Sq** (ISO 25178) |
| 불확실성 | — | TTA/MC 샘플 N개 측정 → mean ± 90% CI; 면적 근사 `Var=Σα(1−α)·px⁴` |

전제: **α 캘리브레이션** (temperature/Platt scaling)이 없으면 위 수치는 편향. soft 헤드는 sigmoid + soft-Dice + soft GT(SoftSeg 레시피).

## 5. GaugeAnything 통합 — regime 라우팅

```
SAM3 마스크 + 프롬프트
   │
   ├─ classify regime (경계 선명도/장 성질)
   │    선명 → binary (현행)
   │    애매 → guided_matte() → soft α   ───┐
   │    장   → illumination_residual() ────┤
   │                                        ▼
   └────────────────────────────► soft 측정 모듈 (§4)
                                   {width/area sub-pixel, severity, ±CI}
```

regime 분류는 (i) SAM3 마스크 경계의 그래디언트 선명도, (ii) 마스크 외부 잔차장의 매끄러움으로 추정.
fray처럼 SAM3가 마스크조차 못 줄 땐 anomaly map(잔차/DRAEM)으로 시드 → matting/soft.

## 6. 라이선스 요약 (상업 트랙)
- ✅ guided filter / closed-form matting (고전, 자유) · MODNet(Apache) · P3M(MIT) · DRAEM(MIT) · anomalib(Apache, PatchCore/RD4AD/FastFlow)
- ❌ ViTMatte/MAM/Matte-Anything/DiffMatte/MatteFormer **가중치**(Adobe Comp-1k 비상업) · RVM(GPL) · EfficientAD(MVTec 특허)
- 전략: 고전 PoC로 가설 검증 → 자체 데이터로 license-clean 헤드 fine-tune.

## 7. 핵심 레퍼런스
- 의료 Matting (fuzzy 경계 α): MICCAI'21 Paper0097 · ScienceDirect S0010482523001798
- Matte-Anything trimap 레시피: arXiv 2306.04121 · MAM M2M: arXiv 2306.05399
- mura/uneven: basis-image S0031320309003446 · saliency+intrinsic IEEE 9229127 · STAR Retinex 1906.06690 · RPCANet++ 2508.04190
- soft 측정: Continuous Dice 1906.11031 · SoftSeg 2011.09041 · partial volume S1361841502000610
- severity: ASTM D610 · AASHTO CS · 부식 깊이 회귀 PMC12227716
- 불확실성: biomarker ± 1806.08640 · contour sampling 2502.12713
- roughness: ISO 25178 (Sa/Sq)
- anomaly(연속맵): DRAEM(MIT) 2108.07610 · anomalib
