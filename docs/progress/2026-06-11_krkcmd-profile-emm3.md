# 2026-06-11 — E-mm-3 krkCMd 물리 폭 GT: profile-level μm 검증

## 배경
E-mm-1 동전은 실사진의 분할→직경 체인을 검증했지만, 크랙 폭 자체의 물리 단위 GT는 아니었다.
krkCMd는 6400dpi 스캐너 기반 콘크리트 크랙 profile 19,098개와 수동 폭(`MANwidth`, μm)을
제공한다. 전체 이미지 zip은 약 38GB지만, 공개 table만으로 501px cross-crack brightness profile의
폭 추정 벤치마크를 수행할 수 있다.

단위 변환은 동봉 스크립트와 동일하게 `25.4 / 6400 * 1000 = 3.96875 μm/px`.

## 프로토콜
- 데이터: `datasets/krkcmd/krkCMd_table.csv`
- GT: `MANwidth` (manual crack width, μm)
- 크기: 19,098 profiles, 36 series/image groups, stage 1-6
- split: series/image group 단위 80/20 deterministic split
  - train 14,424 profiles
  - test 4,674 profiles
- 평가: MAE/RMSE/median abs err/bias/pass@50μm/pass@100μm/Pearson r
- 보고 수치: group-split test 우선, all도 JSON에 보존
- 스크립트: `experiments/krkcmd_profile_eval.py`
- 산출물: `experiments/results/krkcmd_profile_eval.json`, `docs/assets/krkcmd_profile.png`

## 공식 수치 (test)

| 방법 | test MAE ↓ | RMSE ↓ | medAE ↓ | pass@50μm | r |
|---|---:|---:|---:|---:|---:|
| **DLMwidth (저자 DLM)** | **11.1 μm** | 22.9 | 5.5 | 96.5% | 0.973 |
| **GaugeProfile-minrun5 + linear-cal** | **25.9 μm** | 50.6 | 16.0 | 84.6% | 0.864 |
| AEDwidth (저자 고전 분석법) | 26.5 μm | 40.0 | 23.8 | 91.3% | 0.930 |
| GaugeProfile-minrun5 (무보정) | 31.3 μm | 50.1 | 23.8 | 88.6% | 0.864 |
| GaugeProfile-halfdepth + linear-cal | 55.2 μm | 83.3 | 41.9 | 62.9% | 0.630 |
| GaugeProfile-otsu + linear-cal | 97.5 μm | 124.6 | 80.0 | 34.3% | 0.014 |

All-profile 수치: DLM MAE 13.9μm, GaugeProfile-minrun5+cal 27.3μm, AED 34.1μm.

## 발견
1. **비교표 (d)의 물리 폭 GT 셀을 채움**: krkCMd에서 promptable/image-level이 아닌
   profile-level이지만, 크랙 폭을 μm 단위로 평가했다.
2. **저자 DLM은 매우 강함**: test MAE 11.1μm로 수동 폭에 거의 붙는다. 이는 우리 논문에서
   "profile-specialized supervised upper/anchor"로 두는 것이 정직하다.
3. **단순한 고정 profile rule도 AED와 같은 급**: `min+5 brightness run` 규칙은 무보정
   31.3μm, group-split 선형 보정 후 25.9μm로 AED 26.5μm와 거의 동일. GaugeAnything의
   폭 계측 코어를 물리 GT에 연결할 때 필요한 baseline floor가 생겼다.
4. **Otsu/FWHM류는 profile 도메인에서 취약**: 이미지 분할식 threshold를 profile에 그대로
   쓰면 55-174μm 수준으로 악화된다. profile 폭은 valley-local 규칙이 더 맞는다.
5. **한계**: 이번 결과는 full image segmentation이 아니라 krkCMd table profile 입력이다.
   "RGB promptable crack mask → μm" 주장은 아직 ArUco/캘리퍼 또는 이미지 zip 기반 추출이 필요하다.

## 다음
- E-mm-2: T-LESS CAD+pose로 산업 부품 치수 유도 검증
- krkCMd image zip subset 추출 가능성 확인: 원본 image ROI와 profile 위치를 연결해
  image-level segmentation→profile-level width 비교로 확장
- M2 v2: profile/scale 토큰을 쓰는 domain-conditioned width calibration
