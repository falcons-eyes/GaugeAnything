# Related Work & Baseline Landscape — 비교표 설계 (검증 수치)

> 논문 비교 실험 설계용. **[PDF-verified]** = 원문 PDF에서 직접 추출한 수치 (인용 가능).
> 조사 시점 2026-06-11. 캐벗은 각 절 끝에 명시 — camera-ready 전 재확인 목록 포함.

## ⚠️ 핵심 함정: CrackSeg9k의 두 가지 mIoU (논문 정확성의 사활)

- CrackSeg9k 계열 논문: **2-class mIoU = mean(배경 IoU≈0.95+, crack IoU)** → 0.77~0.82대
- SAM 계열 크랙 논문: **crack-only IoU** → 0.44~0.65대
- **혼합 비교 금지.** 우리 Gauge-Bench(0.442±0.011)는 **crack-only** — SAM 계열과 직접 비교 가능.

## 전략적 발견 4가지 (포지셔닝 결정)

1. **우리 zero-shot 0.442 ≈ SAC의 fine-tuned 44.13 IoU** — LayerNorm 튜닝된 SAM(SAC,
   ASCE JCCE 2026)이 OmniCrack30k 학습 후 도달한 수치를 우리는 **학습 0으로** 달성.
2. **SAM3 크랙 분할의 peer-reviewed 픽셀 IoU는 아직 부재** (2026-06) — 블로그 벤치(AP 0.109)와
   detection-level 교량 파이프라인(arXiv 2601.17254)뿐. **우리가 첫 보고가 됨.**
3. **Measure Anything은 캘리퍼 MAE를 보고하지 않음** — 와인병 1개 "±10%"가 전부.
   promptable 계측의 정량 비교표는 사실상 비어 있음 → 캘리퍼/물리 GT MAE를 내는 쪽이 표를 지배.
4. **MT fray/uneven의 per-class 수치는 거의 표로 보고된 적 없음** (DLI는 그림만, CINFormer는
   전체 mIoU만) → 우리 soft-map per-class 보고가 보고 공백을 채움.

## (a) 크랙 분할 비교표 (행 후보)

| 베이스라인 | 유형 | 수치 (출처) |
|---|---|---|
| DeepLabV3+(R101) | 지도 | CrackSeg9k 2-class mIoU 0.7599 (+DINO 0.7712) [원논문] |
| HrSegNet-B48 | 지도 실시간 | 2-class mIoU 80.32 @140FPS [arXiv 2307.00270; 저널판 80.56 — 한 판본 일관 인용] |
| CrackMamba | 지도 2024 | 81.75 (⚠ 재구성 split 8,751 — 우리 split으로 재실행 필요, 코드 공개) |
| nnU-Net | 지도 (OmniCrack30k 우승) | clIoU₄px 64% — "전문 크랙넷 전부 격파" [CVPRW 2024] |
| CrackSAM(LoRA) | SAM1 PEFT | crack-IoU 0.6416 in-domain / 0.45–0.68 zero-shot [arXiv 2312.04233] |
| **SAC** | SAM LayerNorm만 (41K) | **IoU 44.13 / F1 61.22** (OmniCrack30k 학습) [arXiv 2504.14138] |
| SECrackSeg | SAM2 어댑터 | CFD 0.854 (⚠ 2-class 의심) [Sensors 2025] |
| **SAM3 zero-shot (우리)** | 학습 0 | crack-only 0.442±0.011 — 첫 공개 SAM3 크랙 IoU |

지표: crack-only IoU(1차, 정의 명시) + F1 + **clIoU₄px/clDice**(thin 공정성) + noncrack 클린율(우리 고유).

## (b) 카운팅 비교표

| 앵커 | 수치 [PDF-verified] |
|---|---|
| SAM 3 공식 | SA-Co/Gold cgF1 **54.1**(인간 72.8의 74%) · CountBench **MAE 0.12/93.8%** · PixMo 0.21/86.2 · LVIS mask AP 48.8(초록; Table 48.5 — 최신판 확인) |
| FSC-147 SOTA | CountGD(text+exemplar) test **6.75/43.65** (GT보정 5.74) · GeCo 7.91 · DAVE 8.66 · CACViT 9.13(⚠ 인용표 출처) |
| Rebar | RebarDSC SSL **AP 65.7/AP.75 81.6** [IJCAI 2021] · UAV rebar **count acc 86.27%/AP50 87.71** [Sci Rep 2025] · MaskID F1>0.99 [PLOS One 2023] |

우리 셀: ROI-1555/RebarDSC count MAE + SAM3 직접 비교 (E-cnt-1).

## (c) Soft/연속 결함맵 비교표 (MT fray/uneven)

| 앵커 | 수치 |
|---|---|
| MT 지도 SOTA | CINFormer **mIoU 86.5** (UNet 78.7, SegFormer 68.6) [arXiv 2309.12639, PDF-verified] · DLI UNet **0.831±0.018** [ECCV-W 2024] — **둘 다 per-class fray/uneven 표 없음** |
| zero-shot 텍스트 AD | AnomalyCLIP MVTec pixel-AUROC **91.1**/PRO 81.4, VisA 95.5 [ICLR 2024] · WinCLIP 85.1/64.6 |
| 연속맵 UAD SOTA | Dinomaly P-AUROC **98.4**(텍스처 98.1–99.2) [CVPR 2025] · U-Flow 98.74 |
| mura | 공개 pixel-AUROC 벤치 **부재** (TFT-LCD 사설 데이터 acc ~94-97%) — 우리 MT-Uneven 연속맵이 공백 |

우리 셀: 고전 조명잔차 0.669/DRAEM 0.636 (test) + SAM3 binary ~0.50 + per-class 보고.

## (d) 계측 비교표 — 우리의 본진

| 앵커 | 정확도 | 셋업 |
|---|---|---|
| Measure Anything [arXiv 2412.03472] | 와인병 1개 "±10%", **캘리퍼 MAE 없음**; canola는 정성+mAP만 | SAM2+**스테레오** |
| 이웃최단거리 [Eng.Struct 2025] | r=0.962, **RMSE 0.24mm** (합성 크랙) | RGB 단안 |
| RGB-D crossfusion [CBM 2025] | 절대오차 **<0.05mm** (<0.3mm 크랙) | RealSense **L515 필요** |
| 레이저빔 [Appl.Sci 2023] | 0.02–0.57mm | 레이저 투사 필요 |
| k-means 프로파일 [Sensors 2021, 코드有] | t-test 무유의차(수동 대비), 분해능 0.15mm | DSLR 11.5px/mm |

**우리 포지션**: RGB 단안 + promptable + 마커 기반 — "RGB-only 골격법(±0.24mm)과
센서 의존법(±0.05mm) 사이"를 타깃, 단 **promptable로는 최초의 캘리퍼/물리 GT MAE 보고**.
E-mm-3(krkCMd 물리 폭 19k개)가 이 표의 우리 셀을 채움.

## Camera-ready 전 재확인 목록
- SAM3 LVIS 48.8 vs 48.5 (arXiv 최신판) · CACViT 원논문 수치 · HrSegNet 판본 선택
- CrackMamba를 우리 split으로 재실행 (코드 공개됨) · CrackFormer-II 표의 데이터셋↔수치 매핑
- DLI fray/uneven 그림 수치화 시 저자 문의 · ClipSAM 본문 표 추출
- CrackSeg9k 3개 유통본(원본 9,255/khanhha/재구성 8,751) 중 우리 버전 명시

## 인용 키
SAM3 arXiv:2511.16719 · SAC 2504.14138(ASCE JCCE 2026) · CrackSAM 2312.04233 ·
OmniCrack30k CVPRW'24 · CrackMamba 2410.19894 · HrSegNet 2307.00270 · CountGD 2407.04619 ·
GeCo 2409.18686 · DAVE CVPR'24 · RebarDSC IJCAI'21 · CINFormer 2309.12639 · DLI 2408.10031 ·
AnomalyCLIP ICLR'24 · Dinomaly 2405.14325 · MeasureAnything 2412.03472 · SAM3교량 2601.17254
