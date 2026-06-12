> ⚠️ LEGACY: 이 아웃라인은 피벗 이전 센서 트랙용. **현행 논문은 `main.tex`/`refs.bib`** (vision GaugeAnything).

# Paper Outline — *Industrial Anything: A Promptable Multimodal Foundation Model for Machine Health*

> Step 4 산출물. LocateAnything 스타일 공개(논문 + 프로젝트 페이지 + HF + GitHub)를 위한 논문 골격 + Abstract 초안.
> 타깃: NeurIPS/ICML Datasets&Benchmarks, 또는 RESS/MSSP (PHM 저널).

## Abstract (초안)

> Foundation models have transformed vision and language through *promptable*, generalist
> formulations—Segment Anything, Depth Anything—yet industrial machine-health monitoring
> remains fragmented into per-machine, per-sensor, single-modality models that fail to
> transfer across assets and operating conditions. We introduce **Industrial Anything**,
> the first **promptable multimodal foundation model** that unifies vibration, acoustic,
> motor-current, temperature, and thermal/RGB sensing for fault diagnosis, anomaly
> detection, and remaining-useful-life (RUL) estimation under a single task formulation:
> **Promptable Machine Health Inference**—given a synchronized sensor window and a prompt
> (a normal-reference snippet, a fault exemplar, or a text description), the model emits
> anomaly scores, fault labels, localized fault intervals, and RUL estimates as parallel
> atomic units. We contribute (1) a **harmonization pipeline** that aligns heterogeneous,
> differently-sampled sensor streams from 15+ public datasets into a canonical multimodal
> schema; (2) a **Parallel Diagnosis Decoding** head adapting LocateAnything's parallel box
> decoding to health inference; (3) a **simulation-teacher → real pseudo-label** recipe that
> escapes the fault-label bottleneck; and (4) a **cross-machine / cross-condition benchmark**
> where entire machines and operating regimes are held out at test time. [결과 placeholder].
> Weights, data tooling, and the benchmark are released openly.

## 1. Introduction
- 산업 진단의 파편화 문제 (per-machine/per-sensor 모델, 전이 실패)
- "Anything" 패러다임의 부상 (SAM/Depth Anything) → 산업으로 일반화되지 않음
- **갭**: 진짜 멀티모달(진동+음향+전류+열+RGB) promptable PHM FM은 부재
  - UniFault=진동 단일, AnomalyCLIP=비전 전용, BearLLM=신호→텍스트(손실)
- 기여 4가지 (위 abstract)

## 2. Related Work
| 줄기 | 대표 | 우리와의 차이 |
|---|---|---|
| Promptable FM | SAM, SAM2, LocateAnything | 비전/언어 → 우리는 센서 |
| Pseudo-label FM | Depth Anything V2 | depth → 우리는 fault, 시뮬레이션 teacher |
| TS FM | TimesFM, Chronos, Moirai, MOMENT | 저주파 예측 → 고주파 진동 OOD |
| PHM FM | UniFault, BearLLM, PHM-GPT | 단일 모달 / 텍스트 변환 → 우리는 멀티모달 융합 |
| Zero-shot AD | AnomalyCLIP, WinCLIP | 이미지 → 우리는 센서+이미지 |

## 3. Task Formulation — Promptable Machine Health Inference
- 입력: 동기화 멀티모달 윈도우 + 프롬프트(레퍼런스/예시/텍스트/포인트)
- 출력: {이상점수, 고장클래스+신뢰도, [t_start,t_end] 구간, RUL}
- 4개 다운스트림을 프롬프트 엔지니어링으로 흡수 (재학습 X)
- 모호성 → 복수 후보 + 신뢰도 (SAM multi-mask 차용)

## 4. Method
### 4.1 Harmonization (→ `src/harmonization/`)
- 정준 스키마, 채널 정규화, 정준 레이트 리샘플, 공통 윈도우 정렬
- 정준 고장 택소노미 (cross-dataset 라벨 통일)
- Cross-Domain Temporal Fusion (분포 시프트 선제 대응)
### 4.2 Architecture
- 모달별 인코더 (진동→스펙트로그램 ViT/BEATs, 전류→TS Transformer, 열/RGB→DINOv2)
- 융합 트렁크 (Meta-Transformer/Perceiver, any-variate 잠재배열, 모달 드롭아웃)
- 프롬프트 인코더 (레퍼런스/텍스트/포인트)
- **Parallel Diagnosis Decoding** (LocateAnything PBD 차용)
### 4.3 Pretraining
- 모달별 SSL (마스킹 복원 + 센서인식 대조학습)
- 시뮬레이션 teacher → 실측 pseudo-label
- 멀티모달 대조 정렬 (ImageBind식)

## 5. The Industrial-Anything Benchmark
- 15+ 데이터셋 통합 (Step 1 인벤토리), 상업/연구 트랙 분리
- 프로토콜: in-dist / cross-load / **cross-machine** / few-shot (EXP1)
- 지표: macro-F1, cross-machine acc, few-shot 곡선, RUL RMSE, 모달결측 강건성
- PHM-Bench 연동

## 6. Experiments (계획 + 예비결과)

### 예비 motivating result (실측, GB10) ⭐
naive 1D-CNN으로 **cross-source 베어링 분류**(HF `adyady` 코퍼스, 동일 라벨 공유 소스쌍):
- in-source macro-F1 최대 **1.000** (mechanical_bearing 단일 소스)
- 같은 모델 **cross-source macro-F1 0.17** (랜덤 수준)
- **도메인 갭 +0.588** → 단일소스 정확도는 허상, cross-source 일반화 실패가 핵심 문제.
- 이것이 본 연구의 동기: 하모나이제이션 + 도메인 불변 사전학습으로 이 갭을 닫는다.
- (방법론 교훈: 소스-라벨 confound 시 100% 허위 정확도 발생 → 공유 라벨 쌍 설계 필수)

### 본 실험 (계획)
- Table 1: 백본 비교 (cross-source/cross-machine 전이) — H1/H2 검증
- Table 2: 멀티모달 vs 단일모달 — H3
- Table 3: few-shot 곡선 vs UniFault(IMS 82.94%@100)
- Fig 3: cross-source 전이 히트맵 + 도메인 갭 축소(우리 모델 vs naive)
- Ablation: 하모나이제이션 / 시뮬 pseudo-label / 모달 드롭아웃

## 7. Limitations
- 공개 멀티모달 동기 데이터 희소 (KAIST/Paderborn/MaFaulDa 중심)
- MCSA 대형 공개셋 부재
- 시뮬레이션-실측 갭 잔존
- 라이선스: 상업 트랙은 데이터 제약

## 8. Release (LocateAnything 스타일)
- 프로젝트 페이지 (이 repo `docs/`)
- HuggingFace: 모델 가중치(상업/연구 2종) + 데이터 툴링
- GitHub: 하모나이제이션·벤치·학습 코드
- 논문 (arXiv)

---

## 네이밍 후보
- **Industrial Anything** (컨셉 명확)
- **FaultAnything** / **DiagnoseAnything** (Anything 계보)
- **FalconFM** (브랜드)
→ 1순위 권장: 논문명 *Industrial Anything*, 모델명 *FaultAnything*, 제품 *FalconEyes*

## 타임라인 (제안)
| 분기 | 마일스톤 |
|---|---|
| Q1 | 데이터 확보·하모나이제이션 완성, 백본 벤치(EXP1) 실측 |
| Q2 | 융합 트렁크 + SSL 사전학습, 시뮬 teacher 구축 |
| Q3 | Promptable 학습 + 벤치 평가, 논문 초고 |
| Q4 | HF/GitHub/프로젝트 페이지 공개 + arXiv |
