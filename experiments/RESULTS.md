# 실험 결과 — Industrial Anything

> DGX Spark (NVIDIA GB10 Grace Blackwell, aarch64, CUDA 13.1) 실측. torch 2.12.0+cu130.
> 데이터는 HuggingFace `datasets`로 확보 (느린 Mendeley/NASA 미러 우회).

## 환경 검증 (GB10)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GB10 (Grace Blackwell), 통합메모리 130.7 GB |
| torch | 2.12.0+cu130 (CUDA 13.0 built), capability sm_121 |
| matmul fp16 (정상상태) | **~98 TFLOP/s** (8192³, 워밍업 후) |
| bf16 | ~99 TFLOP/s |
| 메모리 대역폭 | 218 GB/s (LPDDR5X 통합) |
| 비고 | sm_121 네이티브 커널 없음 → sm_120 PTX JIT 폴백 (성능 양호) |

## EXP — Cross-Source 베어링 분류 (헤드라인) ⭐

**데이터**: `adyady/bearing-fault-dataset` (여러 공개 베어링 셋을 통합·정규화한 318K 코퍼스).
**프로토콜**: 동일 라벨 집합을 공유하는 두 소스 선택 → in-source vs cross-source 비교.
**모델**: 경량 1D-CNN (68K params), 25 epochs, GB10 GPU.

선택 쌍: **A=`imfds_motor`, B=`mechanical_bearing`**, 공유 라벨 = {bearing_inner, bearing_outer, normal} (3클래스).
- A: 4,500 샘플 (1,500/클래스), B: 1,008 샘플
- 두 소스 모두 동일 3클래스 보유 → **소스 confound 없음** (모델이 '소스 정체'로 치팅 불가)

| 분할 | accuracy | macro-F1 |
|---|---|---|
| in-source A→A (imfds_motor) | 0.544 | 0.517 |
| in-source B→B (mechanical_bearing) | **1.000** | **1.000** |
| **cross-source A→B** | 0.357 | **0.175** |
| **cross-source B→A** | 0.333 | **0.167** |
| **in-source 평균** | — | **0.759** |
| **cross-source 평균** | — | **0.171** |
| **도메인 갭 (일반화 손실)** | — | **+0.588** |

### 해석 (논문 motivating result)
- **B→B는 100% 완벽**하지만 같은 모델이 **다른 소스에서는 ~0.17로 붕괴** (3클래스 랜덤 수준).
- 단일 소스 정확도(in-distribution)는 **허상**이다. 머신이 바뀌면 naive 모델은 전혀 일반화하지 못한다.
- 이것이 우리 파운데이션 모델이 풀어야 할 문제 — **하모나이제이션 + 도메인 불변 사전학습 + cross-source 평가**의 필요성을 실데이터로 실증.

### confound 함정 기록 (방법론적 교훈)
첫 시도(분산 샤드 무작위 8개)에서 각 소스가 1개 라벨에 거의 1:1 대응 →
모델이 "어느 소스인가"만 학습해 **test acc 100%**가 나왔으나 **이는 허상**.
소스×고장 행렬을 전수 조사([hf_explore.py](hf_explore.py))해 공유 라벨 쌍을 찾은 뒤에야
의미 있는 cross-source 측정이 가능했다. → cross-source 평가 설계의 중요성.

## GaugeAnything — Gauge-Bench v1: 크랙 분할 (감사 후 공식) ⭐

**데이터**: CrackSeg9k (9,159쌍, 14개 원본 소스). **프로토콜(2026-06-11 감사 반영)**:
crack-only mIoU(빈GT 제외)와 noncrack 클린율(탐지) 분리, 시드 3개 mean±std.
상세: [docs/progress/2026-06-11_audit-fixes.md](../docs/progress/2026-06-11_audit-fixes.md).

| 세그멘터 | 종류 | crack mIoU (±std) | noncrack 클린율 | 속도 |
|---|---|---|---|---|
| frangi | 고전(vesselness) | 0.115 ± 0.005 | 0.26 | 0.09s/img |
| adaptive | 고전(국소임계) | 0.181 ± 0.006 | 0.00 | <0.01s/img |
| **SAM3 zero-shot** | FM | **0.442 ± 0.011** | **0.68** | 0.36s/img |

배율: **2.44×** (감사 전 2.6×에서 정정 — noncrack 빈GT=1.0 규칙이 SAM3에 유리했음).
신규: SAM3는 탐지(거짓양성 회피)도 우위. 프롬프트 민감도: 직관적 변형엔 안정(±0.07)이나
**유효 동의어가 0으로 붕괴**(fracture/pit→0.0) → `segment_sam3_ensemble` 매핑 레이어로 대응.
참고 상한: 지도학습 U-Net류는 이 벤치에서 ~0.7+ — 본 주장은 promptability이지 SOTA 아님.

<details><summary>감사 전 수치 (보존용, deprecated)</summary>

| 세그멘터 | mIoU(혼합) | 소스편차 σ |
|---|---|---|
| frangi | 0.129 | 0.093 |
| adaptive | 0.172 | 0.093 |
| SAM3 | 0.450 | 0.177 |

</details>

### 판정 — 백본 정당화 + 특화 동기 동시 확보
1. **SAM3가 고전 최고 대비 2.6배** (0.450 vs 0.172). 백본으로 SAM3 채택이 실데이터로 정당화.
2. **그러나 SAM3도 0.45에 그침** — 완벽(0.9+)과 큰 격차 → 우리 fine-tune 기여 여지 충분.
3. **thin-structure 약점 실증**: SAM3 per-source가 `cracktree200=0.084`(가는 크랙에서 처참), `gaps384=0.337` vs `c=0.773`, `d=0.722`. σ=0.177로 큼.
   → VISION_DESIGN §3 예측대로 **SAM3는 thin 크랙에 약함**. M2 thin-structure fine-tune의 동기가 데이터로 확정.
4. 고전 기법은 σ도 크고 절대값도 낮음 — cross-source 일반화 실패가 비전에서도 재현 (센서 트랙과 동일 교훈).

### 측정 평가 (width 충실도) — "분할 ≠ 측정" 실증 ⭐⭐
GT 마스크에서 잰 폭을 참값으로, 예측 마스크 폭과 비교 (mm GT 부재 → 마스크폭 기준).

| 세그멘터 | mIoU | width MAE(px) | width 상대오차 | GT폭→예측폭 |
|---|---|---|---|---|
| adaptive | 0.188 | 6.67 | **43.5%** | 11.3→5.0px |
| frangi | 0.109 | 8.87 | 61.6% | 11.3→2.4px |
| **sam3** | **0.431** | **5.67** | 62.9% | 11.3→8.8px |

**발견 (논문 핵심 논거 — 측정을 1급 지표로 둬야 하는 이유):**
1. **분할과 측정은 다른 축**: SAM3가 mIoU·절대 width MAE는 1등이나, **width 상대오차는 adaptive(43.5%)<SAM3(62.9%)**. mIoU 우승이 측정 우승이 아니다.
2. **전 방법이 폭을 체계적 과소추정**: GT 11.3px 대비 SAM3 −22%, adaptive −56%, frangi −79%. 고전은 중심선만 잡음.
3. **'측정 준비된' 모델 부재**: 최선이 상대오차 43.5%. → GaugeAnything의 측정 인식 학습이 풀 빈자리.
4. (주의) length 상대오차는 마스크 파편화로 불안정(adaptive 1007%) → 현재 신뢰 지표는 **width**. length는 연결성 보존 후처리 후 재평가.
결과 JSON: `experiments/results/gauge_bench.json`, `gauge_bench_measure.json`

### 다음 (M2)
- **측정 인식 fine-tune**: SAM3의 폭 과소추정(−22%) 교정 — IoU 아닌 width MAE를 손실에 반영
- thin 크랙 특화: cracktree200=0.084 집중 → σ 축소 + mIoU↑
- 실제 크랙 사진 + ArUco → width(mm) end-to-end 데모 (계측 코어 selftest 통과 상태)

## 학습형 Soft 방법 (#1 DRAEM, #2 Matting) — regime별 학습 헤드 ⭐⭐

고전 PoC로 방향을 확정한 뒤, regime별 **학습형 헤드**를 자체 학습(license-clean)으로 검증.

### #1 DRAEM-lite (field regime, uneven) — 저주파 합성

**공식 수치 (2026-06-11 감사: val/test 분리 프로토콜, usable 97 → val 48/test 49)**:

| 방법 | val 선택 설정 | **test AUC (공식)** |
|---|---|---|
| SAM3 binary | — | ≈0.50 (참고) |
| **DRAEM-lite (학습)** | ensemble=T | **0.636** |
| 고전 조명잔차 | detrend=T, smooth=9 | **0.669** |

→ 분리 프로토콜에서도 순위 유지(감사 전 0.683/0.639와 근사 — 심한 test 오버핏은 아니었음을 검증).
좁은 결함엔 고전 조명모델이 강하고, 학습형은 합성-실분포 정합이 더 필요. 방향(연속>binary)은 확정.
체크포인트: `checkpoints/draem_uneven.pt`.

### #2 Matting 헤드 (fuzzy regime, fray) — 합성 검증 + **실 전이 실패** ⚠️

alpha GT 부재 → 합성 fuzzy 경계(matting 방정식) self-supervised. 체크포인트: `checkpoints/matte_fray.pt`.

| 검증 | binary/고전 | 학습 matting 헤드 |
|---|---|---|
| 합성 holdout α-MAE ↓ | 0.2007 | **0.0100 (20×)** |
| **실제 MT_Fray** 보존성 IoU(α≥0.5 vs GT) ↑ | **guided_matte 0.860** | **0.483** ⚠️ |

→ **정직한 부정적 결과**: 합성에서 20× 우위인 학습 헤드가 실제 fray에선 마스크를 훼손
(블롭 합성 분포 ≠ 방향성 텍스처 실분포 — 감사 B2 적중). **실전 fuzzy regime은 고전
guided_matte 채택 유지**, 학습 헤드는 방향성 실분포 합성 재설계 전까지 보류.
교훈: 합성 수치만으로 배포했다면 실전에서 마스크가 깨졌다 — 실데이터 검증의 가치.

→ **regime별 전략 (감사 후)**: field=고전 조명모델(0.669), fuzzy=고전 guided_matte(0.860),
학습 헤드 2종은 체크포인트로 보존하되 실분포 정합 후 재도전. 연속>binary 원칙은 양 regime에서 유지.

## Soft Inspection — 경계 없는/애매한 결함 (binary 실패 보완) ⭐⭐

**배경**: 멀티도메인에서 SAM3가 fray(0.03)·uneven(0.005)에 실패. 가설(문헌 검증): 결함의
*경계 성질*에 따라 도구가 달라야 함 — 선명→binary, 애매→matting, 장형→조명모델. ([SOFT_INSPECTION.md](../docs/SOFT_INSPECTION.md))
**지표**: 픽셀 ROC-AUC(연속맵 vs GT 마스크) — 임계 없이 "결함>정상" 랭킹. binary IoU≈0과 대조.

| 결함 | SAM3 binary AUC | soft AUC (조명잔차) | raw-gray | 평활화 영향 |
|---|---|---|---|---|
| **Uneven** (장형) | **0.499 (랜덤)** | 0.566 → **0.683** | 0.563 | ↑ 개선 (텍스처 detrend+평활) |
| **Fray** (애매경계) | 0.526 | **0.644** → 0.605 | 0.563 | ↓ 악화 (경계 파괴) |

### 발견 — 3-regime 분류가 실증됨
1. **SAM3 binary ≈ 0.50 = 완전 랜덤** (양쪽). binary segmentation이 경계없는/애매한 결함에 *근본적으로* 실패함을 확증 (프롬프트 문제 아님).
2. **연속 표현은 임계 없이 신호 복원** (AUC 0.68/0.64 > 0.50). binary IoU≈0인 곳에서.
3. **두 결함이 평활화에 정반대 반응** → 서로 다른 도구가 필요함을 직접 증명:
   - uneven은 *장으로 다룰수록* 좋아짐(컬럼 detrend+평활 → 0.57→0.68) = **조명/장 모델링**이 답.
   - fray는 평활화가 *(흐릿한) 경계를 파괴*해 나빠짐(0.64→0.61) = **matting(경계 보존)**이 답.
4. 단순 고전 다항잔차는 약한 시작점(0.68) — 진짜 lift는 텍스처/장 모델링(low-rank·basis-image·DRAEM 저주파합성) 또는 학습형 matting 필요. **방향은 확정, 다음은 적절한 학습 헤드.**
5. severity 연속 출력(ISO 25178): Uneven Sa=11.6 Sq=16.2 — 등급화 가능.

→ GaugeAnything 측정/표현을 **soft로 일반화**: regime별 라우팅(binary/matting/field) + soft 측정(Σα·sub-pixel·severity·±CI). 라이선스: 고전 PoC 자유, 학습형 matting 가중치는 Comp-1k 오염 주의.

## 멀티도메인 일반화 — "GaugeAnything"의 이름값 검증 ⭐⭐

**질문**: 크랙(thin) 너머 다양한 도메인·측정원시로 일반화하는가?
**설정**: 같은 파이프라인(SAM3 + 측정), 도메인별 프롬프트, GT 마스크 mIoU. 측정원시는 형태로 자동선택(thin↔blob).
**데이터**: CrackSeg9k(콘크리트) + Magnetic Tile(금속 표면 결함 5종, GT 마스크).

| 도메인 | prompt | mIoU | 측정원시(자동) | 측정값 |
|---|---|---|---|---|
| Concrete crack | "crack" | 0.367 | thin | width 8.0px |
| Mag-tile blowhole | "hole" | **0.429** | blob | dia 11.6px |
| Mag-tile crack | "crack" | **0.454** | thin | width 3.9px |
| Mag-tile break | "crack" | 0.030 | 혼합 | — |
| Mag-tile fray | "scratch" | 0.025 | 혼합 | — |
| Mag-tile uneven | "stain" | 0.005 | 혼합 | — |

### 발견 — 일반화 경계가 선명하다
1. **측정 코어는 도메인을 가로지른다**: blowhole→blob/직경(11.6px), 크랙→thin/폭. `classify_kind`가 형태로 측정원시를 자동 전환 — 콘크리트·금속 막론 동작.
2. **SAM3는 '구체 명사' 결함에 표면 막론 일반화**: 콘크리트 크랙(0.37) ≈ 금속 크랙(0.45) ≈ blowhole(0.43). zero-shot으로 도메인 전이.
3. **추상 텍스처 이상엔 실패**: break(0.03)·fray(0.03)·uneven(0.005)은 프롬프트 튜닝(scratch/stain)에도 회복 안 됨. → **zero-shot 개념분할의 경계** = 미세 텍스처 anomaly는 fine-tune/few-shot 필요.
4. (다음 데이터) count+직경의 대표 케이스인 **볼트/너트는 깔끔한 공개셋 부재** → 별도 확보 필요(NPU-BOLT 등).

→ "측정의 일반화"는 입증, "분할의 일반화"는 결함 추상도에 의존. 둘의 분리가 다음 fine-tune의 두 축.

## AUDIT — Leakage 검증 + 물리 베이스라인 (자가 리뷰 검증) ⭐

**목적**: 우리 자신의 결과에 대한 두 비판(W1 strawman, W2 leakage)을 실데이터로 검증.
**설정**: 동일 쌍(imfds_motor ↔ mechanical_bearing), CNN 시드 3개, envelope+LogReg 물리 베이스라인.
**결과 JSON**: `experiments/results/audit_baselines.json`

| 평가 | CNN (시드3) | Physics (envelope+LogReg) |
|---|---|---|
| in-source imfds [window-split] | 0.501±0.056 | **0.636** |
| in-source imfds [file-split] | **불가능** (라벨당 파일 1개) | 불가능 |
| in-source mech [window-split] | 0.994±0.009 | 0.995 |
| in-source mech [file-split] | 0.949±0.073 | **1.000** |
| cross imfds→mech | 0.175±0.000 | 0.175 |
| cross mech→imfds | 0.167±0.000 | **0.266** |

### 판정

**[A1 — Leakage]** 부분 확정, 구조적 문제가 더 심각:
- mechanical_bearing: window→file 전환 시 CNN -0.045 (0.994→0.949). leakage 존재하나 이 소스는 원래 쉬움(CWRU 계열, `B007_0.mat` 명명).
- **imfds_motor는 라벨당 녹음 파일이 1개 → 어떤 in-source 분할도 구조적으로 leakage.** 기존 A→A 0.517은 일반화 수치로 무의미. 이런 소스가 코퍼스에 더 있을 것 → **원본 재구축 필요성(W3) 확정**.

**[A2 — 물리 베이스라인]** 예상보다 강력한 발견:
- **in-source에서 물리가 CNN을 전반적으로 이김** (0.636 vs 0.501; file-split 1.000 vs 0.949).
- **cross-source에서는 물리조차 붕괴** (0.175 / 0.266). CNN의 cross-F1 0.167±0.000은 단일 클래스 예측으로의 완전 붕괴.
- → 도메인 갭은 "약한 베이스라인의 artifact"가 **아니다**. 50년 된 물리 특징도 rpm/order 정규화 없이는 전이 실패.
- → **다음 기술 우선순위 = order normalization(rpm 정규화) + 도메인 불변 학습.** 이것이 우리 FM의 존재 이유를 더 강하게 만든다.

### 수정된 헤드라인
도메인 갭 +0.588은 유지되되 해석이 강화됨:
"naive CNN뿐 아니라 **물리 기반 특징도** cross-source에서 붕괴한다 — speed-aware 하모나이제이션과 도메인 불변 사전학습이 필요하다."

## EXP — N-CMAPSS RUL 회귀 ✓

**데이터**: NASA N-CMAPSS DS02-006 (CC0, turbofan run-to-failure, HDF5). 263,173 샘플, 6 엔진, 14 센서.
**프로토콜**: **cross-unit 홀드아웃** (엔진 1개 통째 val 분리 = 진짜 일반화). RUL 상한 125 cycles.
**모델**: 1D-CNN RUL 회귀 (30K params), 15 epochs, GB10 GPU.

| 지표 | 값 |
|---|---|
| cross-unit val RMSE | **9.14 cycles** |
| NASA score | 2,247 |
| 학습 시간 | 1.5s (0.10s/epoch, GB10) |
| 참고 | N-CMAPSS SOTA RMSE ~5-7 (대형/장기학습). 경량 스모크로 9.14는 양호 |

→ 실데이터 RUL 회귀가 cross-unit 일반화 하에 동작 확인. GB10에서 263K 샘플 학습이 초 단위.

## 재현
```bash
# 환경 (DGX Spark)
bash setup_dgx_spark.sh && source .venv/bin/activate

# 코퍼스 구조 매핑
python experiments/hf_explore.py --shards 24

# cross-source 실험 (헤드라인)
python experiments/hf_cross_source.py --epochs 25

# RUL
python experiments/smoke_ncmapss.py --epochs 15
```

## 데이터셋 매핑 (검증된 HF 소스 — adyady 코퍼스 내)
| 코퍼스 내 소스 | 원본 | 클래스수 |
|---|---|---|
| cwru_style | CWRU | 2 |
| paderborn_bearing | Paderborn | 2 |
| ims_bearing | IMS | 2 |
| xjtu_sy | XJTU-SY | RUL |
| mafaulda | MaFaulDa | 2 |
| mechanical_bearing | (다중) | **4** |
| imfds_motor | IMFDS 모터 | 3 |
| vm_bearing | (다중) | 3 |
