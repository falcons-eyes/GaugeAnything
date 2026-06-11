# GaugeAnything — 산업 마이크로 비전 파운데이션 모델 설계서

> 프로젝트 재정의 (2026-06). 진동/센서 PHM 축에서 **비전 중심 산업 마이크로 비전**으로 무게중심 이동.
> 미션: *거리, 미세한 차이, 파이프, 볼트/너트, 벽면 크랙* — 탐지하기 어려운 물리적 공통 문제를
> 비전으로 일반화하는 promptable 모델. "무엇이 있는가"를 넘어 **"몇 mm인가, 몇 개인가, 어느 등급인가"**.

---

## 0. 한 줄 정의

**GaugeAnything**: 이미지 + 프롬프트(명사구/예시/포인트) → **인스턴스 + 개수 + 미터법 치수(mm±오차) + 상태 등급**을
원자 단위로 출력하는 promptable 정량 검사(Quantitative Inspection) 모델.

기존 Anything 패밀리가 *지각*(어디에·무엇이)에 머물 때, 우리는 **계량(metrology)** 을 1급 출력으로 만든다.

---

## 1. 이름 결정 (충돌 조사 완료)

| 후보 | 판정 | 근거 |
|---|---|---|
| MeasureAnything | ❌ **사용 불가** | arXiv 2412.03472 (2024.12, UCLA) — SAM2+스테레오 미터법 측정. 이름+컨셉 정면 충돌 |
| InspectAnything | ⚠️ 제품명 불가 | Snappii "Inspect Anything" 상용 검사 앱 (동일 카테고리 = 상표 혼동 최대). 논문 제목으로는 빈자리 |
| **GaugeAnything** | ✅ **추천** | 논문·repo·제품 충돌 없음 검증. 동사 *gauge* = "측정하다 + 판정하다" — locate+measure+assess를 한 단어로 |
| Industrial Anything | △ 프로젝트 우산명 | 충돌 없으나 행위 동사 부재. 상위 프로젝트/브랜드명으로 유지 |

> 채택: **프로젝트 = Industrial Anything, 모델 = GaugeAnything, 제품 = FalconEyes**.
> 주의: "gauge"의 계기판-판독 연상은 부수 기능(아날로그 게이지 읽기)으로 흡수 가능 — 약점이 아니라 기능.

---

## 2. 6개 레퍼런스 해부 — 가져올 것과 버릴 것

| 레퍼런스 | 라이선스 | 가져올 것 | 우리에게 없는 것 |
|---|---|---|---|
| **SAM 3** (Meta, 2025.11) | ✅ SAM License (**상업 OK, fine-tune 공식 지원**) | **중심 백본**. 개념 프롬프트(명사구/예시)→모든 인스턴스 → 카운팅 원시. PCS+PVS 통합 | thin-structure 약함, 밀집 소형 객체 약함(DETR 쿼리 슬롯 한계), 미세차이 판별 약함, **미터법 없음** |
| **Grounded-SAM** (IDEA) | ✅ Apache (open 경로) | **합성 계약**: `text → scored boxes → masks` 인터페이스 표준. 모든 부품 교체 가능 구조 | 박스 프롬프트는 크랙(thin)에 구조적 부적합 — 박스가 거의 배경 |
| **Depth Anything V2** | ⚠️ **Small만 Apache** (B/L/G는 CC-BY-NC) | teacher→pseudo-label 레시피(우리 distill 단계), 상대 깊이 prior | metric 변형은 0-20m/0-80m 실내외용 — **<1m 마크로 영역 보정 없음**, sub-mm 불가능 |
| **LocateAnything** (NVIDIA) | ❌ 비상업 + Qwen Research | **패턴만**: Parallel Box Decoding → 우리 Parallel Inspection Decoding, 데이터 믹스 설계 | 가중치 사용 불가. 마스크 없음(박스/포인트만) |
| **Count-Anything** (ylqi) | ❌ 라이선스 부재, 2023 방치 | 패턴만(segment-everything→분류→집계). **SAM 3가 완전 대체** | FSC-147에서 전문 카운터에 MAE 10+ 뒤짐 |
| **MeshAnything** V1/V2 | ❌ S-Lab 비상업 | 사실상 없음 — unit-box 정규화가 절대 스케일 파괴, 1600면 한계로 크랙/나사산 표현 불가 | 검측 계측에 부적합. 시각화용 저폴리 트윈만 |

**상업 클린 합성 스택**: SAM 3 + Grounded-SAM open 경로(Apache) + DA-V2 Small(Apache) — 오늘 바로 조립 가능.

---

## 3. 갭 = 우리 자리 (검증된 부재, 2026-06)

선행연구 전수조사 결과, 개별 능력은 다 존재하나 **다음 6개는 아무도 안 함**:

1. **근거리 미터법 스케일 해석** — 픽셀→mm 변환을 하는 파운데이션 모델이 없다. DA-V2 metric은 방/거리 스케일, sub-mm 크랙 폭은 모노큘러 깊이 정확도 너머. → **스케일 리졸버**(레퍼런스 마커/레이저 도트/스테레오/폰 LiDAR/기지 치수 객체) 필요
2. **Thin-structure 분할** — SAM 계열 박스/쿼리 프롬프트는 크랙·용접심·와이어에 구조적으로 약함 → SAM 3 fine-tune + thin 전용 디코더 (공식 지원됨)
3. **밀집 소형 카운팅** — SAM 3 쿼리 슬롯 한계 → SAHI 타일링 + 밀도 회귀 폴백
4. **미세차이/이상 판별** — 거의 동일한 부품 구별(볼트 풀림/장착 누락)은 SAM 3 명시적 약점
5. **반사 금속 강건성** — 모노큘러 깊이·분할 공통 난제, 타겟팅한 모델 없음
6. **측정의 1급 출력화** — 모두 박스/마스크/깊이만 출력. "폭 0.42mm ± 0.05"를 내는 모델 없음

**가장 가까운 경쟁자 3**: Measure Anything(UCLA, 봉형 단면+스테레오+농업 한정), SAM 3(미터법·도메인 튜닝 없음), PaveGPT(포장 단일 수직, 미터법 미접지). **통합 + 미터법 접지가 우리 주장.**

> ⚠️ 냉정한 리스크: SAM3+DepthPro+측정 헤드는 어느 랩이든 글루코드로 조립 가능.
> **모트는 아키텍처가 아니라 (a) 근거리 미터법 정확도, (b) thin-structure 정확도, (c) 산업 데이터 큐레이션 + 현장 메트릭 GT 자체 수집.**

---

## 4. 태스크 정식화 — Promptable Quantitative Inspection

```
입력:  이미지(+선택: 깊이/스케일 레퍼런스/비디오) + 프롬프트
       프롬프트 = 명사구("crack" / "hex bolt") | 예시 박스 | 포인트 | (오케스트레이터 경유) 자연어 질의
출력:  원자 검사 단위(Inspection Atom)의 병렬 집합:
       { mask, class, count_id, metrics{width_mm, length_mm, spacing_mm, dia_mm, ±σ},
         condition(등급), confidence }
```

다운스트림 흡수 (재학습 없이 프롬프트로):
| 다운스트림 | 프롬프트 | 출력 사용 |
|---|---|---|
| 크랙 탐지+폭 측정 | "crack" | mask 골격→폭 프로파일→mm + 코드 임계 비교 |
| 볼트/너트 검사 | "hex bolt" + 예시 | 인스턴스+개수+누락 탐지(기대 패턴 대비) |
| 부품 카운팅 | 예시 박스 1-3개 | 인스턴스 수 (밀집 시 밀도 폴백) |
| 파이프/배관 | "pipe" | 마스크+주행 방향+직경 추정 |
| 부식 등급 | "corrosion" | mask + condition (AASHTO 4등급) |
| 거리/간격 | 포인트 2개 | 미터법 거리 |

LocateAnything PBD 차용 → **Parallel Inspection Decoding**: Inspection Atom을 고정 길이 원자로 병렬 디코딩.

---

## 5. 아키텍처 — 합성 → 특화 → 증류 (검증된 3단)

```
[Stage 0 — Compose (0~6주): GaugeAnything v0, 학습 제로]
  텍스트/예시 ─► SAM 3 (PCS) ─► 전체 인스턴스 마스크+점수 ─► COUNT
                                        │ (밀집 시 SAHI 타일링)
  이미지 ─► DA-V2 Small (상대깊이) ──────┤
  스케일 입력 ─► 스케일 리졸버 ◄─────────┘
   (ChArUco 마커 | 폰 LiDAR/ARKit | 기지 치수 객체[볼트머리 규격!] | 레이저 도트)
                  │
                  ▼
  마스크 기하 모듈: medial-axis 골격 → 폭 프로파일/길이/직경/간격 → mm ± σ
                  │
  (선택) VLM 오케스트레이터: 복합 질의("플랜지 왼쪽의 부식된 볼트") → 명사구 분해 → SAM 3
  → 스마트폰 현장 데모: "이 크랙 폭 몇 mm?" "볼트 몇 개? 누락은?"

[Stage 1 — Specialize (6~14주)]
  • SAM 3 fine-tune (공식 학습 코드): thin-structure 디코더 — CrackSeg9k+LCW로
  • 카운팅 밀도 폴백 헤드 (Roboflow rebar/철근 셋)
  • 상태 등급 헤드 (Corrosion CS 4등급)
  • 스케일 리졸버 학습 보정: 자체 수집 caliper-GT (아래 §6)

[Stage 2 — Distill & Unify (14주~)]
  • Depth Anything 레시피: Stage 0/1 합성계를 teacher로 → 미라벨 산업 이미지 대량 pseudo-label
  • 단일 모델로 증류 + Parallel Inspection Decoding 헤드
  • 모델 사다리: 엣지(스마트폰)/서버
```

---

## 6. 데이터 전략

### 6.1 공개 데이터 (라이선스 검증 완료)

**Tier A — 상업 클린 (~15만 장, 즉시 학습 가능)**
| 데이터셋 | 내용 | 라이선스 |
|---|---|---|
| **CrackSeg9k** | 크랙 분할 9.3k (10개 셋 통합) | CC0 ⚠️ DeepCrack/GAPs 서브셋은 원본 NC → 제외(파일 접두사로 식별) |
| **VT Bianchi 스위트** (LCW+Corrosion CS+COCO-Bridge) | 실제 교량검사: 크랙 분할 3.8k + 부식 4등급 분할 440 + 구조 디테일 검출 1.5k | **CC0** — 저평가된 보물 |
| SDNET2018 / METU 40k | 크랙 분류 56k/40k (볼륨) | CC BY |
| DAGM 2007 | 표면결함 텍스처 11.5k | CC BY |
| NPU-BOLT + Roboflow CC-BY 셋 | 볼트 검출 + 철근/나사 카운팅 | CC0/CC BY |
| RIAWELC | 용접 X-ray 분류 24.4k | 자유 배포(라이선스 파일 없음 — 저자 확인 권장) |

**Tier B — NC (평가 전용 또는 라이선스 협상)**
- **dacl10k** (9.9k, 교량 손상 19클래스 분할 — 우리 태스크에 최적 적합, CC BY-NC) → 평가 + 저자(dacl.ai) 협상 가치
- Sewer-ML (1.3M, 파이프 — 유일한 스케일, NC)

### 6.2 메트릭 GT는 세상에 없다 → 자체 수집이 모트 ⭐

전수조사 결과 **크랙 폭(mm)·볼트 치수·근거리 측정 GT를 가진 공개 데이터셋은 부재**.
→ FalconEyes Phase 2(현장 실측)가 정확히 이 빈자리를 채운다:
- 스마트폰 + **ChArUco 보드/캘리퍼/크랙 게이지** 동시 촬영 프로토콜
- 볼트머리 규격(M8=13mm 등)은 그 자체가 무료 스케일 레퍼런스
- 수백 장이면 스케일 리졸버 보정 + 평가셋으로 충분 — **아무도 없는 데이터 = 논문/제품 모트**

---

## 7. 벤치마크 규율 이어받기 (센서 축에서 배운 것)

진동 실험에서 실증한 함정들이 비전에도 그대로 적용된다:
- **Cross-dataset 전이가 헤드라인**: CrackSeg9k 학습 → LCW/dacl10k 평가 (소스 confound·leakage 회피 분할)
- **물리/고전 베이스라인 필수**: 크랙 폭은 고전 영상처리(이진화+골격화) 베이스라인과 비교
- **수치 3종 보고**: 분할 IoU + **메트릭 오차(mm MAE)** + 카운팅 MAE — 측정이 1급 지표
- 시드 ≥3, 결과 JSON, file-level 분할

→ **Gauge-Bench**: cross-domain 검사 벤치마크 (분할+측정+카운팅 통합) = 첫 공개 아티팩트 후보.

---

## 8. 기존 센서 축과의 관계

| 자산 | 처리 |
|---|---|
| 진동 cross-source 벤치마크 + audit (도메인 갭 +0.588, leakage 교훈) | **유지** — 평가 방법론이 비전으로 직수입됨. 별도 트랙으로 보존 |
| 하모나이제이션 파이프라인 (src/harmonization) | 보존 — 장기 멀티모달(열화상+진동+비전) 융합 시 재사용 |
| GB10/uv/torch 인프라 | 그대로 사용 — SAM 3 fine-tune·증류에 충분 (848M, fine-tune은 LoRA/부분) |
| 장기 비전: Physical World Foundation Model | 불변 — GaugeAnything(비전) → +열화상 → +진동/음향 순서로 합류 |

---

## 9. 로드맵

| 단계 | 기간 | 산출물 | 성공 기준 |
|---|---|---|---|
| **M0 합성 데모** | ~6주 | GaugeAnything v0 (SAM3+DAv2-S+스케일리졸버+기하모듈), 스마트폰 데모 | 크랙 폭 mm 출력 end-to-end 동작, ChArUco 기준 오차 측정 |
| **M1 데이터+벤치** | 병행 | Tier A 수집 + Gauge-Bench v0 (cross-dataset 분할) | 분할 IoU/mm MAE/카운트 MAE 베이스라인표 |
| **M2 특화** | +8주 | SAM3 thin-structure fine-tune + 밀도 카운팅 + 등급 헤드 | crack IoU > SAM3 zero-shot +10pt, 폭 MAE < 고전 베이스라인 |
| **M3 현장 GT** | 병행 | ChArUco/캘리퍼 자체 수집 프로토콜 + 수백 장 | 스케일 리졸버 보정 완료 — 모트 데이터 확보 |
| **M4 증류·공개** | +8주 | 단일 GaugeAnything + 논문 + HF/GitHub/프로젝트 페이지 | LocateAnything식 공개 |

### 즉시 다음 단계 (우선순위)
1. **M0 합성 파이프라인 구축** — SAM 3 가중치 확보(HF gated) + DA-V2-S + 마스크 기하 모듈(`medial-axis 폭 프로파일`) 코드
2. **CrackSeg9k + VT 스위트 다운로드** (CC0 — 즉시)
3. **스케일 리졸버 v0** — ChArUco 검출 + 픽셀→mm 변환 유틸
4. Gauge-Bench 분할 설계 (센서 축 audit 교훈 적용)

---

## 부록 — 핵심 레퍼런스
- SAM 3: https://github.com/facebookresearch/sam3 (SAM License, 상업 OK, fine-tune 지원)
- Grounded-SAM(-2): https://github.com/IDEA-Research/Grounded-Segment-Anything (Apache 경로)
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2 (**Small만 Apache**)
- Measure Anything (충돌 선행연구 — 차별화 대상): arXiv 2412.03472
- PaveGPT (assess 축 선행): arXiv 2604.08212 · SAA+/SAID/AnomalyGPT (defect 축)
- 데이터: CrackSeg9k(Dataverse CC0) · VT Bianchi 스위트(figshare CC0) · dacl10k(NC, 평가) · NPU-BOLT(CC0)
