# GaugeAnything — Dataset Registry

GaugeAnything(비전 계측)이 실제 사용하는 데이터셋의 운영 인덱스. 각 데이터셋이 어느 모델·
실험·GaugeBench 트랙에 쓰이는지, 어디에 저장돼 있는지, 라이선스를 한곳에 연결한다.

- 학습/실험 원본: Spark `/home/hwoo_joo/github/GaugeAnything/datasets/`
- 라이선스·다운로드 상세: [paper/DATASETS.md](paper/DATASETS.md), [data/DATA_ACQUISITION.md](data/DATA_ACQUISITION.md)
- 다운로드 스크립트: [data/scripts/](data/scripts/)

## 사용 중 (실험 완료)

| 데이터셋 | Spark 크기 | 물리량 | 쓰는 곳 (모델/실험) | GaugeBench | 라이선스 |
|---|---:|---|---|:--:|---|
| **CrackSeg9k** | 8.3G | 크랙 분할·폭 | gauge_bench(S), M2 cache→[GaugeHead-Tiny](MODELS.md) | S, W | CC0 (일부 NC) |
| **krkCMd** | 2.1G | 크랙 폭 μm | [Profile-Width CNN](MODELS.md), e2e 멀티인스턴스 | P | CC BY 4.0 |
| **T-LESS** | 1.8G | 부품 치수 mm | tless_upper_bound, tless_sam3 | D | CC BY 4.0 |
| **ROI-1555 rebar** | 579M | 카운트 | [Rebar Density v1](MODELS.md), SAHI | (C 예정) | 미명시(연구) |
| **coins (kaa)** | 455M | known-object mm | coins_mm_eval | K | MIT |
| **SmartDoc15** | 973M | 문서 스케일 | smartdoc_scale, detected_quad(P2-1b) | — | 공개 |
| **Magnetic-Tile** | 109M | fray/uneven/hole | [Matting v2·DRAEM](MODELS.md), 멀티도메인 | — | 미명시(연구) |
| **VT suite (LCW)** | 1.8G | 크랙 보조 | 도메인 확장 | — | CC0 |
| **TUM RGB-D** | 1.7G | 동적 핸드헬드 | tum_dynamic_eval | (Y 예정) | CC BY |
| **ADT ATEK** | 294M+54M | 동적 객체 치수 | adt_atek_* (oracle gate) | (Y 예정) | 연구 |
| **m2_cache** | 77M | 폭 feature 캐시 | GaugeHead-Tiny train/val/test | W | (파생) |

GaugeBench 트랙: S=분할 · W=폭 ladder · P=물리 μm · D=CAD mm · K=known-object ·
C=카운팅(예정) · Y=동적(예정). 트랙 정의: [benchmark/](benchmark/README.md).

## 후보/미사용 (조사 완료)

| 데이터셋 | 잠재 역할 | 비고 |
|---|---|---|
| MIDV-500 | ID 카드 스케일 (P2-1b 2nd family) | 다음 adapter |
| TimberSeg / DeepFish | 통나무/물고기 카운트·길이 | coverage 확장 후보 |
| HB / YCB-V (BOP) | 부품 치수 확장 | D 트랙 확장 |
| corrosion_cs_gh (294M→9.4M) | 부식 면적 | 후보 |

## ⚠️ 레거시 센서 트랙 (GaugeAnything 범위 밖)

Spark에 함께 있으나 **현재 비전 프로젝트와 무관**한 모 프로젝트(Industrial Anything 진동/PHM)
데이터: `ncmapss`(32G), `kaist`(4G), `ims`(3G), `adyady`(5.2G). 평가 방법론(leakage 교훈)만
승계하고 비전 트랙에는 쓰지 않는다. 디스크 정리 시 참고.

## 저장·캐싱 규약

1. 원본은 Spark `datasets/`에만 (로컬 git·OMEN 미포함, `.gitignore`).
2. 파생 feature 캐시(m2_cache 등)는 재생성 가능 → git 미포함.
3. 데모(OMEN)는 데이터셋 불필요 — 추론 입력은 업로드 이미지.
4. 새 데이터셋 사용 시 이 표 + paper/DATASETS.md에 라이선스 확인 후 등재.
