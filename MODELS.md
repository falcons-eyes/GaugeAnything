# GaugeAnything — Model Registry

학습된 모든 task head의 단일 진실 소스. 버전·지표·배포 상태·로드 방법을 한곳에 모은다.
foundation backbone(SAM 3)은 학습하지 않고 frozen으로 사용한다.

> 철학: 새 foundation model을 주장하지 않는다. 각 head는 **measurement specialist**이며,
> 모든 학습 head는 source-held-out에서 strongest simple baseline을 이겨야 등재된다.
> 로드맵: [docs/MODEL_RESEARCH_ROADMAP.md](docs/MODEL_RESEARCH_ROADMAP.md).

## 배포·캐싱 위치

| 위치 | 용도 | 경로 |
|---|---|---|
| **HuggingFace** (배포) | 공개 가중치 레지스트리 | [James-joobs/GaugeAnything](https://huggingface.co/James-joobs/GaugeAnything) |
| Spark (GB10) | 학습·실험 원본 | `/home/hwoo_joo/github/GaugeAnything/checkpoints/` |
| OMEN (RTX 5090) | 데모 서빙 (HF에서 fetch) | `checkpoints/` ([serve/](serve/README.md)) |
| 로컬 git | 가중치 미포함 (`.gitignore`) | HF/Spark에서 받음 |

가중치 받기: `python serve/fetch_weights.py` (HF 공개 7종 → `checkpoints/`).

## Task Heads

| 모델 | 파일 | 크기 | 역할 | 공식 지표 (held-out) | 상태 | HF |
|---|---|---:|---|---|---|:--:|
| **GaugeHead-Tiny** | `gaugehead_tiny_width.pkl` | 1.1M | 크랙 폭 specialist (ExtraTrees, 19 feat) | rel.err **0.4724** vs quantile 0.480 | ✅ 배포 | ✓ |
| **GaugeHead-Tiny + conformal** | `gaugehead_tiny_width_conformal.pkl` | 1.5M | + 90% conformal interval | per-source cov **0.91/1.00/0.95** | ✅ 배포 | ✓ |
| **Profile-Width CNN** | `profile_width_cnn.pt` | 0.7M | 1D 신호 폭 회귀 (501px profile→μm) | table MAE **≈18.6μm** (DLM 11.1) | ✅ 배포 | ✓ |
| **Rebar Density v1** | `rebar_density_head.pt` | 0.46M | rebar 카운트 (density map) | MAE **7.0** (stratified, SAHI 8.9 격파) | ✅ 배포 | ✓ |
| Rebar Density v1.1 | `rebar_density_head_v11.pt` | 0.46M | dense-regime 개선 변형 | dense bias −29→**−19** (overall 7.8) | 🧪 실험 | ✗ Spark만 |
| **Fray Matting v2** | `matte_fray_directional.pt` | 7.8M | fuzzy 경계 α 매팅 (방향성 합성) | real-fray IoU **0.949** vs 0.860 | ✅ 배포 | ✓ |
| Fray Matting v1 | `matte_fray.pt` | 7.8M | v1 (정직 음성: 합성→실전 전이 실패) | real IoU 0.483 | 📕 음성 보존 | ✓ |
| **DRAEM-uneven** | `draem_uneven.pt` | 15.5M | boundaryless field(mura) 이상 | test AUC 0.636 (고전 0.669) | ⚠️ 연구 | ✓ |
| M2 Refiner | `m2_refiner.pt` | 7.8M | 크랙 마스크 refiner (UNet 1.9M) | rel.err 0.564 (superseded) | 📕 superseded | ✓ |

상태 범례: ✅ 배포 = 데모/추론 사용 · 🧪 실험 변형 · 📕 reproducibility/음성 보존 ·
⚠️ 연구(고전 baseline 미달, 정직 표기).

## 라이선스 주의

- 모든 head: 코드 Apache-2.0. 단 일부는 라이선스 미명시 데이터로 학습 → **연구용**.
- `profile_width_cnn`: krkCMd **CC BY 4.0** 학습 → 상업 OK.
- 학습 데이터·라이선스 전수: [paper/DATASETS.md](paper/DATASETS.md).

## 버전 관리 규약

1. 새 학습 head는 이 표에 행 추가 + `experiments/results/<name>.json`에 공식 지표 핀.
2. 배포 승격은 source-held-out에서 strongest simple baseline 격파가 조건.
3. 변형(v1.1 등)은 실험 상태로 Spark 보관, 배포 head만 HF 업로드.
4. GaugeBench 트랙 지표는 [benchmark/](benchmark/README.md)의 release gate로 핀 검증.

## 백본 (학습 안 함)

| backbone | 용도 | 출처 |
|---|---|---|
| SAM 3 | promptable 세그멘테이션 (WHERE) | facebook/sam3 (gated, 별도 라이선스) |
