# GaugeAnything — demo inference server

RTX 5090(OMEN)에서 SAM 3 backbone + GaugeAnything metrology core를 인프로세스로 로드해
promptable quantitative inspection을 HTTP로 서빙한다.

> **왜 FastAPI인가** (ollama/vLLM 아님): ollama·vLLM은 **LLM 전용** 서빙 엔진이다.
> GaugeAnything의 모델군은 LLM이 아니라 SAM 3(세그멘테이션 트랜스포머) + 작은 CV 측정
> 헤드(profile_width_cnn, gaugehead_tiny, rebar_density_head …)다. 이들을 한 프로세스에서
> 로드하고 측정 파이프라인을 조립하는 가장 reliable한 방법이 FastAPI/uvicorn 직접 서빙이다.
> production 확장이 필요하면 NVIDIA Triton(Python backend)으로 승격 가능.

## 엔드포인트

| method | path | 설명 |
|---|---|---|
| GET | `/` | 인터랙티브 데모 UI (이미지 업로드 + 프롬프트 → 측정 오버레이) |
| GET | `/health` | 모델/GPU/체크포인트 상태 |
| POST | `/inspect` | `image`(multipart) + `prompt`,`segmenter`,`marker_size_mm`,`manual_mm_per_px` → atoms + summary + overlay(base64) |
| POST | `/count_rebar` | `image` → rebar density head 카운트 + 히트맵 오버레이 |

## 실행 (OMEN)

```bash
ssh hwoo-joo-OMEN
cd /home/hwoo-joo/github/GaugeAnything
.venv/bin/python -m uvicorn serve.app:app --host 0.0.0.0 --port 8000
# 또는 bin/serve (아래)
```

첫 `/inspect` 호출은 SAM 3 가중치 다운로드+로딩으로 ~40s. 이후 warm 추론은 RTX 5090에서
**~0.2s/이미지**.

## 환경 (검증됨)

- GPU: RTX 5090 (32GB, sm_120 Blackwell), 드라이버 590
- torch 2.12.0+cu130, transformers 5.12 (Sam3Model/Sam3Processor)
- venv: `uv venv --python 3.12 .venv` (system python3-venv 부재 → uv 사용)
- 가중치: HF `James-joobs/GaugeAnything`(공개) 7종 + `facebook/sam3`(gated, 토큰 필요)

설치 재현:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r serve/requirements.txt
# 우리 가중치 다운로드
.venv/bin/python serve/fetch_weights.py
```

## 검증 결과 (2026-06-13, OMEN)

| 항목 | 값 |
|---|---|
| `/health` | ok, RTX 5090, torch 2.12.0+cu130 |
| `/inspect` cold (SAM3 로딩 포함) | 42.9s |
| `/inspect` warm (크랙 4 인스턴스 + 측정 + 오버레이) | 0.196s |
| `/count_rebar` | 카운트 + JET 히트맵 정상 |

## 주의

- `marker_size_mm` 또는 `manual_mm_per_px` 없으면 출력 단위는 px (스케일 미해석).
  실제 mm 측정에는 ArUco 마커(20mm 등) 또는 known-object 스케일이 필요.
- rebar count는 Count v1 density head(stratified MAE 7.0) — 밀집(GT≥40) 영역은 undercount.
- SAM 3는 gated: OMEN에 `~/.cache/huggingface/token`이 있고 facebook/sam3 라이선스 동의 완료.
