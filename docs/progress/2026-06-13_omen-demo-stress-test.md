# 2026-06-13 — OMEN(RTX 5090) 데모 서버 + 데이터셋 stress 테스트

## 1. 무엇을

OMEN(RTX 5090, x86)에 GaugeAnything 데모 추론 서버를 Docker로 띄우고
(`--restart unless-stopped`), 우리 데이터셋 14장(크랙·비크랙·rebar·코인·MT·T-LESS)으로
정확성·지연·동시성·edge case를 stress 테스트했다. 서버: FastAPI(`serve/app.py`),
SAM 3 backbone + metrology core 인프로세스.

## 2. 환경/배포 (검증됨)

- 이미지 `gaugeanything:latest` 6.08GB, torch 2.12.0+cu130, transformers 5.12
- GPU 인컨테이너 인식 OK, 체크포인트 7종 호스트 마운트 로드 OK
- 마운트: `checkpoints/`(ro), `~/.cache/huggingface`(SAM3 토큰+가중치 → 재다운로드 0)
- 외부 노출: 추후 tailscale (현재 localhost:8000)

## 3. Stress 테스트 결과

### 지연 (warm, RTX 5090)

| 엔드포인트 | p50 | p90 | p99 | 비고 |
|---|---:|---:|---:|---|
| `/inspect` (SAM3+측정) | 90ms | 90ms | 90ms | 매우 안정 |
| `/count_rebar` | 239ms | — | 240ms | 고해상 density |

### 동시성·자원

- 8-way 병렬 inspect × 24 req: **24 ok / 0 fail**, 10.4 req/s
- GPU 메모리: 3.2GB / 32GB (10%) — 배칭 여유 충분, util idle 0%
- 컨테이너 RAM 2.1GB

### 정확성 sweep

| 트랙 | 결과 | 판정 |
|---|---|---|
| 크랙(Rissbilder) | 1~2 인스턴스, width 3.9~10.4px | ✓ 정상 |
| 비크랙(concrete wall) | 0 인스턴스 | ✓ 정상 기각 |
| 코인 | 3 인스턴스 | ✓ |
| T-LESS("plastic part") | 4 인스턴스 | ✓ |
| MT blowhole("hole") | 0 인스턴스 | ✗ 단일 프롬프트 miss (앙상블 필요) |

### Rebar 카운트 (density head, GT 대비)

| 이미지 | GT | pred | err | 영역 |
|---|---:|---:|---:|---|
| 001563 | 61 | 60.8 | **0.2** | mid-dense ✓ 우수 |
| 001255 | 81 | 44.0 | 37.0 | dense undercount (문서화됨) |
| 000293 | 1 | 10.4 | 9.4 | **sparse overcount (신규 발견)** |

### Edge cases (전부 graceful, 크래시 0)

8×8/4000×3000/grayscale/solid-white → 0 inst 정상. 거대 이미지 0.34s.
프롬프트 `fracture`→0 (synonym collapse 라이브 재확인), `asdfqwer`→0.

## 4. 발견

1. **서버는 production-stable**: 90ms warm, 10.4 req/s 동시성, 0 실패, edge-case 안전.
   RTX 5090에서 SAM3 warm 추론이 매우 빠르고 일관적.
2. **신규 발견 — rebar density head의 sparse overcount**: GT=1을 10.4로 과대예측.
   density head가 학습 분포 평균(~21)으로 회귀하는 floor가 있다. dense undercount와
   합쳐, 단일 head는 양 극단(sparse·dense)에서 모두 bias. Count v2에서 함께 잡아야 함
   (mid-range는 61→60.8로 이미 우수).
3. 모델의 알려진 특성(synonym collapse, MT 단일프롬프트 miss)이 라이브에서도 재현 —
   서버 버그가 아니라 backbone/prompt 특성.

## 5. 학습 완료 가중치 현황 (재확인)

Spark `checkpoints/` 9종 전부 학습 완료, OMEN 컨테이너에서 로드 검증:

| 가중치 | 역할 | 핵심 수치 | HF |
|---|---|---|---|
| `profile_width_cnn.pt` | 1D 크랙 폭 회귀 | table 18.6μm | ✓ |
| `gaugehead_tiny_width.pkl` | 폭 specialist (ExtraTrees) | rel.err 0.4724 | ✓ |
| `gaugehead_tiny_width_conformal.pkl` | + conformal interval | per-source cov 0.91/1.00/0.95 | ✓ |
| `rebar_density_head.pt` | Count v1 density | MAE 7.0 (stratified) | ✓ |
| `rebar_density_head_v11.pt` | Count v1.1 (dense 개선) | dense bias −29→−19 | ✗ (실험 변형, 미배포) |
| `matte_fray_directional.pt` | fray 매팅 v2 | real IoU 0.949 | ✓ |
| `matte_fray.pt` | fray v1 (정직 음성) | real 0.483 | ✓ |
| `draem_uneven.pt` | uneven/mura DRAEM-lite | test AUC 0.636 | ✓ |
| `m2_refiner.pt` | M2 refiner (superseded) | 0.564 | ✓ |

배포 서버는 `rebar_density_head.pt`(v1, MAE 7.0)를 사용. v11은 dense 개선 실험 변형으로
Spark에만 보관.

## 6. 재현

```bash
# OMEN
cd /home/hwoo-joo/github/GaugeAnything
.venv/bin/python serve/stress_test.py --img-dir /home/hwoo-joo/ga_stress
```
