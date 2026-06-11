# 2026-06-11 — E-cnt-2 rebar SAHI-style tiled SAM3

## 배경
E-cnt-1에서 ROI-1555 rebar count는 zero-shot SAM3가 크게 실패했다. best prompt `metal rod`도
MAE 13.2, acc@10% 0%였다. 하지만 이 실패가 "개념을 모름"인지, "전역 이미지에서 작은/밀집 객체를
놓침"인지 분리되지 않았다.

## 실험
- 데이터: ROI-1555, E-cnt-1과 같은 deterministic sample n=20
- GT: labelme polygon instance count
- prompt: `metal rod`
- 방법: SAHI-style tiled SAM3
  - tile 640px
  - overlap 0.25
  - threshold 0.35 (0.40도 먼저 테스트)
  - no training, tile masks mapped back to full image, IoU/center dedup
- 스크립트: `experiments/rebar_sahi_eval.py`
- 결과: `experiments/results/rebar_sahi_eval.json`

## 결과

| Method | n | MAE ↓ | rel.err mean ↓ | acc@10% ↑ | exact ↑ | pred mean |
|---|---:|---:|---:|---:|---:|---:|
| Global SAM3 `metal rod` (E-cnt-1) | 20 | 13.20 | 80.2% | 0% | 0% | — |
| SAHI SAM3, threshold 0.40 | 20 | 8.35 | 53.9% | 15% | 10% | 13.6 |
| **SAHI SAM3, threshold 0.35** | 20 | **7.35** | **52.9%** | **20%** | 5% | 15.0 |

대표 고밀도 실패:
- GT 61 → pred 30
- GT 81 → pred 40
- GT 48 → pred 28

## 판정
1. **타일링은 효과가 있다**: MAE 13.2→7.35, acc@10% 0→20%. E-cnt-1 실패는 일부 scale/crowding 문제였다.
2. **그러나 counting은 아직 해결되지 않았다**: 고밀도 장면에서 30-40개 undercount가 남는다.
3. **다음 모델 방향은 density/centroid fallback**: SAHI는 instance proposal recall을 올리지만,
   맞닿은 rebar ends를 개별 인스턴스로 충분히 분리하지 못한다.
4. **논문 표현**: "zero-shot global counting fails; tiled inference partially recovers but remains far from supervised rebar counters."

## 다음
- density map 또는 centroid heatmap head를 ROI-1555 masks에서 학습
- density bin별 error 분석 (sparse/medium/dense)
- SAHI tile size/threshold sweep은 보조. 근본 해결은 touching-instance 분리/밀도 회귀.
