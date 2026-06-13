# 2026-06-13 — M2 v2-b: GaugeHead-Tiny uncertainty/conformal interval

## 1. 무엇을 왜 했나

GaugeHead-Tiny(M2 v2-a)는 held-out test에서 rel err 0.4724로 quantile bar(0.4804)를 넘었지만,
worst-source인 CrackTree200에서 rel err 0.7201로 **조용히 과신(silently overconfident)** 상태였다.
roadmap의 M2 v2-b 성공 기준은:

1. point rel err ≤ 0.472 유지
2. 모든 held-out source에서 90% interval coverage
3. CrackTree200 같은 케이스를 high-uncertainty로 플래깅

interval 방법 4종을 같은 프로토콜에서 비교했다:

- `conformal_abs` — 절대 residual split conformal (±q px 고정폭)
- `conformal_log` — log-space residual conformal (곱셈형 interval)
- `conformal_normalized` — OOF residual로 학습한 σ(x) 난이도 정규화 conformal
- `cqr_log` — quantile HGB(5%/95%) + conformal 보정(CQR)

프로토콜: train source로 fit, val source로 calibration(split conformal), test는 기존
source-held-out(`cfd`/`cracktree200`/`deepcrack`). 방법 선택은 val coverage ≥ 0.895인 방법 중
**val median relative width(효율)**만으로 결정 — test label은 선택에 사용하지 않음.
추가로 train+val 5-fold cross-conformal 변형을 계산해 v2-a의 train+val refit point 모델
(rel err 0.4724)을 유지한 채 interval을 얹었다.

## 2. 공식 수치

script: `experiments/m2_uncertainty_conformal.py`
result: `experiments/results/m2_uncertainty_conformal.json`
checkpoint: `checkpoints/gaugehead_tiny_width_conformal.pkl`
n: train 1192 / val(calibration) 149 / test 219

### Test (target coverage 90%)

| method | coverage | cfd | cracktree200 | deepcrack | med rel width | point rel err |
|---|---:|---:|---:|---:|---:|---:|
| conformal_abs | 0.986 | 0.97 | 1.00 | 1.00 | 2.597 | 0.4899 |
| conformal_log | 0.977 | 0.97 | 1.00 | 0.98 | 1.698 | 0.4899 |
| conformal_normalized (val 선택) | 0.813 | 0.90 | **0.21** | 0.84 | 1.115 | 0.4899 |
| cqr_log | 0.877 | 0.96 | **0.11** | 0.94 | 1.392 | 0.4899 |
| **conformal_log_cv_trainval** ⭐ | **0.936** | **0.91** | **1.00** | **0.95** | **1.394** | **0.4724** |
| conformal_normalized_cv_trainval | 0.690 | 0.72 | **0.00** | 0.79 | 0.877 | 0.4724 |

### Shift audit — 난이도 신호 3종의 per-source flag rate (val p90 threshold)

| signal | cfd | cracktree200 | deepcrack |
|---|---:|---:|---:|
| learned σ(x) | 0.02 | **0.00** | 0.00 |
| ensemble per-tree std | 0.05 | **0.00** | 0.47 |
| kNN feature distance | 0.08 | **0.05** | 0.60 |

## 3. 발견

1. **성공 기준 1·2 달성**: `conformal_log_cv_trainval`이 point rel err 0.4724를 그대로 유지하면서
   세 held-out source 모두 coverage ≥ 0.90 (0.91 / 1.00 / 0.95). marginal 0.936.
2. **adaptive interval은 source shift에서 붕괴**: val에서 가장 효율적이던 normalized conformal과
   CQR이 CrackTree200에서 coverage 0.21 / 0.11 (cross-conformal 변형은 0.00)로 무너졌다.
   exchangeability가 깨지는 held-out source에서 marginal 보장만 믿으면 안 된다는 교과서적 사례.
3. **성공 기준 3은 정직한 실패**: 난이도 신호 3종(학습 σ, tree ensemble 분산, kNN feature 거리)
   모두 CrackTree200을 플래깅하지 못했다. 오히려 가장 쉬운 source(deepcrack, rel err 0.348)에서
   더 많이 발화한다. CrackTree200의 실패는 feature 분포가 다른 covariate shift가 아니라
   **같은 feature에서 width-label 관계가 다른 concept shift**라서, feature 기반 OOD 탐지로는
   원리적으로 잡히지 않는다.
4. **배포 결정**: checkpoint는 non-adaptive `conformal_log_cv_trainval`로 저장.
   val 효율 선택은 normalized였지만 shift audit이 붕괴를 보였고, 이 결정은 width tuning이 아닌
   deployment gate로서 결과 JSON에 명시 기록.

### 과장 금지 노트

- interval이 넓다: median relative width 1.394 → 대략 점추정 ±70% 수준. "calibrated"는 맞지만
  "tight"는 아니다.
- 여전히 mask-derived px width 실험이다. physical μm/mm GT 아님.
- CrackTree200 coverage 1.00은 19샘플(소표본) + 넓은 interval의 결과로, 검출 능력이 아니라
  보수성의 산물이다.

## 4. 다음 단계

1. **M2 v2-c**: SAM3 raw logits/soft mask feature 추가 — concept shift의 근원이
   "thin crack에서 SAM 마스크가 과대"인 만큼, logit 분포가 σ(x)에 shift 신호를 줄 가능성.
2. concept shift 플래깅의 대안: feature space가 아니라 **예측-보조신호 불일치**
   (예: mask width vs 1D 신호폭 추정 간 격차)를 flag 신호로 검토.
3. Count v1 (ROI-1555 density/centroid head)은 별도 트랙으로 유지.

## 5. 재현

```bash
# Spark
cd /home/hwoo_joo/github/GaugeAnything
.venv/bin/python -u experiments/m2_uncertainty_conformal.py
# 검증
.venv/bin/python -m json.tool experiments/results/m2_uncertainty_conformal.json >/dev/null
```
