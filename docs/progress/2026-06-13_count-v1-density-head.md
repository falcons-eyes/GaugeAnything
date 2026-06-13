# 2026-06-13 — Count v1: ROI-1555 density/centroid head

## 1. 무엇을 왜 했나

[로드맵](../MODEL_RESEARCH_ROADMAP.md)의 Count v1: rebar 카운팅은 더 이상 prompt 문제가
아니다(E-cnt-1 best prompt MAE 13.1, E-cnt-2 SAHI MAE 8.9 — dense touching bar 구조적
undercount). 밀집 인스턴스는 detection보다 density 회귀가 맞다는 가설을 owned model로 검증.

`image → tiny FCN(0.11M) → 1-channel density map → count = sum(map)`.
GT density는 각 폴리곤 centroid에 정규화 Gaussian(합=1). 목표: held-out MAE < 5,
dense-bin undercount 50%+ 감소.

## 2. 학습 함정 2건 (정직 기록)

1. **zero-collapse**: 첫 학습은 출력 ReLU + 작은 per-pixel density(~1e-3) + 약한 count
   손실로 전부 0 예측(test pred_mean 0.0). 출력 ReLU 제거 + count L1을 주 신호로
   재구성해 해결.
2. **split 분포 이동**: id-정렬 contiguous block split이 우연히 test 블록을 고밀도로
   몰아(test gt_mean 29.8 vs 전체 21.4) 저밀도 학습 모델이 외삽 실패 → MAE 12.0.
   ROI-1555 라벨 이미지는 scene 메타가 없어, count-bin **stratified split**(밀도 분포
   정합)이 가장 방어 가능한 분할. cosine LR로 학습도 안정화.

## 3. 공식 수치

script: `experiments/rebar_density_head.py`
result: `experiments/results/rebar_density_head.json` (stratified, primary)
        `experiments/results/rebar_density_head_block.json` (block, stress)

| split | test MAE | bias | acc@10% | dense-bin MAE (GT≥40) | dense-bin bias |
|---|---:|---:|---:|---:|---:|
| **stratified (primary)** | **7.0** | −2.1 | 0.19 | 30.9 | −29.3 |
| block (density-shift stress) | 12.0 | +0.5 | 0.19 | 29.8 | −28.2 |

앵커: SAHI-SAM3 zero-shot MAE 8.9 (E-cnt-2, n=40 random) · target MAE < 5.

## 4. 판정

1. **owned density head가 zero-shot SAHI를 이긴다**: 공정 split에서 MAE 7.0 < 8.9,
   acc@10% 0.19 vs 0.075 (2.5×). counting 트랙이 negative→**partial**로 전진.
2. **목표 <5 미달**: 7.0. 그리고 **진짜 문제는 그대로** — dense-bin(GT≥40) MAE 30.9,
   bias −29.3. 밀집 영역에서 여전히 절반 가까이 undercount.
3. **근본 원인은 dense crowding**: 학습 분포가 저밀도(평균 21)에 쏠려 모델이 bulk로
   회귀하고, /8 density map(128×80)이 GT~75 밀집 막대를 분해 못 함. LR/split 문제가
   아니라 표현 capacity + count imbalance 문제.

## 5. Count v1.1 — dense-regime 공략 (실행 완료)

v1의 dense undercount를 직접 공략: 고해상도(1536×960) + sharp sigma(1.5) +
count-weighted loss(sqrt(gt) 가중). result: `rebar_density_head_v11.json`.

| 버전 | overall MAE | bias | acc@10% | dense-bin MAE | dense-bin bias |
|---|---:|---:|---:|---:|---:|
| v1 (stratified) | 7.0 | −2.1 | 0.19 | 30.9 | −29.3 |
| **v1.1 (hi-res+weighted)** | 7.8 | +0.5 | 0.14 | **24.7** | **−19.1** |

**판정**: dense-regime는 **공략 가능**하다 — dense-bin undercount를 −29.3→−19.1로 **35%
감소**, 전체 bias도 −2.1→+0.5로 거의 제거. 단 overall MAE는 7.0→7.8로 소폭 악화:
count-weight가 dense에 capacity를 몰면서 bulk(쉬운 케이스)를 희생했다. 단일 tiny head +
global loss로는 bulk/dense를 동시에 못 잡는 트레이드오프가 드러남.

**다음 방향(Count v2)**: 단일 head의 한계가 명확하므로 — (a) multi-scale/SAHI-density
하이브리드(밀집 영역만 고해상 타일), (b) 더 큰 backbone, (c) 밀도-적응 커널. 목표
dense-bin MAE <15, overall <5는 여전히 미달 — counting은 partial로 유지.

## 6. 재현

```bash
# Spark
.venv/bin/python -u experiments/rebar_density_head.py --epochs 80 --split stratified
.venv/bin/python -u experiments/rebar_density_head.py --epochs 80 --split block \
  --out experiments/results/rebar_density_head_block.json
```
