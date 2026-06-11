#!/usr/bin/env bash
# RIGOR_AUDIT 수정 배치 — GPU 순차 실행 (충돌 방지)
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
mkdir -p checkpoints experiments/results

echo "===== [1/5] gauge_bench v1 (시드3, crack/noncrack 분리) ====="
python experiments/gauge_bench.py --n 150 --segmenters adaptive frangi sam3 --seeds 3

echo "===== [2/5] 프롬프트 민감도 스윕 ====="
python experiments/prompt_sweep.py --n 60

echo "===== [3/5] DRAEM 재학습 + 체크포인트 저장 ====="
python experiments/draem_uneven.py --epochs 35 --n-train 900 --save checkpoints/draem_uneven.pt

echo "===== [4/5] uneven val/test 프로토콜 (공식 수치) ====="
python experiments/uneven_protocol.py --ckpt checkpoints/draem_uneven.pt

echo "===== [5/5] matting 재학습 + 저장 + 실제 MT_Fray 검증 ====="
python experiments/matte_fray.py --epochs 35 --save checkpoints/matte_fray.pt --real-eval --gallery docs/assets

echo "AUDIT_BATCH_DONE"
