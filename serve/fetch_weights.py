"""우리 학습 가중치(HF James-joobs/GaugeAnything, 공개)를 checkpoints/로 내려받는다.

SAM 3 backbone(facebook/sam3, gated)은 첫 추론 시 transformers가 자동 다운로드한다
(hf 토큰 + 라이선스 동의 필요).

실행: .venv/bin/python serve/fetch_weights.py
"""
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

REPO = "James-joobs/GaugeAnything"
FILES = [
    "profile_width_cnn.pt",
    "gaugehead_tiny_width.pkl",
    "gaugehead_tiny_width_conformal.pkl",
    "m2_refiner.pt",
    "matte_fray_directional.pt",
    "draem_uneven.pt",
    "rebar_density_head.pt",
]
OUT = Path(__file__).resolve().parents[1] / "checkpoints"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in FILES:
        p = hf_hub_download(REPO, f)
        shutil.copy(p, OUT / f)
        print(f"  {f}")
    print(f"saved {len(FILES)} weights to {OUT}")


if __name__ == "__main__":
    main()
