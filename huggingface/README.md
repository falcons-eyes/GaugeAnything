---
license: apache-2.0
language: en
tags:
  - industrial-inspection
  - metrology
  - crack-detection
  - segmentation
  - measurement
pipeline_tag: image-segmentation
---

# GaugeAnything — task heads for promptable quantitative inspection

**Masks in, millimeters out.** These are the trained task heads of
[GaugeAnything](https://github.com/falcons-eyes/GaugeAnything) — a promptable quantitative
inspection pipeline for industrial micro-vision (SAM 3 backbone + metrology core).

🌐 Project page: https://falcons-eyes.github.io/GaugeAnything/ ·
📊 All numbers below are audited (held-out splits, multi-seed where applicable, protocols in the repo).

## Checkpoints

| File | Task | Audited result | Training data | Use |
|---|---|---|---|---|
| `profile_width_cnn.pt` | **1-D crack-width regression** from a 501-px brightness profile (the "signal for HOW WIDE" head) | table test MAE **17.9 μm**; end-to-end promptable **39.9 μm MAE / 23.2 μm median** (localization-gated) | krkCMd, 14,424 profiles (**CC BY 4.0** — license-clean) | ✅ commercial OK |
| `gaugehead_tiny_width.pkl` | Tiny owned crack-width specialist over SAM-mask/image statistics | held-out source rel.err **0.472** vs 5-bin quantile 0.480 and old neural M2 0.564; worst source still 0.720 | CrackSeg9k M2 cache | ⚠️ research (subset licenses vary) |
| `gaugehead_tiny_width_conformal.pkl` | GaugeHead-Tiny + 90% conformal interval (log cross-conformal; μ + σ-diagnostic + q) | keeps rel.err **0.4724** with per-source coverage **0.91/1.00/0.95** @90%; adaptive variants collapse on the worst source (0.21/0.11) — see repo `experiments/results/m2_uncertainty_conformal.json` | CrackSeg9k M2 cache | ⚠️ research (subset licenses vary) |
| `m2_refiner.pt` | Measurement-aware crack mask refiner (UNet, 1.9M) | superseded baseline: a logit-threshold + quantile calibration beats it (0.437 vs 0.564 rel. err) — kept for reproducibility | CrackSeg9k train sources | ⚠️ research (subset licenses vary) |
| `matte_fray_directional.pt` | Alpha matting head for fuzzy-boundary (fray) defects, directional synthesis v2 | real MT-fray preservation IoU **0.949** vs classical guided filter 0.860 | synthetic compositing over Magnetic-Tile free images | ⚠️ research (MT license unstated) |
| `matte_fray.pt` | v1 (blob synthesis) — kept as the honest negative: real-transfer failure 0.483 | see repo progress logs | same | ⚠️ research |
| `draem_uneven.pt` | DRAEM-lite reconstruction head for boundaryless (uneven/mura) defects | test AUC 0.636 (classical illumination-residual baseline: 0.669) | synthetic mura over Magnetic-Tile free images | ⚠️ research |

The SAM 3 backbone is **not** redistributed here — get it at
[facebook/sam3](https://huggingface.co/facebook/sam3) (separate license, gated).

## Usage (profile width head)

```python
import torch, numpy as np

ckpt = torch.load("profile_width_cnn.pt", map_location="cpu")
# architecture: see experiments/krkcmd_signal_width.py::build_1d_net in the GitHub repo
from gaugeanything_repo.experiments.krkcmd_signal_width import build_1d_net, norm_profile
net = build_1d_net(); net.load_state_dict(ckpt["model"]); net.eval()

profile = np.asarray(...)          # 501 samples of image brightness across the crack
x = torch.from_numpy(norm_profile(profile)).view(1, 1, -1)
width_um = float(net(x))            # crack width in micrometers
```

Full pipeline (`prompt → SAM 3 localization → perpendicular profile → width`) lives in the
[GitHub repo](https://github.com/falcons-eyes/GaugeAnything) — see
`experiments/krkcmd_signal_width.py` and `docs/WIDTH_BOTTLENECK_ANALYSIS.md` for why
width is read from the signal, not from mask geometry.

## Honest limitations

- `profile_width_cnn` is trained on 6400-dpi scanner profiles of concrete (krkCMd);
  transfer to other resolutions/materials is **not yet validated** — scale-normalize inputs.
- End-to-end accuracy is **localization-gated**: coverage 46–66% on the scanner domain;
  points failing the gate are reported as "not measurable", not guessed.
- Heads marked *research* await upstream dataset license clarification before commercial use.

## Citation

```bibtex
@misc{gaugeanything2026,
  title  = {GaugeAnything: Promptable Quantitative Inspection for Industrial Micro-Vision},
  author = {Joo, Hyunwoo},
  year   = {2026},
  url    = {https://github.com/falcons-eyes/GaugeAnything}
}
```
