# 2026-06-11 — Codex handoff after Claude resume

This is a concise handoff index for future Claude/Codex sessions. The full detailed handoff is also stored at:

`/Users/jamesjoo/work/falconoon.com/.claude/GaugeAnything_HANDOFF_2026-06-11.md`

## Environment

- Local repo copy: `/Users/jamesjoo/work/falconoon.com/GaugeAnything`
- DGX Spark source of truth for datasets/checkpoints:
  - `ssh hwoo_joo-Edgexpert-Spark`
  - `cd /home/hwoo_joo/github/GaugeAnything`
- Use Spark `.venv/bin/python`; system `python3` lacks required packages for selftests.

## Work completed after Claude limit

1. EDA report generator and report
   - `experiments/eda_report.py`
   - `docs/EDA_REPORT.md`
   - `docs/eda_report.html`

2. E-mm-3 krkCMd profile-level physical crack-width benchmark
   - `experiments/krkcmd_profile_eval.py`
   - `experiments/results/krkcmd_profile_eval.json`
   - `docs/assets/krkcmd_profile.png`
   - `docs/progress/2026-06-11_krkcmd-profile-emm3.md`

3. Project page/storytelling update
   - `docs/index.html`
   - `docs/assets/story_industrial_hero.png`
   - `docs/assets/story_inspection_atoms.png`

4. Docs updated
   - `README.md`
   - `experiments/RESULTS.md`
   - `paper/DATASETS.md`
   - `paper/RELATED_BASELINES.md`
   - `docs/progress/README.md`

5. Dynamic-scene ADT continuation
   - `experiments/adt_atek_projection_audit.py`
   - `experiments/adt_atek_box_dimension_upper.py`
   - `experiments/adt_atek_depth_upper.py`
   - `docs/progress/2026-06-12_adt-atek-access-probe.md`
   - `docs/assets/adt_atek_projection_audit.png`
   - `docs/assets/adt_atek_depth_upper.png`
   - `docs/assets/adt_dynamic_multiseq_summary.png`
   - ROI-only depth negative control: `experiments/results/adt_atek_depth_roi_ablation.json`

## Key new numbers

E-mm-3 krkCMd, group-split test:

| Method | test MAE |
|---|---:|
| DLMwidth(author) | 11.1 μm |
| GaugeProfile-minrun5 + linear-cal | 25.9 μm |
| AEDwidth(author) | 26.5 μm |
| GaugeProfile-minrun5, uncalibrated | 31.3 μm |

Existing E-mm-1/E-cnt-1 results are now also pulled into EDA/project page:

| Result | Value |
|---|---:|
| coin known-object LOO mean error | 1.74% |
| coin pass@5% / pass@10% | 100% / 100% |
| rebar best prompt MAE | 13.2 |
| rebar acc@10% | 0% |
| matting v2 real fray IoU | 0.949 |
| guided matte real fray IoU | 0.860 |
| ADT oracle RGB-D multiview median error | 8.7% (2 seq / 480 frames / 229 objects) |
| ADT high-speed bin median error | 9.1% (0.5m/s+, n=160) |
| ADT ROI-only depth negative control | 316.0% median error (mask/gate required) |

## Verification

On Spark:

```bash
cd /home/hwoo_joo/github/GaugeAnything
.venv/bin/python experiments/eda_report.py
.venv/bin/python -m gaugeanything.selftest
.venv/bin/python -m gaugeanything.soft_selftest
```

Both selftests passed on Spark.

Local `localhost:8848/index.html` was changed to serve `GaugeAnything/docs`; previously the port was serving
`Industrial_Anything/docs`.

## Next recommended work

- Commit/push the GaugeAnything changes so GitHub Pages updates.
- Continue E-mm-2 T-LESS CAD+pose dimension derivation.
- Try krkCMd image zip subset extraction for image-level promptable measurement.
- Try rebar counting improvement via SAHI tiling/density fallback/small supervised head.
- Continue E-dyn-3d/e: replace ADT oracle GT-volume gate with ADT segmentation or SAM3 masks, then report
  speed/blur/occlusion bins and gate failure rate.
