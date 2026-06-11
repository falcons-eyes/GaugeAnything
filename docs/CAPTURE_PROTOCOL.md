# Real-Metric Capture Protocol (ArUco + Caliper)

> Roadmap step: **real-metric ground truth**. There is no public dataset pairing defect photos
> with caliper-measured millimeters — this protocol creates one. Contributions following this
> protocol are welcome (open an issue with a sample first).

## What you need

1. **Printed marker board** — generate with:
   ```bash
   python experiments/make_print_board.py --out board_a4.png   # A4 @ 300 DPI, 4× ArUco 30mm
   ```
   Print at **100% scale** (no "fit to page"), matte paper. Verify with a ruler that the
   marker side is exactly the printed legend (e.g. 30.0 mm). Laminate if reused outdoors.
2. **Caliper** (digital, 0.01 mm) or crack-width gauge card.
3. Any camera (smartphone OK). Avoid digital zoom.

## Shooting checklist (per defect)

1. **Place the board on the same plane** as the defect, as close as practical (≤ 15 cm away).
   The plane assumption is what makes mm conversion valid.
2. Frame so that **both the defect and ≥1 full marker** are sharp and unoccluded.
3. Tilt is OK (PlaneScale corrects up to ~50°, validated ≤1% error) — but avoid extreme
   glancing angles (>60°) and motion blur.
4. Take **2–3 shots** (slightly different positions) per defect.
5. **Measure with caliper**: for cracks, the width at the *widest visible point* and at one
   *typical point*; mark the spots lightly (chalk) and photograph the caliper reading too.
6. Record into the manifest (below) immediately.

## Manifest format (`captures/manifest.csv`)

```csv
image,marker_size_mm,defect_type,prompt,caliper_mm_max,caliper_mm_typical,notes
IMG_0012.jpg,30.0,crack,crack,0.85,0.60,"north wall, chalk mark A"
IMG_0013.jpg,30.0,bolt,hex bolt,13.00,,"M8 head across-flats as sanity"
```

- `caliper_mm_max` is the primary GT (widest point); `typical` optional.
- Include a few **known-dimension sanity objects** (bolt heads: M8=13.0 mm) in every session —
  free calibration checks.

## Evaluate

```bash
python experiments/real_mm_eval.py --captures captures/   # manifest + images
# → per-image: predicted width_max (mm) vs caliper_mm_max, error %, tilt-corrected scale used
# → summary: MAE (mm), relative error distribution, pass rate at ±10% / ±20%
```

## Privacy / licensing for contributions

Only submit photos you have the right to share. Contributed captures are released under
**CC BY 4.0** (stated in your PR). Avoid faces, license plates, and identifiable site info.

## Why this matters

The geometric chain is already validated synthetically (ArUco scale ±0.5%, tilt ≤1% via
homography, e2e width 5.6%). What's missing is the **physical** end of the chain — printer
accuracy, lens distortion, blur, real surface texture. This protocol closes that loop and
produces the first open defect-mm dataset.
