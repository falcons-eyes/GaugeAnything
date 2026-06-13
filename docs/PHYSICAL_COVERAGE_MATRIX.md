# Physical AI Coverage Matrix — GaugeAnything

날짜: 2026-06-12

목표: GaugeAnything을 단일 crack/ADT 모델이 아니라, 이미지·비디오·센서 입력에서
현장 물리량을 추출하는 promptable physical measurement layer로 정리한다.

총 coverage atoms: **15**

![Physical coverage matrix](assets/physical_coverage_matrix.png)

## Status Summary

| status | count | meaning |
|---|---|---|
| candidate | 1 | Useful coverage expansion, but download/license/protocol still needs work. |
| negative | 1 | Audited negative result that defines a real bottleneck. |
| next | 2 | Dataset path and metric target are identified; adapter/eval is the next sprint item. |
| official | 8 | Audited result exists and is suitable for headline use with stated caveats. |
| partial | 3 | Pipeline/result exists but has a known gate, coverage, or protocol limitation. |

## Coverage Table

| track | id | quantity | unit | datasets | GT mechanism | status | headline | caveat | next step |
|---|---|---|---|---|---|---|---|---|---|
| Crack / Thin Structure | `crack_seg_width_px` | crack mask + pixel width | px | CrackSeg9k, VT LCW | pixel mask; width is measured from mask geometry | official | SAM3 crack mIoU 0.442 vs adaptive 0.181, but width relative error remains 62.9%. | Pixel-mask GT is not physical mm; segmentation rank is not measurement rank. | Use source/category holdout to train measurement-aware gates, not just IoU heads. |
| Crack / Physical GT | `crack_physical_um_profile` | crack width | um | krkCMd table, krkCMd image subset | manual profile width MANwidth, 3.96875 um/px | partial | GaugeProfile+cal 27.8±2.5um 5-fold; signal-width path 39.9um MAE / 23.2um median where localized. | Profile-level and localization-gated results are not yet full image-level promptable coverage. | Expand image-level krkCMd subset and report coverage/error jointly. |
| Surface Defect / Sharp | `surface_hole_diameter` | hole diameter and area | px now, mm when scale exists | Magnetic-Tile Defect | pixel mask | official | Metal-tile blowhole prompt maps to blob measurement; mIoU 0.429 and diameter 11.6px. | Needs physical scale or known part dimensions for mm claim. | Pair surface defects with SmartDoc/marker scale or factory part specs. |
| Surface Defect / Fuzzy Boundary | `fuzzy_fray_alpha` | soft boundary alpha and affected area | alpha, px area | Magnetic-Tile Defect, synthetic directional fray | pixel mask plus synthetic alpha supervision | official | Directional matting v2 reaches 0.949 real-fray mask IoU vs guided matte 0.860. | Real alpha GT is unavailable; evaluation is mask preservation/transfer, not true alpha. | Collect or synthesize directional fray with calibrated alpha-like edge labels. |
| Surface Defect / Boundaryless Field | `field_uneven_severity` | mura/uneven severity | Sa, Sq, AUC | Magnetic-Tile Defect | pixel defect masks evaluated as ranking AUC | official | Binary SAM3 is chance-like on uneven; illumination residual reaches 0.669 test AUC. | Severity units are image-field statistics, not yet calibrated to factory tolerances. | Add calibrated panels or synthetic low-frequency defects with known amplitude. |
| Known Object Scale | `known_object_coin_mm` | coin diameter consistency | mm, relative % | kaa coins | legal coin diameters; leave-one-out same-denomination scale | official | 22 real-photo scenes: mean/median relative error 1.74% / 1.68%, pass@5% 100%. | Known-object consistency, not absolute marker calibration. | Add cards/documents/signs as other known-size scale families. |
| Industrial Parts / CAD Metric | `industrial_part_cad_mm` | visible part dimension | mm, relative % | T-LESS | CAD(mm)+pose+intrinsics | official | T-LESS SAM3 masks match GT objects at 100%; median dimension error 2.5%. | Studio-like BOP setting; object classes are limited and textureless. | Add HB/YCB-V/ITODD for broader object families and category holdout. |
| Counting / Dense Industrial | `rebar_count_density` | count, density, spacing | count | ROI-1555 Rebar, RebarDSC candidate | instance masks / boxes | negative | Global SAM3 MAE 13.2; SAHI tiled improves to MAE 7.35 but dense touching bars remain open. | Prompt tuning is insufficient; density/centroid fallback or supervision is needed. | Train a small centroid/density head and evaluate dense bins separately. |
| Dynamic / Handheld Scale | `dynamic_handheld_scale` | checkerboard cell/edge consistency | relative % | TUM RGB-D checkerboard_large | checkerboard geometry + RGB-D + motion capture | official | Gated handheld RGB-D gives 1.06% median / 2.60% p90 relative error over 160 frames. | Coverage cost is real; gate rejects roughly half of candidate frames. | Add EuRoC blur ladder and SmartDoc video scale stress test. |
| Dynamic / Object Dimensions | `dynamic_object_dimension` | 3D object dimension | relative % | Aria Digital Twin ATEK EFM | RGB-D + object pose/dimensions; oracle volume gate | partial | ADT oracle multiview depth: 8.7% median over 2 seq/480 frames/229 objects; 0.5m/s+ is 9.1%. | Oracle GT-volume gate; ROI-only negative control collapses to 316%. | Replace oracle gate with SAM3/segmentation gate and report gate failure rate. |
| Known Object Scale / Documents | `document_card_scale` | document/card metric scale | mm, relative % | SmartDoc15-CH1, MIDV-500 | A4 or ID-1 known-size quadrilateral | official | P2-1b promptable detected quad: SAM3 'document' prompt reaches gate pass 96%, quad IoU 0.968, edge rel err median 1.5% (vs naive 10-17%); tail remains heavy (p90 27-32%). | Median is promptable and real; the p90 tail (14-32% depending on prompt) means a per-frame quality gate is still required before field claims. MIDV-500 not yet run. | Add per-frame quality gate to cut the p90 tail; evaluate MIDV-500 ID-card scale. |
| Biological / Length | `fish_length_mm` | fish length | mm or cm | DeepFish tray, AutoFish candidate | tray homography or provided length labels | next | Next adapter: non-industrial but physical length shows GaugeAnything beyond rigid parts. | Domain is biological; useful as physical-AI coverage, not core industrial proof. | Add length-regression adapter with segmentation/exemplar gate. |
| Counting / Natural Industrial | `timber_log_count` | log count and diameter distribution | count, px/mm if scale exists | TimberSeg 1.0, FSC-147 logs candidate | instance masks | next | Next adapter: count/density family outside rebar, less touching and more natural clutter. | Scale may be absent; count result is first target. | Implement log instance count adapter and compare SAM3 global vs tiled. |
| Outdoor / Known Standard | `road_sign_diameter` | traffic sign diameter | mm, relative % | KITTI object/sign candidate | 3D boxes plus standard sign diameters | candidate | Candidate adapter: outdoor moving-camera known-object metric sanity check. | Sign category extraction and diameter standard matching need validation. | Filter round signs, snap to standard diameters, report outdoor dynamic error. |
| Physical Sensors / State | `sensor_physical_state` | fault state or RUL | class, cycles, health score | bearing cross-source, N-CMAPSS | machine labels and run-to-failure telemetry | partial | Bearing cross-source collapses even for physics features; N-CMAPSS RUL anchors physical-state framing. | Outside image-only GaugeAnything; belongs to broader Industrial/Physical AI framing. | Keep as motivation appendix, not main visual benchmark. |

## Adapter Sprint Priorities

| priority | adapter | dataset | protocol | why it matters |
|---|---|---|---|---|
| P1 | Document/card scale adapter | SmartDoc15-CH1 + MIDV-500 | known-size quadrilateral -> PlaneScale -> edge-length error | Adds a very legible marker-free scale story. |
| P2 | Timber/log counting adapter | TimberSeg 1.0 | global vs tiled SAM3 count, density fallback hook | Expands counting beyond rebar and looks visually obvious. |
| P3 | Fish/tray length adapter | DeepFish tray / AutoFish | segmentation/exemplar gate -> major-axis length in physical units | Shows physical AI beyond rigid industrial parts. |
| P4 | BOP object-family expansion | HB/YCB-V/ITODD | CAD+pose dimensions with category holdout | Turns T-LESS from one dataset into a family result. |
| P5 | Outdoor standard-object adapter | KITTI signs | round sign prompt -> diameter -> standard-size snap | Adds road/outdoor uncontrolled coverage. |

## Readout

- 현재 breakthrough는 `mask=WHERE, signal=WIDTH`, dynamic metric signal, ROI-only collapse, regime routing이다.
- 현재 약점은 coverage가 흩어져 보인다는 점, counting 미해결, ADT oracle gate, image-level physical crack GT coverage 부족이다.
- 따라서 다음 sprint는 새 SOTA 하나가 아니라, physical quantity family를 넓히는 adapter coverage가 우선이다.
