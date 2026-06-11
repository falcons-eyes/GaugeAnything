"""Build an exploratory analysis report from audited GaugeAnything artifacts.

The report is intentionally artifact-first: it uses checked-in result JSON files
and gallery images, so it can run on a laptop without the private Spark datasets
or SAM 3 weights. If raw datasets/checkpoints are present, their availability is
recorded in the report, but the official numbers still come from results/.

Usage:
    python experiments/eda_report.py
"""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
ASSETS = ROOT / "docs" / "assets"
OUT_MD = ROOT / "docs" / "EDA_REPORT.md"
OUT_HTML = ROOT / "docs" / "eda_report.html"


def load_json(name: str) -> dict[str, Any]:
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_json(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100 * value:.{digits}f}%"


def asset(name: str, alt: str, width: str = "100%") -> str:
    return f'<img src="assets/{name}" alt="{html.escape(alt)}" width="{width}">'


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def result_inventory() -> list[list[str]]:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        size_kb = path.stat().st_size / 1024
        rows.append([path.name, f"{size_kb:.1f} KB"])
    return rows


def local_artifact_inventory() -> list[list[str]]:
    checks = [
        ("datasets/", ROOT / "datasets", "raw benchmark images and labels"),
        ("checkpoints/m2_refiner.pt", ROOT / "checkpoints" / "m2_refiner.pt", "measurement-aware refiner weights"),
        ("checkpoints/matte_fray_directional.pt", ROOT / "checkpoints" / "matte_fray_directional.pt", "directional matting v2 weights"),
        ("checkpoints/draem_uneven.pt", ROOT / "checkpoints" / "draem_uneven.pt", "field-regime DRAEM-lite weights"),
        ("docs/assets/", ASSETS, "checked-in visual evidence"),
        ("datasets/coins", ROOT / "datasets" / "coins", "known-object real-photo mm substitute"),
        ("datasets/rebar_roi1555", ROOT / "datasets" / "rebar_roi1555", "rebar instance-count GT"),
        ("datasets/tless", ROOT / "datasets" / "tless", "CAD+pose metric object candidate"),
        ("datasets/krkcmd", ROOT / "datasets" / "krkcmd", "crack-width real-mm candidate"),
    ]
    rows = []
    for label, path, meaning in checks:
        state = "present" if path.exists() else "not present in this checkout"
        rows.append([label, state, meaning])
    return rows


def build_markdown() -> str:
    gauge = load_json("gauge_bench.json")
    measure = load_json("gauge_bench_measure.json")
    multi = load_json("gauge_multidomain.json")
    soft = load_json("soft_inspection.json")
    uneven = load_json("uneven_protocol.json")
    matte = load_json("matte_fray.json")
    m2 = load_json("m2_refiner.json")
    prompt = load_json("prompt_sweep.json")
    ensemble = load_json("prompt_ensemble.json")
    scale = load_json("scale_perspective.json")
    coins = load_optional_json("coins_mm_eval.json")
    rebar = load_optional_json("rebar_count_eval.json")
    rebar_sahi = load_optional_json("rebar_sahi_eval.json")
    krk = load_optional_json("krkcmd_profile_eval.json")

    sam3 = gauge["results"]["sam3"]
    adaptive = gauge["results"]["adaptive"]
    frangi = gauge["results"]["frangi"]
    bench_gain = sam3["crack_mIoU_mean"] / adaptive["crack_mIoU_mean"]
    width_gain = 1 - m2["test_overall"]["refined"]["width_rel_err"] / m2["test_overall"]["raw"]["width_rel_err"]
    tilt50 = scale["rows"][-1]

    md: list[str] = []
    md.append("# GaugeAnything EDA Report")
    md.append("")
    md.append(f"Generated: {date.today().isoformat()}")
    md.append("")
    md.append(
        "This exploratory report consolidates the training/evaluation artifacts from the current "
        "GaugeAnything session: audited result JSON files, representative images, model/regime "
        "metadata, and the known limits of each result. It is designed to be re-generated with "
        "`python experiments/eda_report.py`."
    )
    md.append("")

    md.append("## Executive findings")
    md.append("")
    md.append(
        table(
            ["Finding", "Evidence", "Interpretation"],
            [
                [
                    "SAM 3 is the best zero-shot crack segmenter",
                    f"mIoU {fmt(sam3['crack_mIoU_mean'])} vs adaptive {fmt(adaptive['crack_mIoU_mean'])} ({bench_gain:.2f}x)",
                    "Good backbone choice, but still far from supervised upper bounds.",
                ],
                [
                    "Segmentation quality is not measurement quality",
                    "SAM3 width rel.err 62.9%; adaptive 43.5%",
                    "GaugeAnything should optimize measurement atoms, not only IoU.",
                ],
                [
                    "M2 refiner helps width without hurting IoU",
                    f"width rel.err {fmt(m2['test_overall']['raw']['width_rel_err'])}->{fmt(m2['test_overall']['refined']['width_rel_err'])} ({pct(width_gain)} relative reduction)",
                    "Promising, but bias sign is domain-dependent.",
                ],
                [
                    "Fuzzy/field defects need continuous outputs",
                    "SAM3 binary AUC about 0.50; soft methods 0.60-0.67",
                    "Route by visual regime: binary, matting, or field residual.",
                ],
                [
                    "PlaneScale fixes tilt-driven mm error",
                    f"50 deg naive {fmt(tilt50['naive_err_pct'])}% vs homography {fmt(tilt50['homog_err_pct'])}%",
                    "Metric claims need local homography scale, not one global px/mm.",
                ],
                [
                    "Known-object real photos validate the diameter chain",
                    (
                        f"coin LOO mean {pct(coins['summary']['rel_err_mean'])}, pass@5% {pct(coins['summary']['pass@5pct'])}"
                        if coins else "coin JSON not present in this checkout"
                    ),
                    "This is not a field caliper dataset, but it is real-image mm consistency.",
                ],
                [
                    "krkCMd gives physical crack-width MAE in micrometers",
                    (
                        f"GaugeProfile+cal test MAE {krk['summary']['GaugeProfile-minrun5+linear-cal']['group_split_test']['MAE_um']:.1f}um; author DLM {krk['summary']['DLMwidth(author)']['group_split_test']['MAE_um']:.1f}um"
                        if krk else "krkCMd JSON not present in this checkout"
                    ),
                    "This completes a profile-level physical GT cell; image-level promptable validation remains next.",
                ],
                [
                    "Rebar counting is a confirmed zero-shot failure",
                    (
                        (
                            f"global MAE {min(v['summary']['MAE'] for v in rebar.values()):.2f}; "
                            f"SAHI MAE {rebar_sahi['summary']['MAE']:.2f}"
                        )
                        if rebar_sahi else
                        f"best prompt MAE {min(v['summary']['MAE'] for v in rebar.values()):.2f}, acc@10% {pct(max(v['summary']['count_acc@10pct'] for v in rebar.values()))}"
                        if rebar else "rebar JSON not present in this checkout"
                    ),
                    "Tiling helps, but touching, low-contrast circular ends still need density or supervision.",
                ],
                [
                    "Prompt synonyms can collapse completely",
                    "fracture/pit single prompt mIoU 0.0; ensemble recovers 0.35-0.37",
                    "Prompt-set routing is a required reliability layer.",
                ],
            ],
        )
    )
    md.append("")

    md.append("## Artifact inventory")
    md.append("")
    md.append("### Local files")
    md.append("")
    md.append(table(["Artifact", "State", "Meaning"], local_artifact_inventory()))
    md.append("")
    md.append("### Result JSON inputs")
    md.append("")
    md.append(table(["File", "Size"], result_inventory()))
    md.append("")

    md.append("## Visual case gallery")
    md.append("")
    md.append("These are the checked-in visual artifacts used as qualitative anchors for the EDA.")
    md.append("")
    md.append(table(["Regime", "Image", "What to inspect"], [
        ["End-to-end gauge demo", asset("gauge_demo.png", "crack gauge demo"), "Mask-to-width/length atoms on a real concrete surface."],
        ["Real-metric substitute", asset("coins_mm.png", "coin millimeter evaluation"), "Known-object leave-one-out mm consistency with many instances."],
        ["Crack source: CFD", asset("gallery_cfd.png", "CFD crack gallery"), "Concrete crack morphology and SAM3 thin-structure behavior."],
        ["Crack source: CrackTree200", asset("gallery_cracktree200.png", "CrackTree200 crack gallery"), "Hard thin-crack source where per-source IoU is weak."],
        ["Crack source: DeepCrack", asset("gallery_deepcrack.png", "DeepCrack gallery"), "Held-out source used by M2 reporting."],
        ["Magnetic tile: blowhole", asset("dom_mt_blowhole.png", "magnetic tile blowhole"), "Blob-like defect where diameter is the natural measurement."],
        ["Magnetic tile: fray", asset("dom_mt_fray.png", "magnetic tile fray"), "Fuzzy boundary regime: binary masks become brittle."],
        ["Matting v2", asset("matte_fray_real.png", "real fray matting"), "Directional matting transfer evidence on real fray."],
        ["Field uneven", asset("soft_uneven.png", "soft uneven map"), "Boundaryless field anomaly where residual maps beat binary segmentation."],
        ["krkCMd profiles", asset("krkcmd_profile.png", "krkCMd brightness profiles"), "Physical crack-width GT in micrometers over 501-pixel profiles."],
        ["Rebar counting", asset("rebar_count.png", "rebar count failure/success panel"), "Instance-count behavior on dense touching bars."],
    ]))
    md.append("")

    md.append("## Experiment 1: zero-shot crack segmentation")
    md.append("")
    md.append("**Data/labels.** CrackSeg9k, crack-only masks; empty-GT images are excluded from crack mIoU and reported separately as non-crack clean rate.")
    md.append("")
    md.append(table(
        ["Model", "Type", "mIoU", "std", "non-crack clean", "sec/img"],
        [
            ["SAM3", "foundation model", fmt(sam3["crack_mIoU_mean"]), fmt(sam3["crack_mIoU_std"]), pct(sam3["noncrack_clean_rate"]), fmt(sam3["sec_per_img"])],
            ["adaptive", "classical threshold", fmt(adaptive["crack_mIoU_mean"]), fmt(adaptive["crack_mIoU_std"]), pct(adaptive["noncrack_clean_rate"]), fmt(adaptive["sec_per_img"])],
            ["frangi", "classical vesselness", fmt(frangi["crack_mIoU_mean"]), fmt(frangi["crack_mIoU_std"]), pct(frangi["noncrack_clean_rate"]), fmt(frangi["sec_per_img"])],
        ],
    ))
    md.append("")
    md.append("**EDA read.** SAM3 wins both segmentation and false-positive avoidance, but the per-source spread shows thin-crack brittleness. This is the empirical reason for keeping SAM3 as the backbone while adding measurement-aware heads and calibration.")
    md.append("")

    md.append("## Experiment 2: measurement fidelity")
    md.append("")
    md.append("**Data/labels.** Same crack masks, but the target is geometry derived from GT masks: width, length, and skeleton/EDT statistics. This is not real-mm ground truth.")
    md.append("")
    rows = []
    for name, r in measure["results"].items():
        rows.append([name, fmt(r["mIoU"]), fmt(r["width_mae_px"]), pct(r["width_rel_err"]), f"{fmt(r['gt_width_mean_px'])}->{fmt(r['pred_width_mean_px'])} px"])
    md.append(table(["Method", "mIoU", "width MAE", "width rel.err", "GT->pred width"], rows))
    md.append("")
    md.append("**EDA read.** SAM3 has the best mIoU and absolute width MAE, but adaptive has the best relative width error. This mismatch is the strongest evidence that an inspection model needs metric atoms and calibration, not a mask score alone.")
    md.append("")

    md.append("## Experiment 3: M2 measurement-aware refiner")
    md.append("")
    md.append("**Model.** A small refiner over frozen SAM3 masks and image features. **Training/evaluation split.** Source-held-out test: CFD, CrackTree200, DeepCrack.")
    md.append("")
    md.append(table(
        ["Scope", "Raw mIoU", "Refined mIoU", "Raw width err", "Refined width err", "Raw bias", "Refined bias"],
        [["overall", fmt(m2["test_overall"]["raw"]["mIoU"]), fmt(m2["test_overall"]["refined"]["mIoU"]), fmt(m2["test_overall"]["raw"]["width_rel_err"]), fmt(m2["test_overall"]["refined"]["width_rel_err"]), fmt(m2["test_overall"]["raw"]["width_bias"]), fmt(m2["test_overall"]["refined"]["width_bias"])]]
        + [
            [src, fmt(v["raw"]["mIoU"]), fmt(v["refined"]["mIoU"]), fmt(v["raw"]["width_rel_err"]), fmt(v["refined"]["width_rel_err"]), fmt(v["raw"]["width_bias"]), fmt(v["refined"]["width_bias"])]
            for src, v in m2["test_per_source"].items()
        ],
    ))
    md.append("")
    md.append("**EDA read.** The refiner reduces held-out width error by roughly 23% relative while preserving mIoU. The caution is structural: bias direction differs by source, so M2 v2 should be domain- or scale-conditioned.")
    md.append("")

    md.append("## Experiment 4: multi-domain defect behavior")
    md.append("")
    rows = []
    for domain, r in multi.items():
        rows.append([domain, r["n"], fmt(r["mIoU"]), r["kind"], fmt(r["mean_width_px"]), fmt(r["mean_dia_px"])])
    md.append(table(["Domain", "n", "mIoU", "shape tags", "mean width px", "mean diameter px"], rows))
    md.append("")
    md.append("**EDA read.** Concrete cracks, metal cracks, and blowholes transfer reasonably because they map to concrete visual nouns and stable measurement primitives. Fray, break, and uneven collapse because their labels are textural or boundaryless anomalies.")
    md.append("")

    md.append("## Experiment 5: soft inspection and learned fuzzy/field heads")
    md.append("")
    md.append(table(
        ["Defect", "n", "SAM3 binary AUC", "residual AUC", "raw gray AUC", "Sa", "Sq"],
        [[k, v["n"], fmt(v["sam3_binary_AUC"]), fmt(v["residual_AUC"]), fmt(v["raw_gray_AUC"]), fmt(v["Sa"]), fmt(v["Sq"])] for k, v in soft.items()],
    ))
    md.append("")
    md.append(table(
        ["Field model", "val", "test", "Selected config"],
        [
            ["classical residual", fmt(uneven["classical"]["val"]), fmt(uneven["classical"]["test"]), uneven["classical"]["config"]],
            ["DRAEM-lite", fmt(uneven["draem"]["val"]), fmt(uneven["draem"]["test"]), uneven["draem"]["config"]],
        ],
    ))
    md.append("")
    md.append(table(
        ["Matting metric", "Value"],
        [
            ["synthetic alpha MAE, directional matting", fmt(matte["alpha_mae_matting"], 4)],
            ["synthetic alpha MAE, binary", fmt(matte["alpha_mae_binary"], 4)],
            ["real fray IoU at alpha>=0.5", fmt(matte["real_fray"]["iou_at_0.5_vs_gt"])],
            ["guided matte IoU", fmt(matte["real_fray"]["guided_matte_iou"])],
            ["boundary softness", fmt(matte["real_fray"]["boundary_softness"])],
        ],
    ))
    md.append("")
    md.append("**EDA read.** Soft maps recover signal where binary masks are near random. The directional matting v2 result is the important reversal: learned matting now beats the classical guided matte on real fray preservation, while alpha accuracy remains synthetic-only because real alpha GT is unavailable.")
    md.append("")

    md.append("## Experiment 6: prompts, confidence, and metadata")
    md.append("")
    prompt_rows = []
    for domain, r in prompt.items():
        prompt_rows.append([domain, r["n"], fmt(r["mean"]), fmt(r["best"]), fmt(r["spread"]), r["per_prompt"]])
    md.append(table(["Domain", "n", "mean", "best", "spread", "per-prompt mIoU"], prompt_rows))
    md.append("")
    md.append(table(
        ["Ensemble case", "broken prompt", "single broken", "single best", "ensemble via broken"],
        [[k, v["broken_prompt"], fmt(v["single_broken"]), fmt(v["single_best"]), fmt(v["ensemble_via_broken"])] for k, v in ensemble.items()],
    ))
    md.append("")
    md.append("**Metadata availability.** Learned heads can expose logits directly because they are local PyTorch modules. SAM3 metadata depends on the HuggingFace `Sam3Processor`/`Sam3Model` return schema; `experiments/eda_probe.py` probes tensor attributes, post-processed masks, scores, and low-threshold candidate-score spectra before any confidence-heavy analysis.")
    md.append("")

    md.append("## Experiment 7: scale, mm rigor, and real-metric substitutes")
    md.append("")
    md.append(table(
        ["Tilt", "detected", "naive mm", "naive err", "homography mm", "homography err"],
        [[r["theta"], r["detected"], fmt(r["naive_mm"]), f"{fmt(r['naive_err_pct'])}%", fmt(r["homog_mm"]), f"{fmt(r['homog_err_pct'])}%"] for r in scale["rows"]],
    ))
    md.append("")
    md.append("**EDA read.** The scale failure is geometric, not model-specific. A single marker-derived px/mm can be nearly 20% wrong under tilt; local PlaneScale homography brings the 50-degree case below 1% error.")
    md.append("")
    md.append("**Real-metric substitute note.** The coin leave-one-out result from the progress log reports 22 images with 8-60 coins/image, mean relative error 1.74%, median 1.68%, and 100% pass at ±5%/±10%. This validates the segmentation-to-diameter chain on real photos, even though it is known-object consistency rather than an industrial caliper dataset.")
    md.append("")

    if coins:
        md.append("### E-mm-1: coin known-object consistency")
        md.append("")
        s = coins["summary"]
        md.append(table(
            ["Metric", "Value"],
            [
                ["images", s["n_images"]],
                ["mean relative error", pct(s["rel_err_mean"])],
                ["median relative error", pct(s["rel_err_median"])],
                ["mean px coefficient of variation", pct(s["cv_mean"])],
                ["pass@5%", pct(s["pass@5pct"])],
                ["pass@10%", pct(s["pass@10pct"])],
                ["scope note", s["note"]],
            ],
        ))
        worst = sorted(coins["rows"], key=lambda r: r["rel_err"], reverse=True)[:8]
        md.append("")
        md.append(table(
            ["Worst scene", "denom", "n coins", "rel.err", "px CV"],
            [[r["file"], r["denom"], r["n_coins"], pct(r["rel_err"]), pct(r["cv_px"])] for r in worst],
        ))
        md.append("")
        md.append("**EDA read.** The worst image is still below 3% relative error, and all scenes pass ±5%. The failure surface is not count density itself: several 45-60 coin scenes remain stable. The metric caveat is that this is same-denomination leave-one-out consistency, not an independent absolute-scale marker measurement.")
        md.append("")

    if krk:
        md.append("### E-mm-3: krkCMd physical crack-width profiles")
        md.append("")
        meta = krk["meta"]
        md.append(table(
            ["Meta", "Value"],
            [
                ["profiles", meta["n_profiles"]],
                ["groups", meta["n_groups"]],
                ["train/test profiles", f"{meta['train_profiles']} / {meta['test_profiles']}"],
                ["unit", meta["unit"]],
                ["px_to_um", meta["px_to_um"]],
                ["GT", meta["gt"]],
                ["scope", meta["note"]],
            ],
        ))
        preferred = [
            "DLMwidth(author)",
            "GaugeProfile-minrun5+linear-cal",
            "AEDwidth(author)",
            "GaugeProfile-minrun5",
            "GaugeProfile-halfdepth+linear-cal",
        ]
        rows = []
        for name in preferred:
            s = krk["summary"][name]["group_split_test"]
            rows.append([name, f"{s['MAE_um']:.1f}um", f"{s['RMSE_um']:.1f}um", f"{s['median_abs_err_um']:.1f}um", pct(s["pass@50um"]), s["pearson_r"]])
        md.append("")
        md.append(table(["Method", "test MAE", "test RMSE", "test medAE", "pass@50um", "r"], rows))
        md.append("")
        md.append("**EDA read.** The author DLM is the strong specialized anchor at 11.1um MAE. A deterministic GaugeProfile valley rule with group-split linear calibration reaches 25.9um MAE, essentially matching the author AED baseline on this split. This is a profile-level physical GT result, not yet a full image prompt-to-mask measurement.")
        md.append("")

    if rebar:
        md.append("### E-cnt-1: rebar counting")
        md.append("")
        rebar_rows = []
        for prompt_name, data in rebar.items():
            s = data["summary"]
            rebar_rows.append([prompt_name, s["n"], s["gt_mean"], fmt(s["MAE"]), pct(s["rel_err_mean"]), pct(s["count_acc@10pct"]), pct(s["exact_rate"])])
        md.append(table(["Prompt", "n", "GT mean", "MAE", "mean rel.err", "acc@10%", "exact"], rebar_rows))
        best_prompt, best_data = min(rebar.items(), key=lambda kv: kv[1]["summary"]["MAE"])
        examples = sorted(best_data["rows"], key=lambda r: abs(r["pred"] - r["gt"]), reverse=True)[:8]
        md.append("")
        md.append(table(
            [f"Worst rows for best prompt '{best_prompt}'", "GT", "pred", "abs err"],
            [[r["img"], r["gt"], r["pred"], abs(r["pred"] - r["gt"])] for r in examples],
        ))
        md.append("")
        md.append("**EDA read.** Prompt wording does not rescue the task: all prompts have 0% acc@10%. The best prompt undercounts dense scenes sharply and can overcount sparse scenes, consistent with a visual-domain failure on touching, rusty, low-contrast bar ends rather than a pure language mismatch.")
        md.append("")

    if rebar_sahi and rebar:
        md.append("### E-cnt-2: SAHI-style tiled rebar counting")
        md.append("")
        best_global_name, best_global_data = min(rebar.items(), key=lambda kv: kv[1]["summary"]["MAE"])
        g = best_global_data["summary"]
        s = rebar_sahi["summary"]
        md.append(table(
            ["Method", "n", "MAE", "mean rel.err", "acc@10%", "exact", "pred mean"],
            [
                [f"Global SAM3 '{best_global_name}'", g["n"], fmt(g["MAE"]), pct(g["rel_err_mean"]), pct(g["count_acc@10pct"]), pct(g["exact_rate"]), "-"],
                ["SAHI SAM3 tiled", s["n"], fmt(s["MAE"]), pct(s["rel_err_mean"]), pct(s["count_acc@10pct"]), pct(s["exact_rate"]), s["pred_mean"]],
            ],
        ))
        worst = sorted(rebar_sahi["rows"], key=lambda r: r["abs_err"], reverse=True)[:6]
        md.append("")
        md.append(table(
            ["Worst SAHI rows", "GT", "pred", "abs err"],
            [[r["img"], r["gt"], r["pred"], r["abs_err"]] for r in worst],
        ))
        md.append("")
        md.append("**EDA read.** Tiling materially improves MAE (13.2 to 7.35) and creates nonzero acc@10%, so the global failure is partly a scale/crowding problem. It does not solve dense touching counts: the largest piles are still undercounted by 20-40 bars, pointing to density/centroid supervision.")
        md.append("")

    md.append("## Limits and next EDA passes")
    md.append("")
    datasets_present = (ROOT / "datasets").exists()
    md.append(table(
        ["Limit", "Why it matters", "Next action"],
        [
            [
                "Raw dataset panel depth",
                (
                    "Datasets are mounted in this Spark checkout, but this report uses audited aggregate artifacts for reproducibility."
                    if datasets_present
                    else "Raw datasets are not present in this checkout, so only checked-in aggregate artifacts are used."
                ),
                "Extend the generator with per-sample exports for the most useful failure clusters.",
            ],
            ["SAM3 logits not assumed stable", "Processor/model schemas can change.", "Use `eda_probe.py` to record available fields before relying on confidence curves."],
            ["Real alpha GT missing", "Fray alpha accuracy is synthetic-only.", "Collect or synthesize calibrated translucent/fuzzy boundary targets."],
            ["Mask-derived width GT", "M2 evaluates measurement consistency, not physical crack width.", "Proceed with krkCMd/T-LESS/ArUco real-mm tracks."],
            ["Counting zero-shot gap", "Rebar result shows SAM3 concept limits on touching low-contrast instances.", "Evaluate SAHI tiling, density fallback, or small supervised head."],
        ],
    ))
    md.append("")

    return "\n".join(md) + "\n"


def markdown_to_html(md: str) -> str:
    """Small dependency-free renderer good enough for this report."""
    lines = md.splitlines()
    body: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        if table_rows:
            head = table_rows[0]
            rows = table_rows[2:] if len(table_rows) > 1 else []
            body.append("<table>")
            body.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead>")
            body.append("<tbody>")
            for row in rows:
                body.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
            body.append("</tbody></table>")
        in_table = False
        table_rows = []

    for line in lines:
        if line.startswith("| ") and line.endswith(" |"):
            in_table = True
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        flush_table()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif not line.strip():
            body.append("")
        else:
            escaped = html.escape(line)
            escaped = escaped.replace("&lt;img ", "<img ").replace("&gt;", ">")
            escaped = escaped.replace("`python experiments/eda_report.py`", "<code>python experiments/eda_report.py</code>")
            body.append(f"<p>{escaped}</p>")
    flush_table()

    return """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GaugeAnything EDA Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; margin: 0; color: #17202a; background: #f7f8fa; }
main { max-width: 1180px; margin: 0 auto; padding: 40px 24px 80px; background: white; }
h1, h2, h3 { line-height: 1.18; color: #0c1720; }
h1 { font-size: 34px; margin: 0 0 20px; }
h2 { font-size: 24px; margin-top: 42px; border-top: 1px solid #dfe5ea; padding-top: 24px; }
h3 { font-size: 18px; margin-top: 26px; }
p { max-width: 980px; }
table { width: 100%; border-collapse: collapse; margin: 18px 0 28px; font-size: 14px; }
th, td { border: 1px solid #d8dee4; padding: 8px 10px; vertical-align: top; }
th { background: #eef3f7; text-align: left; }
img { max-width: 360px; height: auto; border: 1px solid #d8dee4; background: #fff; }
code { background: #edf2f7; padding: 2px 5px; border-radius: 4px; }
</style>
</head>
<body><main>
""" + "\n".join(body) + "\n</main></body></html>\n"


def main() -> int:
    md = build_markdown()
    OUT_MD.write_text(md, encoding="utf-8")
    OUT_HTML.write_text(markdown_to_html(md), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"wrote {OUT_HTML.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
