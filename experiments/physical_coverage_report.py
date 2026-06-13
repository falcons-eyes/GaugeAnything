"""Generate the Physical AI coverage matrix report and visual.

This is P0 for the GaugeAnything "Anything" story: keep one machine-readable
registry of physical quantities, datasets, GT mechanisms, audited results,
limitations, and next adapters.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "physical_coverage_matrix.json"
OUT_MD = ROOT / "docs" / "PHYSICAL_COVERAGE_MATRIX.md"
OUT_PNG = ROOT / "docs" / "assets" / "physical_coverage_matrix.png"


STATUS_COLOR = {
    "official": "#16805d",
    "partial": "#2b83ba",
    "negative": "#e0394b",
    "next": "#b5791a",
    "candidate": "#8a97a3",
}


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", "<br>") for x in row) + " |")
    return "\n".join(out)


def write_markdown(reg: dict) -> None:
    entries = reg["entries"]
    counts = Counter(e["status"] for e in entries)
    by_track: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_track[entry["track"]].append(entry)

    rows = []
    for e in entries:
        rows.append(
            [
                e["track"],
                f"`{e['id']}`",
                e["quantity"],
                e["unit"],
                ", ".join(e["datasets"]),
                e["gt_mechanism"],
                e["status"],
                e["headline"],
                e["caveat"],
                e["next_step"],
            ]
        )

    status_rows = [[k, v, reg["status_legend"][k]] for k, v in sorted(counts.items())]
    priority = [
        [
            "P1",
            "Document/card scale adapter",
            "SmartDoc15-CH1 + MIDV-500",
            "known-size quadrilateral -> PlaneScale -> edge-length error",
            "Adds a very legible marker-free scale story.",
        ],
        [
            "P2",
            "Timber/log counting adapter",
            "TimberSeg 1.0",
            "global vs tiled SAM3 count, density fallback hook",
            "Expands counting beyond rebar and looks visually obvious.",
        ],
        [
            "P3",
            "Fish/tray length adapter",
            "DeepFish tray / AutoFish",
            "segmentation/exemplar gate -> major-axis length in physical units",
            "Shows physical AI beyond rigid industrial parts.",
        ],
        [
            "P4",
            "BOP object-family expansion",
            "HB/YCB-V/ITODD",
            "CAD+pose dimensions with category holdout",
            "Turns T-LESS from one dataset into a family result.",
        ],
        [
            "P5",
            "Outdoor standard-object adapter",
            "KITTI signs",
            "round sign prompt -> diameter -> standard-size snap",
            "Adds road/outdoor uncontrolled coverage.",
        ],
    ]

    lines = [
        "# Physical AI Coverage Matrix — GaugeAnything",
        "",
        "날짜: 2026-06-12",
        "",
        "목표: GaugeAnything을 단일 crack/ADT 모델이 아니라, 이미지·비디오·센서 입력에서",
        "현장 물리량을 추출하는 promptable physical measurement layer로 정리한다.",
        "",
        f"총 coverage atoms: **{len(entries)}**",
        "",
        "![Physical coverage matrix](assets/physical_coverage_matrix.png)",
        "",
        "## Status Summary",
        "",
        md_table(status_rows, ["status", "count", "meaning"]),
        "",
        "## Coverage Table",
        "",
        md_table(
            rows,
            [
                "track",
                "id",
                "quantity",
                "unit",
                "datasets",
                "GT mechanism",
                "status",
                "headline",
                "caveat",
                "next step",
            ],
        ),
        "",
        "## Adapter Sprint Priorities",
        "",
        md_table(priority, ["priority", "adapter", "dataset", "protocol", "why it matters"]),
        "",
        "## Readout",
        "",
        "- 현재 breakthrough는 `mask=WHERE, signal=WIDTH`, dynamic metric signal, ROI-only collapse, regime routing이다.",
        "- 현재 약점은 coverage가 흩어져 보인다는 점, counting 미해결, ADT oracle gate, image-level physical crack GT coverage 부족이다.",
        "- 따라서 다음 sprint는 새 SOTA 하나가 아니라, physical quantity family를 넓히는 adapter coverage가 우선이다.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def draw_visual(reg: dict) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    entries = reg["entries"]
    width, height = 2000, 1450
    img = Image.new("RGB", (width, height), "#f8fbfc")
    d = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 58)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 25)
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 21)
    except Exception:
        font_title = font_bold = font = font_small = ImageFont.load_default()

    def rr(box, fill, outline="#dce7ee", radius=18, width_px=2):
        d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width_px)

    d.text((70, 58), "GaugeAnything Physical AI Coverage", font=font_title, fill="#13202b")
    d.text(
        (72, 128),
        "Promptable visual measurements across cracks, defects, parts, counts, known objects, dynamic scenes, and physical state.",
        font=font,
        fill="#526477",
    )

    counts = Counter(e["status"] for e in entries)
    x = 70
    for status in ["official", "partial", "negative", "next", "candidate"]:
        txt = f"{status}: {counts.get(status, 0)}"
        color = STATUS_COLOR[status]
        rr((x, 190, x + 245, 255), "#ffffff", "#dce7ee", 14)
        d.ellipse((x + 20, 211, x + 44, 235), fill=color)
        d.text((x + 58, 208), txt, font=font, fill="#13202b")
        x += 270

    cards = entries[:14]
    cols = 4
    card_w, card_h = 445, 205
    start_x, start_y = 70, 310
    gap_x, gap_y = 28, 28
    def short(text: str, n: int) -> str:
        return text if len(text) <= n else text[: n - 3] + "..."

    for i, e in enumerate(cards):
        row, col = divmod(i, cols)
        x0 = start_x + col * (card_w + gap_x)
        y0 = start_y + row * (card_h + gap_y)
        color = STATUS_COLOR.get(e["status"], "#8a97a3")
        rr((x0, y0, x0 + card_w, y0 + card_h), "#ffffff", "#dce7ee", 18)
        d.rectangle((x0, y0, x0 + 8, y0 + card_h), fill=color)
        d.text((x0 + 24, y0 + 22), short(e["track"], 29), font=font_small, fill="#526477")
        d.text((x0 + 24, y0 + 55), short(e["quantity"], 23), font=font_bold, fill="#13202b")
        d.text((x0 + 24, y0 + 100), short(f"unit: {e['unit']}", 34), font=font_small, fill="#526477")
        d.text((x0 + 24, y0 + 132), e["status"].upper(), font=font_small, fill=color)
        d.text((x0 + 24, y0 + 162), short(e["headline"], 43), font=font_small, fill="#526477")

    rr((70, 1342, 1930, 1410), "#eef8f4", "#cde7dd", 18, 1)
    d.text(
        (100, 1362),
        "Next sprint: detected document/card scale -> timber/log counting -> fish/tray length -> BOP family -> outdoor signs.",
        font=font,
        fill="#22513f",
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PNG)


def main() -> None:
    reg = load_registry()
    write_markdown(reg)
    draw_visual(reg)
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
