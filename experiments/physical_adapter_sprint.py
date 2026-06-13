"""P2 adapter sprint planner for physical-quantity coverage.

This does not download large datasets. It inspects the local DATA_ROOT, records
which next adapters are runnable, and emits a concrete sprint queue.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "datasets"))
OUT = ROOT / "experiments" / "results" / "physical_adapter_sprint.json"


@dataclass
class AdapterSpec:
    adapter_id: str
    priority: int
    datasets: list[str]
    local_paths: list[str]
    ready_patterns: list[str]
    target_quantity: str
    unit: str
    protocol: str
    command: str
    blocker: str

    def available_paths(self) -> list[str]:
        out = []
        for rel in self.local_paths:
            p = DATA_ROOT / rel
            if p.exists():
                out.append(str(p))
        return out

    def ready_evidence(self) -> list[str]:
        out = []
        for pattern in self.ready_patterns:
            out.extend(str(p) for p in DATA_ROOT.glob(pattern))
        return sorted(set(out))


ADAPTERS = [
    AdapterSpec(
        adapter_id="smartdoc_midv_known_quad_scale",
        priority=1,
        datasets=["SmartDoc15-CH1", "MIDV-500"],
        local_paths=["smartdoc", "smartdoc/repo", "midv500", "midv_500"],
        ready_patterns=[
            "smartdoc/metadata.csv.gz",
            "smartdoc/frames.tar.gz",
            "smartdoc/frames/metadata.csv.gz",
            "smartdoc/**/metadata.csv.gz",
            "midv500/**/*.json",
            "midv_500/**/*.json",
        ],
        target_quantity="known document/card edge length",
        unit="mm, relative %",
        protocol="quad GT -> PlaneScale homography -> A4/ID-1 edge error under perspective",
        command="DATA_ROOT=./datasets bash data/scripts/download_metric.sh smartdoc",
        blocker="Need dataset layout/quad parser and MIDV download path validation.",
    ),
    AdapterSpec(
        adapter_id="timberseg_log_count",
        priority=2,
        datasets=["TimberSeg 1.0"],
        local_paths=["timberseg", "timberseg1", "timberseg_1_0"],
        ready_patterns=["timberseg/**/*.json", "timberseg/**/*.png", "timberseg/**/*.jpg"],
        target_quantity="log count and diameter distribution",
        unit="count, px diameter",
        protocol="SAM3 global vs tiled instance count; optional diameter distribution if masks are clean",
        command="Add TimberSeg downloader, then run a tiled count eval analogous to rebar_sahi_eval.py.",
        blocker="Need direct download URL/license check in downloader.",
    ),
    AdapterSpec(
        adapter_id="deepfish_tray_length",
        priority=3,
        datasets=["DeepFish tray", "AutoFish"],
        local_paths=["deepfish_tray", "deepfish", "autofish"],
        ready_patterns=["deepfish_tray/**/*.csv", "deepfish/**/*.csv", "autofish/**/*.csv"],
        target_quantity="fish length",
        unit="mm or cm",
        protocol="fish gate -> major-axis length -> tray/homography or label length comparison",
        command="DATA_ROOT=./datasets bash data/scripts/download_metric.sh deepfish",
        blocker="Need Zenodo record selection and label schema parser.",
    ),
    AdapterSpec(
        adapter_id="bop_family_cad_dimensions",
        priority=4,
        datasets=["HB", "YCB-V", "ITODD"],
        local_paths=["hb", "ycbv", "itodd"],
        ready_patterns=["hb/**/scene_gt.json", "ycbv/**/scene_gt.json", "itodd/**/scene_gt.json"],
        target_quantity="CAD object dimensions",
        unit="mm, relative %",
        protocol="Reuse T-LESS CAD+pose adapter over additional BOP families with category holdout",
        command="Extend data/scripts/download_metric.sh with hb/ycbv; reuse tless_upper_bound.py.",
        blocker="Need per-dataset BOP camera/pose/model path normalization.",
    ),
    AdapterSpec(
        adapter_id="kitti_round_sign_diameter",
        priority=5,
        datasets=["KITTI object/sign candidate"],
        local_paths=["kitti", "kitti_object"],
        ready_patterns=["kitti/**/label_2/*.txt", "kitti_object/**/label_2/*.txt"],
        target_quantity="traffic sign diameter",
        unit="mm, relative %",
        protocol="round sign prompt -> 2D/3D extent -> snap to standard sign diameters",
        command="Add KITTI adapter after sign-category extraction is validated.",
        blocker="Need robust sign filtering; KITTI object labels do not directly isolate all sign subclasses.",
    ),
    AdapterSpec(
        adapter_id="arkitscenes_furniture_dimensions",
        priority=6,
        datasets=["ARKitScenes 3DOD"],
        local_paths=["arkitscenes", "arkit_scenes"],
        ready_patterns=["arkitscenes/**/*.json", "arkit_scenes/**/*.json"],
        target_quantity="furniture dimensions",
        unit="m, relative %",
        protocol="prompt chair/table/sofa -> RGB-D fusion -> 3DOD box dimensions",
        command="Add ARKitScenes downloader/adapter; compare to ADT dynamic object result.",
        blocker="Need sample download path and 3DOD/depth frame parser.",
    ),
]


def main() -> None:
    rows = []
    for spec in ADAPTERS:
        found = spec.available_paths()
        ready_evidence = spec.ready_evidence()
        rows.append(
            {
                **asdict(spec),
                "data_root": str(DATA_ROOT),
                "available_paths": found,
                "ready_evidence": ready_evidence,
                "ready": bool(ready_evidence),
                "recommended_next_action": (
                    "Implement/evaluate adapter now." if ready_evidence else f"Acquire data first: {spec.command}"
                ),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"data_root": str(DATA_ROOT), "adapters": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"data_root": str(DATA_ROOT), "ready": [r["adapter_id"] for r in rows if r["ready"]]}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
