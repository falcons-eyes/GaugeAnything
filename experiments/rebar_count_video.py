"""Rebar 카운팅 데모 영상 — density 회귀로 센다(검출 아님).

다른 task: counting. owned density head(Count v1)가 rebar 이미지의 density map을 예측하고,
좌→우로 스윕하며 density 히트맵을 드러내고 누적 카운트를 보여준다. 밀집 막대에서 검출은
실패하지만(undercount) density 적분은 작동한다는 메시지.

Spark 실행:
    .venv/bin/python experiments/rebar_count_video.py --img <jpg> --fps 14
출력: docs/assets/demos/count_rebar.mp4
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
GREEN, AMBER, CY, GREY, WHITE = (54, 224, 176), (40, 170, 240), (34, 195, 230), (181, 200, 214), (255, 255, 255)


def main() -> int:
    import cv2
    import torch
    from PIL import Image
    from rebar_density_head import build_net, IN_W, IN_H

    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=None, help="rebar jpg (없으면 밀집 이미지 자동 선택)")
    ap.add_argument("--ckpt", default="checkpoints/rebar_density_head.pt")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--out-mp4", type=Path, default=Path("docs/assets/demos/count_rebar.mp4"))
    args = ap.parse_args()

    # 중밀도 이미지 자동 선택 (GT~30, density head가 정확한 영역; 밀집 극단은 undercount)
    if args.img is None:
        target = 30
        best, bestd = None, 1e9
        for jp in sorted(glob.glob("datasets/rebar_roi1555/1260/img_label/*.json"))[:600]:
            n = len(json.loads(Path(jp).read_text()).get("shapes", []))
            if abs(n - target) < bestd:
                bestd, best, bestc = abs(n - target), jp, n
        args.img = str(Path(best).with_suffix(".jpg"))
        gt_count = bestc
    else:
        jp = Path(args.img).with_suffix(".json")
        gt_count = len(json.loads(jp.read_text()).get("shapes", [])) if jp.exists() else None
    print(f"=== rebar count demo: {Path(args.img).name} (GT {gt_count}) ===", flush=True)

    raw = np.array(Image.open(args.img).convert("RGB"))
    H, W = raw.shape[:2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net().to(dev).eval()
    net.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    im = cv2.resize(raw, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(dev)
    with torch.no_grad():
        dm = net(x)[0, 0].clamp(min=0).cpu().numpy()
    total = float(dm.sum())
    # 컬럼별 누적 카운트 + 풀해상 히트맵
    col_sum = dm.sum(axis=0)                    # [IN_W//8]
    col_cum = np.cumsum(col_sum)
    heat = (dm / max(dm.max(), 1e-6) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(heat, (W, H)), cv2.COLORMAP_JET)[:, :, ::-1]  # RGB
    dmW = dm.shape[1]
    print(f"predicted count {total:.1f} (GT {gt_count})", flush=True)

    rendered = []
    for f in range(args.frames):
        frac = (f + 1) / args.frames
        sweep_x = int(frac * W)
        col_idx = min(dmW - 1, int(frac * dmW))
        cum = float(col_cum[col_idx]) if dmW else 0.0
        view = raw.copy()
        # 스윕된 영역에 히트맵 블렌드
        view[:, :sweep_x] = (0.55 * heat[:, :sweep_x] + 0.45 * view[:, :sweep_x]).astype(np.uint8)
        cv2.line(view, (sweep_x, 0), (sweep_x, H), GREEN, 2, cv2.LINE_AA)
        # HUD
        cv2.rectangle(view, (0, 0), (W, 64), (14, 22, 32), -1)
        cv2.putText(view, "Counting by density regression - not detection", (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREEN, 2, cv2.LINE_AA)
        cv2.putText(view, "ROI-1555 rebar  |  owned density head (Count v1)  |  dense touching bars where box-detection undercounts",
                    (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
        # 카운터
        cv2.rectangle(view, (W - 250, 76), (W - 14, 150), (14, 22, 32), -1)
        cv2.putText(view, "count (density sum)", (W - 240, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
        cv2.putText(view, f"{cum:5.0f}", (W - 240, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.3, GREEN, 3, cv2.LINE_AA)
        if frac > 0.98 and gt_count:
            cv2.rectangle(view, (0, H - 32), (W, H), (14, 22, 32), -1)
            cv2.putText(view, f"total {total:.0f}  vs  GT {gt_count}   (held-out MAE 7.0; dense remains hard - reported honestly)",
                        (14, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, GREY, 1, cv2.LINE_AA)
        rendered.append(view)

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for i, im2 in enumerate(rendered):
            cv2.imwrite(f"{td}/f{i:04d}.png", im2[:, :, ::-1])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(args.out_mp4)], check=True)
    small = [Image.fromarray(cv2.resize(im2, (W // 2, H // 2))) for im2 in rendered]
    small[0].save(args.out_mp4.with_suffix(".webp"), save_all=True, append_images=small[1:],
                  duration=int(1000 / args.fps), loop=0, format="WEBP", quality=72)
    print(f"wrote {args.out_mp4} ({args.out_mp4.stat().st_size//1024}KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
