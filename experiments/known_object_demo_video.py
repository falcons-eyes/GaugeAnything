"""Known-object 스케일 데모 영상 — 실사진에서 동전들을 훑으며 각 직경을 mm로 측정.

다른 task: known-object scale. SAM3 'coin'으로 한 장면의 모든 동전을 검출하고, 동전들의
중앙값 직경 = 법정 직경(1유로 23.25mm)으로 px→mm 캘리브레이션. 가상 카메라(Ken Burns)가
동전들을 순서대로 훑으며 현재 동전의 mm 직경을 표시한다. 같은 권종 동전들이 일관되게
~23mm로 측정되는 것(LOO 1.74% 스토리)을 보여준다.

⚠️ 합성 모션: 정지 실사진에 Ken Burns. 화면 명시.

Spark 실행:
    .venv/bin/python experiments/known_object_demo_video.py \
        --img-glob 'datasets/coins/kaa/src/1e/*.JPG' --prompt coin --known-mm 23.25 \
        --label "1 euro coin" --fps 12
출력: docs/assets/demos/coin_<label>.mp4
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
GREEN, AMBER, CY, GREY, WHITE = (54, 224, 176), (40, 170, 240), (34, 195, 230), (181, 200, 214), (255, 255, 255)
VIEW_W, VIEW_H = 1000, 640
ZOOM = 2.6


def equiv_diam_px(mask):
    return float(np.sqrt(4 * mask.sum() / np.pi))


def main() -> int:
    import cv2
    from PIL import Image
    from gaugeanything.segmenters import segment_sam3

    ap = argparse.ArgumentParser()
    ap.add_argument("--img-glob", default="datasets/coins/kaa/src/1e/*.JPG")
    ap.add_argument("--prompt", default="coin")
    ap.add_argument("--known-mm", type=float, default=23.25)
    ap.add_argument("--label", default="1 euro coin")
    ap.add_argument("--proc-width", type=int, default=2400)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold", type=int, default=5, help="동전당 프레임 수")
    ap.add_argument("--max-coins", type=int, default=14)
    ap.add_argument("--out-mp4", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.img_glob))
    if not files:
        print(f"no image for {args.img_glob}"); return 1
    out_mp4 = args.out_mp4 or Path(f"docs/assets/demos/coin_{args.label.replace(' ','_')}.mp4")
    out_webp = out_mp4.with_suffix(".webp")
    print(f"=== known-object demo: {files[0].split('/')[-1]} prompt={args.prompt!r} ===", flush=True)

    full = np.array(Image.open(files[0]).convert("RGB"))
    H0, W0 = full.shape[:2]
    s = min(1.0, args.proc_width / W0)
    img = cv2.resize(full, (int(W0 * s), int(H0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else full
    H, W = img.shape[:2]

    inst = segment_sam3(img, args.prompt)
    coins = []
    for k in inst:
        m = k.mask.astype(bool)
        if m.sum() < 0.0004 * H * W:
            continue
        ys, xs = np.nonzero(m)
        cy, cx = float(ys.mean()), float(xs.mean())
        coins.append({"mask": m, "cx": cx, "cy": cy, "d_px": equiv_diam_px(m)})
    if not coins:
        print("no coins detected"); return 1
    # 위치 순서(좌상→우하)로 정렬
    coins.sort(key=lambda c: (c["cy"] // (H / 4), c["cx"]))
    coins = coins[: args.max_coins]
    med_d = float(np.median([c["d_px"] for c in coins]))
    mm_per_px = args.known_mm / med_d                     # known-object 캘리브
    for c in coins:
        c["d_mm"] = c["d_px"] * mm_per_px
    diffs = [abs(c["d_mm"] - args.known_mm) / args.known_mm for c in coins]
    print(f"{len(coins)} {args.prompt}s, median {med_d:.1f}px -> {args.known_mm}mm, "
          f"consistency mean {np.mean(diffs)*100:.1f}%", flush=True)

    scan_bgr = img  # RGB
    rendered = []
    for ci, coin in enumerate(coins):
        for _ in range(args.hold):
            cx, cy = coin["cx"], coin["cy"]
            hw, hh = int(VIEW_W / ZOOM / 2), int(VIEW_H / ZOOM / 2)
            x0 = int(np.clip(cx - hw, 0, W - 2 * hw)); y0 = int(np.clip(cy - hh, 0, H - 2 * hh))
            view = cv2.resize(scan_bgr[y0:y0 + 2 * hh, x0:x0 + 2 * hw], (VIEW_W, VIEW_H), interpolation=cv2.INTER_CUBIC)
            vx, vy = int((cx - x0) * ZOOM), int((cy - y0) * ZOOM)
            r = int(coin["d_px"] / 2 * ZOOM)
            cv2.circle(view, (vx, vy), r, GREEN, 3, cv2.LINE_AA)
            cv2.line(view, (vx - r, vy), (vx + r, vy), CY, 2, cv2.LINE_AA)
            cv2.putText(view, f"{coin['d_mm']:.1f} mm", (vx - 40, vy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 3, cv2.LINE_AA)
            cv2.putText(view, f"{coin['d_mm']:.1f} mm", (vx - 40, vy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
            # HUD
            cv2.rectangle(view, (0, 0), (VIEW_W, 66), (14, 22, 32), -1)
            cv2.putText(view, "Known-object scale - every coin read in millimetres", (14, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
            cv2.putText(view, f"prompt: {args.prompt}  |  {args.label} = {args.known_mm} mm legal diameter  |  synthetic Ken Burns over a real photo  |  coin {ci+1}/{len(coins)}",
                        (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
            cv2.rectangle(view, (0, VIEW_H - 34), (VIEW_W, VIEW_H), (14, 22, 32), -1)
            cv2.putText(view, f"{len(coins)} coins, all same denomination -> all measured ~{args.known_mm:.1f} mm (consistency {np.mean(diffs)*100:.1f}%)",
                        (14, VIEW_H - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.46, GREY, 1, cv2.LINE_AA)
            rendered.append(view)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for i, im in enumerate(rendered):
            cv2.imwrite(f"{td}/f{i:04d}.png", im[:, :, ::-1])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_mp4)], check=True)
    small = [Image.fromarray(cv2.resize(im, (VIEW_W // 2, VIEW_H // 2))) for im in rendered]
    small[0].save(out_webp, save_all=True, append_images=small[1:], duration=int(1000 / args.fps), loop=0, format="WEBP", quality=72)
    print(f"wrote {out_mp4} ({out_mp4.stat().st_size//1024}KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
