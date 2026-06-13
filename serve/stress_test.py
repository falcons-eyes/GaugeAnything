"""GaugeAnything 데모 서버 stress/correctness 테스트.

OMEN 로컬에서 실행 (서버는 같은 호스트 컨테이너 :8000):
    .venv/bin/python serve/stress_test.py --img-dir /home/hwoo-joo/ga_stress

검사:
  1) correctness sweep — 트랙별 이미지로 /inspect, /count_rebar 호출, 출력 sanity
  2) latency 분포 — 동일 이미지 반복 호출, p50/p90/p99
  3) concurrency — 병렬 요청 안정성
  4) edge cases — 빈/거대/회색조 이미지, 잘못된 프롬프트
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import statistics
import time
from pathlib import Path

import requests
from PIL import Image

BASE = "http://localhost:8000"


def post(path: str, img_bytes: bytes, fields: dict | None = None, timeout=120):
    files = {"image": ("x.png", img_bytes, "image/png")}
    t = time.time()
    r = requests.post(f"{BASE}{path}", files=files, data=fields or {}, timeout=timeout)
    dt = time.time() - t
    r.raise_for_status()
    return r.json(), dt


def img_bytes(path: Path) -> bytes:
    return path.read_bytes()


def synthetic(w, h, mode="RGB", color=0) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


# 알려진 rebar GT count (ROI-1555 labelme)
REBAR_GT = {"000000000001": None, "000000000293": 1, "000000001255": 81, "000000001563": 61}


def section(t):
    print(f"\n{'='*60}\n{t}\n{'='*60}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path, default=Path.home() / "ga_stress")
    ap.add_argument("--lat-iters", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    imgs = sorted(p for p in args.img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    print(f"server: {requests.get(BASE+'/health').json()['device']} · {len(imgs)} images")

    cracks = [p for p in imgs if "Rissbilder" in p.name or "CRACK" in p.name]
    noncracks = [p for p in imgs if "noncrack" in p.name]
    rebars = [p for p in imgs if p.stem.isdigit()]
    coins = [p for p in imgs if p.name.startswith("test_")]
    mts = [p for p in imgs if "exp" in p.name]
    tless = [p for p in imgs if "tless" in p.name]

    # 1) CORRECTNESS SWEEP
    section("1) CORRECTNESS SWEEP")
    for p in cracks:
        j, dt = post("/inspect", img_bytes(p), {"prompt": "crack", "segmenter": "sam3"})
        ws = [a["width_mean"] for a in j["atoms"]]
        print(f"  crack  {p.name[:34]:34s} → {j['count']} inst, widths_px={[round(w,1) for w in ws][:5]} ({dt:.2f}s)")
    for p in noncracks:
        j, dt = post("/inspect", img_bytes(p), {"prompt": "crack"})
        print(f"  NONcrack {p.name[:32]:32s} → {j['count']} inst (기대 0~소수) ({dt:.2f}s)")
    for p in coins:
        j, dt = post("/inspect", img_bytes(p), {"prompt": "coin"})
        print(f"  coin   {p.name[:34]:34s} → {j['count']} inst ({dt:.2f}s)")
    for p in mts:
        j, dt = post("/inspect", img_bytes(p), {"prompt": "hole"})
        print(f"  MT     {p.name[:34]:34s} → {j['count']} inst ({dt:.2f}s)")
    for p in tless:
        j, dt = post("/inspect", img_bytes(p), {"prompt": "plastic part"})
        print(f"  tless  {p.name[:34]:34s} → {j['count']} inst ({dt:.2f}s)")
    for p in rebars:
        j, dt = post("/count_rebar", img_bytes(p))
        gt = REBAR_GT.get(p.stem)
        err = f"GT={gt} err={abs(j['count']-gt):.1f}" if gt else "GT=?"
        print(f"  rebar  {p.name[:34]:34s} → pred {j['count']:.1f} ({err}) ({dt:.2f}s)")

    # 2) LATENCY 분포 (warm)
    section(f"2) LATENCY ({args.lat_iters} warm calls, 1 crack image)")
    p = cracks[0] if cracks else imgs[0]
    b = img_bytes(p)
    post("/inspect", b, {"prompt": "crack"})  # warmup
    lats = [post("/inspect", b, {"prompt": "crack"})[1] for _ in range(args.lat_iters)]
    lats.sort()
    pct = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]
    print(f"  inspect  p50={pct(.5)*1000:.0f}ms p90={pct(.9)*1000:.0f}ms p99={pct(.99)*1000:.0f}ms "
          f"min={lats[0]*1000:.0f} max={lats[-1]*1000:.0f} mean={statistics.mean(lats)*1000:.0f}")
    if rebars:
        rb = img_bytes(rebars[0])
        post("/count_rebar", rb)
        rlats = sorted(post("/count_rebar", rb)[1] for _ in range(args.lat_iters))
        print(f"  rebar    p50={rlats[len(rlats)//2]*1000:.0f}ms max={rlats[-1]*1000:.0f}ms")

    # 3) CONCURRENCY
    section(f"3) CONCURRENCY ({args.concurrency} parallel inspect)")
    t = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(post, "/inspect", b, {"prompt": "crack"}) for _ in range(args.concurrency * 3)]
        ok, fail = 0, 0
        for f in cf.as_completed(futs):
            try:
                f.result(); ok += 1
            except Exception as e:
                fail += 1; print(f"    FAIL: {e}")
    wall = time.time() - t
    print(f"  {ok} ok / {fail} fail · {args.concurrency*3} reqs in {wall:.1f}s "
          f"→ {args.concurrency*3/wall:.1f} req/s")

    # 4) EDGE CASES
    section("4) EDGE CASES")
    edge = [
        ("tiny 8x8 black", synthetic(8, 8)),
        ("large 4000x3000", synthetic(4000, 3000, color=120)),
        ("grayscale->RGB", synthetic(512, 512, "L", 128)),
        ("solid white", synthetic(512, 512, color=255)),
    ]
    for name, eb in edge:
        try:
            j, dt = post("/inspect", eb, {"prompt": "crack"}, timeout=60)
            print(f"  {name:22s} → {j['count']} inst, {dt:.2f}s  OK")
        except Exception as e:
            print(f"  {name:22s} → ERROR {type(e).__name__}: {str(e)[:60]}")
    # 잘못된/희귀 프롬프트 (synonym collapse 확인)
    for pr in ["fracture", "asdfqwer", ""]:
        try:
            j, _ = post("/inspect", b, {"prompt": pr or "crack"})
            print(f"  prompt={pr!r:12s} → {j['count']} inst")
        except Exception as e:
            print(f"  prompt={pr!r:12s} → ERROR {str(e)[:50]}")

    # GPU 메모리
    section("GPU after stress")
    print("  (run: nvidia-smi on host)")
    print("\nDONE")


if __name__ == "__main__":
    raise SystemExit(main())
