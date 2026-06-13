"""GaugeAnything demo inference server (FastAPI).

reliable backend: SAM 3 backbone + GaugeAnything metrology core를 인프로세스로 로드해
HTTP로 promptable quantitative inspection을 서빙한다. (ollama/vLLM은 LLM 전용이라
SAM3 세그멘테이션 + CV 측정 헤드에는 부적합 — 그래서 FastAPI 직접 서빙.)

엔드포인트
---------
  GET  /            : 인터랙티브 데모 UI (이미지 업로드 + 프롬프트 → 측정 오버레이)
  GET  /health      : 모델/GPU 상태
  POST /inspect     : multipart(image) + form(prompt, scale...) → atoms + summary + overlay(base64)
  POST /count_rebar : multipart(image) → rebar density head 카운트 + 히트맵

실행 (RTX 5090):
    cd /home/hwoo-joo/github/GaugeAnything
    .venv/bin/python -m uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from gaugeanything.geometry import classify_kind, measure
from gaugeanything.scale import resolve
from gaugeanything.segmenters import get_segmenter

app = FastAPI(title="GaugeAnything", version="1.0")

# 색상 팔레트 (인스턴스 오버레이)
_PALETTE = [
    (22, 128, 93), (43, 131, 186), (181, 121, 26), (224, 57, 75),
    (109, 63, 191), (54, 224, 176), (255, 182, 72), (47, 106, 168),
]
_CKPT = ROOT / "checkpoints"
_STATE: dict = {"rebar": None}


def _img_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _overlay(image: np.ndarray, instances, atoms) -> np.ndarray:
    import cv2

    out = image.copy()
    for i, (inst, atom) in enumerate(zip(instances, atoms)):
        color = _PALETTE[i % len(_PALETTE)]
        m = inst.mask.astype(bool)
        out[m] = (0.55 * np.array(color) + 0.45 * out[m]).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        mt = atom.metrics
        if mt.kind == "thin":
            lbl = f"#{i} {atom.label} w={mt.width_mean:.2f}{mt.unit}"
        else:
            lbl = f"#{i} {atom.label} d={mt.equiv_diameter:.2f}{mt.unit}"
        cv2.putText(out, lbl, (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, lbl, (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
    return out


@app.get("/health")
def health():
    import torch

    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "checkpoints": sorted(p.name for p in _CKPT.glob("*.pt")) + sorted(p.name for p in _CKPT.glob("*.pkl")),
    }


@app.post("/inspect")
async def inspect_endpoint(
    image: UploadFile = File(...),
    prompt: str = Form("crack"),
    segmenter: str = Form("sam3"),
    kind: str = Form("auto"),
    marker_size_mm: float | None = Form(None),
    ref_size_mm: float | None = Form(None),
    manual_mm_per_px: float | None = Form(None),
    max_instances: int = Form(50),
):
    t0 = time.time()
    raw = np.array(Image.open(io.BytesIO(await image.read())).convert("RGB"))
    seg = get_segmenter(segmenter)
    instances = seg(raw, prompt)[:max_instances]

    mm_per_px = None
    scale = resolve(raw if marker_size_mm else None, marker_size_mm=marker_size_mm,
                    manual_mm_per_px=manual_mm_per_px) if (marker_size_mm or manual_mm_per_px) else None
    if scale is not None:
        mm_per_px = scale.mm_per_px

    atoms = []
    from gaugeanything.pipeline import InspectionAtom
    for j, inst in enumerate(instances):
        k = kind if kind != "auto" else classify_kind(inst.mask)
        mt = measure(inst.mask, mm_per_px=mm_per_px, kind=k)
        atoms.append(InspectionAtom(label=inst.label or prompt, confidence=float(inst.score),
                                    metrics=mt, instance_id=j))
    overlay = _overlay(raw, instances, atoms) if instances else raw
    return JSONResponse({
        "prompt": prompt, "segmenter": segmenter,
        "count": len(atoms),
        "scale": ({"mm_per_px": round(scale.mm_per_px, 5), "method": scale.method} if scale else None),
        "unit": "mm" if mm_per_px else "px",
        "atoms": [a.to_dict() for a in atoms],
        "latency_s": round(time.time() - t0, 3),
        "overlay": _img_to_b64(overlay),
    })


@app.post("/count_rebar")
async def count_rebar(image: UploadFile = File(...)):
    import torch

    raw = np.array(Image.open(io.BytesIO(await image.read())).convert("RGB"))
    if _STATE["rebar"] is None:
        sys.path.insert(0, str(ROOT / "experiments"))
        from rebar_density_head import build_net, IN_W, IN_H
        ck = torch.load(_CKPT / "rebar_density_head.pt", map_location="cuda")
        net = build_net().cuda().eval()
        net.load_state_dict(ck["model"])
        _STATE["rebar"] = (net, ck.get("input", f"{IN_W}x{IN_H}"))
    net, spec = _STATE["rebar"]
    import cv2
    iw, ih = (int(x) for x in spec.split()[0].split("x"))
    im = cv2.resize(raw, (iw, ih), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).cuda()
    with torch.no_grad():
        dm = net(x)
    count = float(dm.sum().clamp(min=0).cpu())
    heat = dm[0, 0].clamp(min=0).cpu().numpy()
    heat = (heat / max(heat.max(), 1e-6) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(heat, (raw.shape[1], raw.shape[0])), cv2.COLORMAP_JET)
    blend = (0.5 * heat[:, :, ::-1] + 0.5 * raw).astype(np.uint8)
    return JSONResponse({"count": round(count, 1), "overlay": _img_to_b64(blend),
                         "note": "Count v1 density head (stratified MAE 7.0); dense bars may undercount."})


@app.get("/", response_class=HTMLResponse)
def index():
    return _DEMO_HTML


_DEMO_HTML = """<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>GaugeAnything · live inference</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#0e1620;color:#dfe9f0}
 header{padding:18px 24px;border-bottom:1px solid #1d2a38}
 h1{margin:0;font-size:1.3rem}h1 span{background:linear-gradient(120deg,#16805d,#22c3e6);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
 .sub{color:#7d8fa0;font-size:.82rem;margin-top:4px}
 main{display:grid;grid-template-columns:340px 1fr;gap:0;height:calc(100vh - 70px)}
 .panel{padding:20px 24px;border-right:1px solid #1d2a38;overflow:auto}
 label{display:block;font-size:.78rem;color:#9fb2c0;margin:14px 0 5px}
 input,select,button{width:100%;box-sizing:border-box;padding:9px 11px;border-radius:8px;border:1px solid #2a3a4a;background:#16212e;color:#dfe9f0;font-size:.9rem}
 button{background:linear-gradient(120deg,#16805d,#1ba37a);border:none;font-weight:600;margin-top:18px;cursor:pointer}
 button:disabled{opacity:.5}
 .view{padding:20px;overflow:auto}
 img{max-width:100%;border-radius:10px;border:1px solid #1d2a38}
 pre{background:#0a1219;border:1px solid #1d2a38;border-radius:10px;padding:14px;font-size:.8rem;overflow:auto}
 .chip{display:inline-block;background:#16212e;border:1px solid #2a3a4a;border-radius:99px;padding:3px 10px;font-size:.72rem;margin:3px 3px 0 0}
 .row{display:flex;gap:10px}.row>*{flex:1}
</style></head><body>
<header><h1><span>GaugeAnything</span> · live inference</h1>
<div class=sub>RTX 5090 · SAM 3 backbone + metrology core · masks in, millimeters out</div></header>
<main>
 <div class=panel>
  <label>이미지</label><input type=file id=img accept=image/*>
  <label>프롬프트 (noun phrase)</label><input id=prompt value=crack>
  <div class=row>
   <div><label>모드</label><select id=task><option value=inspect>inspect (measure)</option><option value=count_rebar>count rebar</option></select></div>
   <div><label>segmenter</label><select id=seg><option>sam3</option><option>sam3_ensemble</option><option>adaptive</option></select></div>
  </div>
  <div class=row>
   <div><label>marker mm (옵션)</label><input id=marker placeholder="e.g. 20"></div>
   <div><label>manual mm/px</label><input id=mmpx placeholder="옵션"></div>
  </div>
  <button id=run>측정 실행</button>
  <div id=chips style=margin-top:14px></div>
 </div>
 <div class=view>
  <div id=status class=sub>이미지와 프롬프트를 넣고 실행하세요.</div>
  <div id=out></div>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
$('run').onclick=async()=>{
 const f=$('img').files[0]; if(!f){$('status').textContent='이미지를 선택하세요.';return;}
 const task=$('task').value; const fd=new FormData(); fd.append('image',f);
 $('run').disabled=true; $('status').textContent='추론 중… (SAM 3 첫 호출은 모델 로딩으로 느릴 수 있음)';
 let url='/inspect';
 if(task==='count_rebar'){url='/count_rebar';}
 else{fd.append('prompt',$('prompt').value);fd.append('segmenter',$('seg').value);
  if($('marker').value)fd.append('marker_size_mm',$('marker').value);
  if($('mmpx').value)fd.append('manual_mm_per_px',$('mmpx').value);}
 try{
  const t=performance.now();
  const r=await fetch(url,{method:'POST',body:fd}); const j=await r.json();
  const dt=((performance.now()-t)/1000).toFixed(2);
  if(j.error){$('status').textContent='오류: '+j.error;}
  else{
   let chips='';
   if(task==='count_rebar'){chips=`<span class=chip>count ${j.count}</span>`;}
   else{chips=`<span class=chip>${j.count} instances</span><span class=chip>unit ${j.unit}</span>`+
     (j.scale?`<span class=chip>${j.scale.mm_per_px} mm/px (${j.scale.method})</span>`:'')+
     `<span class=chip>${j.latency_s}s server</span>`;}
   $('chips').innerHTML=chips;
   $('status').textContent='완료 ('+dt+'s round-trip)';
   $('out').innerHTML=`<img src="${j.overlay}"><pre>${JSON.stringify(task==='count_rebar'?{count:j.count,note:j.note}:j.atoms,null,1)}</pre>`;
  }
 }catch(e){$('status').textContent='요청 실패: '+e;}
 $('run').disabled=false;
};
</script></body></html>"""
