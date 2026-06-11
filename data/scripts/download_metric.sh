#!/usr/bin/env bash
# 실측 mm GT 대체 데이터 다운로드 (data/REAL_MM_SOURCES.md Tier S)
# 사용: DATA_ROOT=./datasets bash data/scripts/download_metric.sh [tless|coins|smartdoc|rebar|deepfish|krkcmd|all]
set -uo pipefail
DATA_ROOT="${DATA_ROOT:-./datasets}"; mkdir -p "$DATA_ROOT"
log() { printf '\033[1;36m[metric]\033[0m %s\n' "$*"; }
HFB="https://huggingface.co/datasets/bop-benchmark"

tless() {  # CC BY 4.0 — CAD(mm)+pose. base+models+test_primesense만 (~수GB)
  local d="$DATA_ROOT/tless"; mkdir -p "$d"
  for f in tless_base.zip tless_models.zip tless_test_primesense_bop19.zip; do
    [ -f "$d/$f" ] && { log "skip $f"; continue; }
    log "↓ $f"; wget -q -c "$HFB/tless/resolve/main/$f" -O "$d/$f"
  done
  ( cd "$d" && for z in *.zip; do unzip -n -q "$z"; done ) && log "tless OK"
}

coins() {  # MIT (kaa) + HF coins-1apki
  local d="$DATA_ROOT/coins"; mkdir -p "$d"
  [ -d "$d/kaa" ] || git clone -q --depth 1 https://github.com/kaa/coins-dataset "$d/kaa"
  log "kaa OK ($(find "$d/kaa" -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l) imgs)"
}

smartdoc() {  # A4 quad GT — GitHub 가공판 (zenodo 원본은 대용량 비디오)
  local d="$DATA_ROOT/smartdoc"; mkdir -p "$d"
  [ -d "$d/repo" ] || git clone -q --depth 1 https://github.com/jchazalon/smartdoc15-ch1-dataset "$d/repo"
  log "smartdoc repo OK (데이터 링크는 repo README 참조)"
}

rebar() {  # ROI-1555 (HF 비gated)
  local d="$DATA_ROOT/rebar_roi1555"; mkdir -p "$d"
  log "HF tsrobcvai/ROI-1555 나열·다운로드"
  python3 - "$d" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("tsrobcvai/ROI-1555_Rebar_Detection_and_Instance_Segmentation_Dataset",
                  repo_type="dataset", local_dir=sys.argv[1])
print("rebar OK")
PY
}

deepfish() {  # CC BY 4.0, 개체별 mm 길이 — zenodo 레코드 자동 탐색
  local d="$DATA_ROOT/deepfish_tray"; mkdir -p "$d"
  log "DeepFish(tray): Zenodo 레코드는 sources 문서 참조 — 수동 1회 (레코드 검색 필요)"
}

krkcmd() {  # CC BY 4.0, 크랙 폭 GT — 36GB 대용량
  local d="$DATA_ROOT/krkcmd"; mkdir -p "$d"
  log "krkCMd 파일 목록 (Zenodo 14568863)"
  curl -s "https://zenodo.org/api/records/14568863" | python3 -c "
import sys, json
for f in json.load(sys.stdin).get('files', []):
    print(f['size'], f['key'], f['links']['self'])" | head -20 | tee "$d/files.txt"
  log "→ 대용량(36GB). 폭 GT 테이블·샘플 스택만 우선: files.txt에서 선택 wget"
}

case "${1:-all}" in
  tless) tless ;; coins) coins ;; smartdoc) smartdoc ;; rebar) rebar ;;
  deepfish) deepfish ;; krkcmd) krkcmd ;;
  all) coins; smartdoc; rebar; tless; krkcmd ;;
  *) echo "usage: $0 [tless|coins|smartdoc|rebar|deepfish|krkcmd|all]"; exit 1 ;;
esac
log "완료"
