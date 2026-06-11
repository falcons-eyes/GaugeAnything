#!/usr/bin/env bash
# GaugeAnything 비전 트랙 — 상업 클린(CC0/CC BY) 데이터셋 다운로드
# 사용: DATA_ROOT=/path bash download_vision.sh [crackseg9k|vt_suite|sdnet|all]
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./datasets}"
mkdir -p "$DATA_ROOT"
log() { printf '\033[1;35m[vision]\033[0m %s\n' "$*"; }

# ---- CrackSeg9k (CC0, Harvard Dataverse) — 크랙 분할 9.3k ----
# 주의: DeepCrack/GAPs384 서브셋은 원본이 NC → 상업 학습 시 파일 접두사로 제외
crackseg9k() {
  local d="$DATA_ROOT/crackseg9k"; mkdir -p "$d"
  if [ -f "$d/dataverse_files.zip" ]; then log "skip crackseg9k (exists)"; return; fi
  log "CrackSeg9k 전체 다운로드 (Dataverse API, 수 GB)"
  curl -L -o "$d/dataverse_files.zip" \
    "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/EGIEBY"
}

# ---- VT Bianchi 스위트 (CC0, figshare) — 실제 교량검사 ----
#   LCW 크랙 분할 3.8k / Corrosion CS 4등급 분할 440 / COCO-Bridge 검출 1.5k
vt_suite() {
  local articles=("16624672:lcw_cracks" "16624663:corrosion_cs" "16624495:coco_bridge")
  for entry in "${articles[@]}"; do
    local id="${entry%%:*}" name="${entry##*:}"
    local d="$DATA_ROOT/vt_suite/$name"; mkdir -p "$d"
    log "VT $name (figshare $id) 파일 목록 조회"
    curl -s "https://api.figshare.com/v2/articles/$id" \
      | python3 -c "
import sys, json
for f in json.load(sys.stdin).get('files', []):
    print(f['download_url'], f['name'])" \
      | while read -r url fname; do
          if [ -f "$d/$fname" ]; then log "  skip $fname"; continue; fi
          log "  ↓ $fname"
          curl -sL -o "$d/$fname" "$url"
        done
  done
}

# ---- SDNET2018 (CC BY, Kaggle 미러가 가장 스크립트 친화적) ----
sdnet() {
  local d="$DATA_ROOT/sdnet2018"; mkdir -p "$d"
  if ! command -v kaggle >/dev/null; then log "kaggle CLI 필요 (pip install kaggle)"; return; fi
  kaggle datasets download -d aniruddhsharma/structural-defects-network-concrete-crack-images -p "$d" || \
    log "SDNET kaggle 미러 실패 — USU Digital Commons에서 수동: https://digitalcommons.usu.edu/all_datasets/48/"
}

case "${1:-all}" in
  crackseg9k) crackseg9k ;;
  vt_suite)   vt_suite ;;
  sdnet)      sdnet ;;
  all)        vt_suite; crackseg9k; sdnet ;;
  *) echo "usage: $0 [crackseg9k|vt_suite|sdnet|all]"; exit 1 ;;
esac
log "완료"
