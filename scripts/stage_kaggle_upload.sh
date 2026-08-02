#!/usr/bin/env bash
#
# Assemble the upload folder for the Kaggle dataset
#   udvasbasak2/ipl-auction-model-inputs
#
# RUN THIS BEFORE scripts/purge_data_from_history.sh.
# The purge deletes the only local copy of these files. There is no undo.
#
#   ./scripts/stage_kaggle_upload.sh [REPO_DIR] [OUT_DIR]
#
# Defaults: REPO_DIR = the repo this script lives in
#           OUT_DIR  = ../kaggle-upload/ipl-auction-model-inputs
#
set -euo pipefail

REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${2:-$(dirname "$REPO")/kaggle-upload/ipl-auction-model-inputs}"

echo "repo : $REPO"
echo "out  : $OUT"
echo

if [ ! -d "$REPO/data" ]; then
  echo "ERROR: $REPO/data does not exist."
  echo "Either you are pointing at the wrong repo, or the purge has already"
  echo "run. If it has, recover the files from a clone made before the purge:"
  echo "    git clone <url> recovered && cd recovered && git checkout <old-sha>"
  exit 1
fi

mkdir -p "$OUT"/{cricsheet,shotquality,identity,auction}

# --- 1. Cricsheet match json ------------------------------------------------
# The 20 zips, uploaded as zips. Kaggle does NOT auto-extract a .zip that is
# one file among many in an upload, and build_bbb streams json straight out of
# a zip anyway, so there is nothing to gain by extracting them first.
echo "cricsheet/"
n=0
for z in "$REPO"/data/raw/*_json.zip; do
  [ -e "$z" ] || continue
  cp "$z" "$OUT/cricsheet/"
  n=$((n+1))
done
echo "  $n zips"

# people.csv: the Cricsheet register. Optional for build_bbb (the per-match
# registry is enough for identity) but it adds the cross-site Cricinfo ids,
# and Kaggle notebooks often run without internet, so pin it.
if [ -f "$REPO/data/raw/people.csv" ]; then
  cp "$REPO/data/raw/people.csv" "$OUT/cricsheet/"
  echo "  people.csv"
fi

# --- 2. Shot quality feeds --------------------------------------------------
# Irreplaceable: these are the source for ball_attributes.parquet and cannot
# be regenerated from anything else in this project.
echo "shotquality/"
for f in t20_bbb.parquet t20_bbb-updated.parquet t20_combined.parquet; do
  if [ -f "$REPO/data/raw/shotquality/$f" ]; then
    cp "$REPO/data/raw/shotquality/$f" "$OUT/shotquality/"
    echo "  $f"
  else
    echo "  WARNING: missing $f"
  fi
done

# --- 3. Curated CSVs --------------------------------------------------------
# Hand-made. Losing either of these means redoing work by hand.
echo "identity/ + auction/"
cp "$REPO/data/identity/cricinfo_resolution.csv" "$OUT/identity/"
cp "$REPO/data/auction/player_archetypes.csv"    "$OUT/auction/"
echo "  cricinfo_resolution.csv"
echo "  player_archetypes.csv"

# --- 4. Kaggle metadata -----------------------------------------------------
cat > "$OUT/dataset-metadata.json" <<'JSON'
{
  "title": "IPL Auction Model Inputs",
  "id": "udvasbasak2/ipl-auction-model-inputs",
  "licenses": [{"name": "other"}]
}
JSON

# --- 5. A manifest, so a later session can prove nothing was lost -----------
( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )

echo
echo "staged $(find "$OUT" -type f | wc -l) files, $(du -sh "$OUT" | cut -f1)"
echo
echo "NOT uploaded, on purpose:"
echo "  data/bbb/*.parquet   derived -- rebuilt by pipelines/build_bbb.py"
echo "  ball_attributes      derived -- rebuilt by pipelines/build_shot_attributes.py"
echo
echo "Next:"
echo "  cd $OUT"
echo "  kaggle datasets create -p . --dir-mode zip"
echo
echo "Then, and only then: ./scripts/purge_data_from_history.sh"
