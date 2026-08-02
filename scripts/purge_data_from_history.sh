#!/usr/bin/env bash
#
# Remove every data file from the working tree, the index, AND all git
# history, then force-push.
#
#   ############################################################
#   #  DESTRUCTIVE AND IRREVERSIBLE.                           #
#   #  Run ./scripts/stage_kaggle_upload.sh FIRST, and upload   #
#   #  the result to Kaggle, and confirm it is there.           #
#   #  This deletes the only local copy of ~231 MB of data.     #
#   ############################################################
#
# Why history and not just `git rm`:
# `git rm --cached` leaves every byte in the pack files. The clone stays 204 MB
# and every `git clone` on Kaggle still drags it down -- which is the entire
# problem this is meant to solve. The blobs have to come out of history.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BACKUP="../$(basename "$REPO")-backup-$(date +%Y%m%d-%H%M%S)"

echo "repo: $REPO"
echo
echo "This will rewrite every commit on every branch and force-push."
echo "Data in HEAD right now:"
git ls-files \
  | grep -E '\.(parquet|zip|csv)$' \
  | grep -v '^configs/' \
  | sed 's/^/    /' || echo "    (none -- already clean)"
echo
echo "Repository size now: $(du -sh .git | cut -f1)"
echo
read -r -p "Have you already staged AND uploaded the Kaggle dataset? [type: yes] " ok
[ "$ok" = "yes" ] || { echo "Aborted. Run scripts/stage_kaggle_upload.sh first."; exit 1; }

# --- 0. local backup, in case ----------------------------------------------
echo
echo "==> full backup to $BACKUP"
cp -a "$REPO" "$BACKUP"
echo "    keep this until you have confirmed the Kaggle dataset works."

# --- 1. git-filter-repo -----------------------------------------------------
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "==> installing git-filter-repo"
  pip install --user git-filter-repo || {
    echo "Install it manually: pip install git-filter-repo"
    exit 1
  }
fi

# filter-repo refuses to run on a repo with a remote unless --force, and it
# removes the remote afterwards; both are recorded here so re-adding it below
# is not a surprise.
REMOTE_URL="$(git remote get-url origin)"

echo
echo "==> rewriting history"
git filter-repo --force \
  --invert-paths \
  --path data/bbb/ \
  --path data/raw/ \
  --path data/auction/ \
  --path data/identity/cricinfo_resolution.csv \
  --path 02-kaggle-paths.patch \
  --path-glob '*.parquet' \
  --path-glob '*_json.zip'

# --- 2. reflog + gc, or nothing actually shrinks ----------------------------
echo
echo "==> expiring reflog and repacking"
rm -rf .git/refs/original
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo
echo "Repository size now: $(du -sh .git | cut -f1)"

# --- 3. push ----------------------------------------------------------------
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"

echo
echo "History is rewritten LOCALLY. Nothing has been pushed yet."
echo
echo "To publish (this rewrites the remote -- anyone else with a clone will"
echo "have to re-clone, they cannot pull):"
echo
echo "    git push --force --all origin"
echo "    git push --force --tags origin"
echo
echo "Then, on every other machine you use, DELETE the old clone and"
echo "re-clone. Do not pull -- a pull will merge the old history back in and"
echo "put all 231 MB straight back."
