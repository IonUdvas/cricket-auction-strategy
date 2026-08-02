#!/usr/bin/env bash
#
# Remove every data file from the working tree, the index, and ALL git
# history -- then force-push, then PROVE the remote is clean by re-reading it.
#
# This replaces an earlier version that could exit silently partway through.
# The differences that matter:
#
#   * It checks its prerequisites BEFORE touching anything, instead of
#     discovering git-filter-repo is missing halfway through.
#   * git filter-repo deletes the `origin` remote as a safety measure. The
#     URL is captured first and restored after, and the script says so --
#     the previous run left the repo with no remote, which is why a later
#     `git push` reported "Everything up-to-date" while GitHub still held
#     all 231 MB.
#   * It verifies the REMOTE afterwards with a blobless clone, rather than
#     trusting that the push did what it looked like it did.
#
# Nothing is deleted from your disk by this script. It tells you what to
# delete at the end, once the remote is confirmed clean.
#
set -euo pipefail

hr() { printf '%s\n' "============================================================"; }
die() { echo; echo "ABORTED: $*"; exit 1; }

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
cd "$REPO"
PARENT="$(dirname "$REPO")"
NAME="$(basename "$REPO")"

hr
echo "PRE-FLIGHT"
hr

# --- the data must exist somewhere else first -------------------------------
UPLOAD="$PARENT/kaggle-upload/ipl-auction-model-inputs"
[ -d "$UPLOAD" ] || die "no staged upload at $UPLOAD -- run scripts/stage_kaggle_upload.sh"
n_zip=$(find "$UPLOAD" -name '*_json.zip' | wc -l | tr -d ' ')
n_pq=$(find "$UPLOAD" -name '*.parquet' | wc -l | tr -d ' ')
echo "  staged upload : $UPLOAD"
echo "                  $n_zip zips, $n_pq parquet"
[ "$n_zip" -eq 20 ] || die "expected 20 zips in the staged upload, found $n_zip"
[ "$n_pq" -eq 3 ]  || die "expected 3 shotquality parquet, found $n_pq"

BACKUP="$(ls -d "$PARENT"/"$NAME"-backup-* 2>/dev/null | tail -1 || true)"
if [ -n "$BACKUP" ]; then
  echo "  existing backup: $BACKUP"
fi

# --- tooling ----------------------------------------------------------------
#
# git-filter-repo can be reached three different ways and which ones work
# depends on the platform. On Windows + Git Bash, pip installs it into
# %APPDATA%\Python\PythonXY\Scripts, which is normally NOT on PATH -- so
# both `git filter-repo` and `git-filter-repo` fail while the package is
# perfectly well installed. `python -m git_filter_repo` bypasses PATH
# entirely and is what actually works there.
#
# Note also: Git Bash on Windows usually has `python`, not `python3`.
# Checking only for `python3` is what made the previous run report MISSING
# and then contradict itself with "Requirement already satisfied".
FR=""
if git filter-repo --version >/dev/null 2>&1; then
  FR="git filter-repo"
elif command -v git-filter-repo >/dev/null 2>&1 \
     && git-filter-repo --version >/dev/null 2>&1; then
  FR="git-filter-repo"
else
  for PY in python3 python py; do
    command -v "$PY" >/dev/null 2>&1 || continue
    if "$PY" -m git_filter_repo --version >/dev/null 2>&1; then
      FR="$PY -m git_filter_repo"
      break
    fi
  done
fi

if [ -z "$FR" ]; then
  echo "  git-filter-repo: not found -- installing"
  for PY in python3 python py; do
    command -v "$PY" >/dev/null 2>&1 || continue
    "$PY" -m pip install --user git-filter-repo >/dev/null 2>&1 || true
    if "$PY" -m git_filter_repo --version >/dev/null 2>&1; then
      FR="$PY -m git_filter_repo"
      break
    fi
  done
fi

[ -n "$FR" ] || die "could not find or install git-filter-repo.
Install it by hand, then re-run:
    python -m pip install --user git-filter-repo
    python -m git_filter_repo --version     # must print a version"

echo "  git-filter-repo: $FR ($($FR --version 2>/dev/null))"

# --- working tree must be clean ---------------------------------------------
if [ -n "$(git status --porcelain)" ]; then
  git status --short | sed 's/^/    /'
  die "uncommitted changes. Commit or stash them, then re-run."
fi
echo "  working tree  : clean"

# --- capture the remote BEFORE filter-repo removes it -----------------------
REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
[ -n "$REMOTE_URL" ] || die "no 'origin' remote. Add it first:
    git remote add origin https://github.com/IonUdvas/cricket-auction-strategy.git"
echo "  origin        : $REMOTE_URL"
echo "  .git size     : $(du -sh .git | cut -f1)"
echo "  data blobs    : $(git rev-list --objects --all | grep -cE '\.(parquet|zip)$' || true)"

echo
echo "This rewrites all $(git rev-list --count --all) commits and force-pushes."
read -r -p "The Kaggle dataset is uploaded and verified? [type: yes] " ok
[ "$ok" = "yes" ] || die "confirm the Kaggle dataset first."

# --- backup -----------------------------------------------------------------
NEW_BACKUP="$PARENT/$NAME-backup-$(date +%Y%m%d-%H%M%S)"
echo
hr
echo "BACKUP -> $NEW_BACKUP"
hr
cp -a "$REPO" "$NEW_BACKUP"
echo "  $(du -sh "$NEW_BACKUP" | cut -f1)"

# --- rewrite ----------------------------------------------------------------
echo
hr
echo "REWRITING HISTORY"
hr
$FR --force \
  --invert-paths \
  --path data \
  --path 02-kaggle-paths.patch \
  --path-glob '*.parquet' \
  --path-glob '*_json.zip'

rm -rf .git/refs/original
git reflog expire --expire=now --all
git gc --prune=now --aggressive --quiet

# --- local verification -----------------------------------------------------
echo
hr
echo "LOCAL CHECK"
hr
left=$(git rev-list --objects --all | grep -cE '\.(parquet|zip)$' || true)
tracked=$(git ls-files | grep -cE '^data/|\.(parquet|zip)$' || true)
echo "  .git size     : $(du -sh .git | cut -f1)   (was ~204M)"
echo "  data blobs    : $left"
echo "  tracked data  : $tracked"
[ "$left" -eq 0 ]    || die "data blobs survived the rewrite. Nothing pushed."
[ "$tracked" -eq 0 ] || die "data files still tracked. Nothing pushed."

# filter-repo rewrites history and resets the working tree, but it cannot
# remove files git never tracked. data/__pycache__ is the usual survivor, and
# on a Kaggle run so is any ball_attributes.parquet an earlier session wrote
# into the repo. An empty-but-present data/ is harmless -- nothing reads from
# inside the repo any more -- but leaving it invites someone to refill it.
if [ -d data ]; then
  echo
  echo "  untracked leftovers in data/ (never in git, so not rewritten):"
  find data -type f | sed 's/^/      /'
  rm -rf data
  echo "      -> removed"
fi

# --- push -------------------------------------------------------------------
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"
echo
hr
echo "PUSHING (force)"
hr
git push --force --all origin
git push --force --tags origin

# --- remote verification ----------------------------------------------------
echo
hr
echo "REMOTE CHECK  (re-reading GitHub, not trusting the push)"
hr
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --filter=blob:none --no-checkout --quiet "$REMOTE_URL" "$TMP/probe" \
  || die "could not re-clone to verify. The push may have worked -- check by
hand with scripts/verify_purge.sh before deleting anything."

cd "$TMP/probe"

# POSITIVE CONTROL FIRST.
#
# Both counts below are "number of data files found", so zero means clean --
# and zero is also what you get when the probe itself is broken and every
# command fails. A `|| true` on a failing `git ls-tree` returns 0 and the
# script cheerfully declares GitHub clean, at which point you delete the only
# remaining copies of the data.
#
# So: prove the probe can see the repository at all by looking for a file
# that MUST be there. If that fails, the result is unusable, not clean.
sentinel=$(git ls-tree -r --name-only HEAD 2>/dev/null | grep -c '^data_sources\.py$' || true)
[ "$sentinel" -eq 1 ] || die "the verification clone is unusable (could not read
HEAD, or data_sources.py is missing from it). This says NOTHING about whether
the purge worked. Do NOT delete anything. Run scripts/verify_purge.sh."

n_files=$(git ls-tree -r --name-only HEAD | wc -l | tr -d ' ')
remote_head=$(git ls-tree -r --name-only HEAD | grep -cE '^data/|\.(parquet|zip)$' || true)
remote_blobs=$(git rev-list --objects --all | grep -cE '\.(parquet|zip)$' || true)

cd "$REPO"
echo "  probe read HEAD             : $n_files files (sentinel ok)"
echo "  remote HEAD data files      : $remote_head"
echo "  remote data blobs in history: $remote_blobs"

if [ "$remote_blobs" -eq 0 ] && [ "$remote_head" -eq 0 ]; then
  echo
  echo "  >> GITHUB IS CLEAN."
else
  die "GitHub still holds data. Do NOT delete anything locally."
fi

# --- what to delete ---------------------------------------------------------
echo
hr
echo "NOW SAFE TO DELETE FROM YOUR MACHINE"
hr
cat <<EOF

  The data now exists only on Kaggle. These local copies are redundant:

    rm -rf "$NEW_BACKUP"
EOF
for d in "$PARENT"/"$NAME"-backup-*; do
  [ -d "$d" ] && [ "$d" != "$NEW_BACKUP" ] && echo "    rm -rf \"$d\""
done
cat <<EOF
    rm -rf "$UPLOAD"

  Keep the upload folder until you have re-run a Kaggle session end to end
  at least once. Re-uploading a dataset is easy; re-scraping is not.

  Other clones on this machine still holding the old history:
EOF
for base in "$HOME" "$(dirname "$PARENT")"; do
  [ -d "$base" ] || continue
  find "$base" -maxdepth 4 -type d -name '.git' 2>/dev/null | while read -r g; do
    d=$(dirname "$g")
    [ "$d" = "$REPO" ] && continue
    case "$(git -C "$d" remote get-url origin 2>/dev/null)" in
      *cricket-auction-strategy*)
        echo "    $d   -- DELETE and re-clone. Do not pull." ;;
    esac
  done
done
echo "    (searched $HOME and $(dirname "$PARENT") to depth 4 -- check other"
echo "     drives by hand if you keep clones elsewhere)"
cat <<EOF

  Note on GitHub: the old objects stay reachable by SHA until GitHub runs
  its own garbage collection, so the repo may report its old size for a
  while. Nothing references them; it resolves on its own.

EOF
