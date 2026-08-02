#!/usr/bin/env bash
#
# Answer three questions, with evidence rather than inference:
#
#   1. Is the data still in this repo's history?
#   2. Is the data still on disk anywhere nearby (backups, other clones)?
#   3. Does the remote actually have the rewritten history, or is
#      "Everything up-to-date" telling you something else?
#
# Read-only. Changes nothing.
#
#   ./scripts/verify_purge.sh
#
set -uo pipefail

# Resolve the repo from git itself, not from where this script happens to
# live -- so it still works if you copy it somewhere else to run it.
REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO" ]; then
  echo "Not inside a git repository. cd into your clone and re-run."
  exit 1
fi
cd "$REPO"

hr() { printf '%s\n' "------------------------------------------------------------"; }

echo "repo: $REPO"
hr

# --- 1. HISTORY -------------------------------------------------------------
echo "1. IS THE DATA STILL IN GIT HISTORY?"
echo
echo "   .git size: $(du -sh .git 2>/dev/null | cut -f1)"
echo "   (was 204M before the purge; if it is still ~200M, the purge did not"
echo "    take effect in THIS directory)"
echo
echo "   Data files tracked in HEAD:"
n_head=$(git ls-files | grep -Ec '\.(parquet|zip)$|^data/' || true)
if [ "$n_head" -eq 0 ]; then
  echo "     none"
else
  git ls-files | grep -E '\.(parquet|zip)$|^data/' | sed 's/^/     /'
fi
echo
echo "   Data blobs anywhere in history (all branches, all commits):"
n_hist=$(git rev-list --objects --all 2>/dev/null \
  | grep -Ec '\.(parquet|zip)$' || true)
echo "     $n_hist objects"
echo
echo "   10 largest objects still in history:"
git rev-list --objects --all 2>/dev/null \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' 2>/dev/null \
  | awk '$1=="blob" {print $3, $4}' \
  | sort -rn | head -10 \
  | awk '{printf "     %8.1f MB  %s\n", $1/1048576, $2}'
hr

# --- 2. ON DISK -------------------------------------------------------------
echo "2. WHERE IS THE DATA ON THIS MACHINE?"
echo
echo "   Backups made by the purge script:"
found=0
for d in "$(dirname "$REPO")"/*-backup-*; do
  [ -d "$d" ] || continue
  found=1
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  nf=$(find "$d/data" -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "     $d  ($sz, $nf files under data/)"
done
[ "$found" -eq 0 ] && echo "     none found next to the repo"
echo
echo "   Staged Kaggle upload folder:"
for d in "$(dirname "$REPO")"/kaggle-upload/*; do
  [ -d "$d" ] || continue
  echo "     $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"
done
echo
echo "   Any data/ directory left inside the repo:"
if [ -d "$REPO/data" ]; then
  echo "     $REPO/data EXISTS ($(find "$REPO/data" -type f | wc -l | tr -d ' ') files)"
else
  echo "     none"
fi
hr

# --- 3. THE REMOTE ----------------------------------------------------------
echo "3. WHAT DOES THE REMOTE ACTUALLY HAVE?"
echo
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "   NO 'origin' REMOTE."
  echo "   git filter-repo deletes it deliberately, to stop an accidental"
  echo "   push of rewritten history. If you saw 'Everything up-to-date',"
  echo "   you did not push from this directory."
  echo
  echo "   Re-add it with:"
  echo "     git remote add origin https://github.com/IonUdvas/cricket-auction-strategy.git"
else
  echo "   origin: $(git remote get-url origin)"
  echo
  local_sha=$(git rev-parse HEAD)
  branch=$(git rev-parse --abbrev-ref HEAD)
  remote_sha=$(git ls-remote origin "refs/heads/$branch" 2>/dev/null | cut -f1)
  echo "   local  $branch : $local_sha"
  echo "   remote $branch : ${remote_sha:-<branch not on remote>}"
  echo
  if [ -z "$remote_sha" ]; then
    echo "   >> The branch does not exist on the remote."
  elif [ "$local_sha" = "$remote_sha" ]; then
    echo "   >> MATCH. The remote has this exact history."
    if [ "$n_hist" -eq 0 ]; then
      echo "      And this history has no data blobs, so the remote is clean."
    else
      echo "      But this history STILL CONTAINS $n_hist data blobs --"
      echo "      so the purge did not work, and the remote is not clean."
    fi
  else
    echo "   >> MISMATCH. The remote is on a DIFFERENT commit."
    echo "      A plain 'git push' would have been REJECTED, not reported as"
    echo "      'Everything up-to-date' -- so that message came from a"
    echo "      different directory, or a different branch."
    echo
    echo "      To publish this history:"
    echo "        git push --force --all origin"
    echo "        git push --force --tags origin"
  fi
fi
hr

# --- 4. OTHER CLONES --------------------------------------------------------
echo "4. OTHER CLONES ON THIS MACHINE"
echo
echo "   Any other copy still holding the old history will push it all back"
echo "   the moment you run 'git push' from it. Searching your home dir..."
find "$HOME" -maxdepth 6 -type d -name '.git' 2>/dev/null \
  | while read -r g; do
      d=$(dirname "$g")
      [ "$d" = "$REPO" ] && continue
      case "$(git -C "$d" remote get-url origin 2>/dev/null)" in
        *cricket-auction-strategy*)
          echo "     $d  (.git = $(du -sh "$g" 2>/dev/null | cut -f1))" ;;
      esac
    done
echo
echo "   Delete and re-clone any listed above. Do NOT pull into them."
hr
echo "done."
