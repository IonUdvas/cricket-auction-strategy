"""
One call to get a Kaggle session from "repo just cloned" to "ready to train".

    !git clone -q https://github.com/IonUdvas/cricket-auction-strategy.git
    %cd cricket-auction-strategy
    from kaggle_session import prepare
    paths = prepare()

`prepare()` is idempotent and safe to re-run: each stage checks whether its
output already exists and skips if so, so re-running a cell after an
unrelated error does not re-parse 10,000 json files.

What it does, in order
----------------------
1. Puts the repo root on sys.path.  The notebook's working directory is
   /kaggle/working, not the clone, so `import src.training` fails without it.
2. Prints `data_sources.describe()` -- what is mounted and what resolved.
3. Builds bbb into /kaggle/working/bbb if it is not already there.
   (~3-6 minutes, ~10,000 json files, once per session.)
4. Builds ball_attributes.parquet into the same directory if it is not
   already there.  (~2-4 minutes, duckdb.)
5. Returns the resolved paths, so the training call needs no arguments.

Why these two are built rather than stored
------------------------------------------
Both are fully derived from udvasbasak2/ipl-auction-model-inputs.  A stored
copy is a second thing to keep in sync with the zips it came from, and the
failure mode when it drifts is silent: a delivery table that is quietly short
by a season, producing career totals that are wrong for exactly the players
who played most recently.  Rebuilding costs minutes; that bug costs a
retraining cycle and is very hard to see.

If you would rather pay the minutes once
----------------------------------------
Save /kaggle/working/bbb as its own Kaggle dataset at the end of a session and
attach it next time.  data_sources searches /kaggle/working BEFORE
/kaggle/input, so a fresh in-session build still wins over the stored copy --
which is the ordering you want, not the other way round.  Do NOT put it in the
inputs dataset: keeping sources and derivations in one dataset is how they
drift apart without anyone noticing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import data_sources as ds  # noqa: E402


def _run(module, *args):
    """Run a pipeline module in-process-ish, streaming output to the cell."""
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{module} failed with exit code {proc.returncode}. "
            f"Scroll up for the traceback; run data_sources.describe() to see "
            f"what was and was not resolvable."
        )
    print(f"\n[{module} finished in {time.time() - t0:.0f}s]", flush=True)


def prepare(build_bbb=True, build_shot_quality=True, verbose=True):
    """
    Returns a dict of resolved paths:
        bbb_dir, ball_attributes, resolution, archetypes,
        player_template, bid_template
    """
    if verbose:
        print("=" * 72)
        print("DATA")
        print("=" * 72)
        ds.describe()

    # --- stage 1: the ball-by-ball set -------------------------------------
    have_bbb = ds.bbb_dir(required=False)
    if build_bbb and not have_bbb:
        print("\n" + "=" * 72)
        print("BUILDING bbb  (this is the slow one -- once per session)")
        print("=" * 72)
        _run("pipelines.build_bbb")
    elif verbose and have_bbb:
        print(f"\nbbb already built: {have_bbb} -- skipping")

    bbb = ds.bbb_dir()

    # --- stage 2: shot quality ---------------------------------------------
    # Checked by path rather than by ds.ball_attributes_path(), because that
    # searches every mount and would find a stale copy in an attached dataset
    # and skip the build. Here we want to know specifically whether THIS
    # session has produced one next to the deliveries it belongs to.
    attrs_here = os.path.join(bbb, "ball_attributes.parquet")
    if build_shot_quality and not os.path.exists(attrs_here):
        print("\n" + "=" * 72)
        print("BUILDING ball_attributes")
        print("=" * 72)
        _run("pipelines.build_shot_attributes", "--bbb-dir", bbb,
             "--out-dir", bbb)
    elif verbose and os.path.exists(attrs_here):
        print(f"ball_attributes already built: {attrs_here} -- skipping")

    paths = {
        "bbb_dir": bbb,
        "ball_attributes": ds.ball_attributes_path(),
        "resolution": ds.resolution_path(),
        "archetypes": ds.archetypes_path(required=False),
        "player_template": ds.player_template(),
        "bid_template": ds.bid_template(),
    }

    if verbose:
        print("\n" + "=" * 72)
        print("READY")
        print("=" * 72)
        for k, v in paths.items():
            print(f"  {k:18s} {v or 'MISSING'}")
        if not paths["ball_attributes"]:
            print("\n  NOTE: no ball_attributes -- every shot-quality metric "
                  "will be None.\n  That is a degradation, not an error; the "
                  "aggregator reports None rather\n  than inventing zeros.")
        print("\nNext:")
        print("    from src.training import run_training_pipeline_with_holdout")
        print("    import pandas as pd")
        print("    archetypes = pd.read_csv(paths['archetypes'])")
        print("    out = run_training_pipeline_with_holdout(")
        print("        train_years=[2018,2019,2020,2021,2022,2023,2024,2025],")
        print("        val_years=[2026],")
        print("        player_role_df=archetypes,")
        print("    )")

    return paths


if __name__ == "__main__":
    prepare()
