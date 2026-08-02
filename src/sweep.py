"""
Fast sweeps: build the data once, train many times, optionally in parallel.

The problem
-----------
`run_training_pipeline_with_holdout` used to rebuild everything on every
call: a fresh PlayerFeatureContext over ~2.4M deliveries, identity resolution
across all nine rosters, and nine auction replays -- all before a single
gradient step. None of that depends on the seed, on `model.*`, or on
`training.*`.

So a 36-config x 5-seed sweep paid for 180 full data builds when it needed
one. If the build is B and the training is T, the sweep cost 180*(B+T) when
the floor is B + 180*T. With B around 3 minutes and T around 20 seconds --
measure yours, `time_stages()` below does it -- that is 3 hours 20 minutes
against 61 minutes, and the 3 hours is 90% waiting for the same nine auction
replays to happen again.

Which is why the first thing to do about a slow sweep is NOT to parallelize
it. Parallelism divides the 3 hours across cores. Caching deletes most of it
outright, and it composes with parallelism afterwards.

    build once ..... ~3 min      (unavoidable, one time)
    180 trainings .. divide by n_jobs

Usage
-----
    from src.sweep import SweepRunner

    runner = SweepRunner(
        train_years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        val_years=[2026],
        player_role_df=archetype_df_filtered,
    )

    runner.check_rng_neutral()          # once, before trusting the cache

    results = {}
    for name, dims in [("linear", []), ("[32]", [32]),
                       ("[64,32]", [64, 32]), ("[128,64]", [128, 64])]:
        results[name] = runner.run_seeds(
            {"model": {"intrinsic_hidden_dims": dims}}, seeds=(0, 1, 2, 3, 4))

    from src.experiments import compare
    print(compare(results))

The returned frames are the same shape `src.experiments.run_seeds` returns,
so `compare()` and any table you already have stay directly comparable.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd

import src.training as training_module
from src.training import prepare_holdout_data, train_from_prepared
from src.experiments import _deep_update, summarize_predictions


# Config keys that reach stage 1. Anything else can change freely without
# invalidating a cached build.
#
# `scale_features` is NOT here on purpose: it lives in stage 2, so toggling
# it costs a scaler fit rather than a full rebuild.
#
# `verify` / `verify_strict` are not here either, and that one is a genuine
# trade rather than an oversight. They run inside stage 1 but do not change
# a single value in the frame, so including them would throw away a good
# cached build to re-run checks. The cost is that turning verify_strict on
# mid-sweep will not re-check an already-cached build -- clear the cache if
# that is what you are after.
DATA_KEYS_AFFECTING_BUILD = (
    "player_context_columns",
    "max_role_cardinality",
    "drop_role_identity_columns",
    "drop_role_leaky_columns",
)


def _build_key(pipeline_kwargs, cfg):
    """A hashable identity for one stage-1 build."""
    data_cfg = (cfg.get("data") or {})
    parts = {
        "years": (tuple(pipeline_kwargs.get("train_years") or ()),
                  tuple(pipeline_kwargs.get("val_years") or ())),
        "competitions": tuple(pipeline_kwargs.get("competitions") or ()),
        "bbb_dir": pipeline_kwargs.get("bbb_dir"),
        "resolution": pipeline_kwargs.get("resolution"),
        "player_template": pipeline_kwargs.get("player_template"),
        "bid_template": pipeline_kwargs.get("bid_template"),
        "data": {k: data_cfg.get(k) for k in DATA_KEYS_AFFECTING_BUILD},
    }
    # The role table is derived from player_role_df, so its COLUMNS matter.
    # Its contents are a fixed curated table; if you edit it mid-sweep, clear
    # the cache yourself. Hashing 700 rows on every lookup is not worth it.
    role_df = pipeline_kwargs.get("player_role_df")
    if role_df is not None:
        parts["role_columns"] = tuple(role_df.columns)
        parts["role_nrows"] = len(role_df)
    return json.dumps(parts, sort_keys=True, default=str)


class SweepRunner:
    """
    Holds one or more prepared builds and runs training against them.

    Everything passed to the constructor is a stage-1 input. Per-run config
    goes to `run_seeds` as overrides, exactly as with
    `src.experiments.run_seeds`.
    """

    def __init__(self, verbose=True, **pipeline_kwargs):
        self.pipeline_kwargs = pipeline_kwargs
        self.verbose = verbose
        self._cache = {}
        self._build_seconds = None

    # -- stage 1 -----------------------------------------------------------
    def prepared(self, cfg=None):
        """The prepared build for `cfg`, building it if it is not cached."""
        cfg = cfg or training_module.config
        key = _build_key(self.pipeline_kwargs, cfg)
        if key not in self._cache:
            if self.verbose:
                print(f"  building data (cache miss, {len(self._cache)} "
                      f"build(s) already held)...", flush=True)
            t0 = time.time()
            self._cache[key] = prepare_holdout_data(**self.pipeline_kwargs)
            self._build_seconds = time.time() - t0
            if self.verbose:
                print(f"  build took {self._build_seconds:.0f}s", flush=True)
        return self._cache[key]

    def clear(self):
        self._cache.clear()

    # -- the sanity check that makes the cache trustworthy -----------------
    def check_rng_neutral(self):
        """
        Prove that stage 1 draws no random numbers.

        If it does not, then seeding before stage 2 (what the cached path
        does) and seeding before stage 1 (what the old monolith did) leave the
        RNGs in the same state, and cached results are bit-identical to
        uncached ones. If it DOES draw, cached numbers are still internally
        consistent -- every config sees the same treatment -- but they will
        not match a previously recorded uncached run, and you should say so
        when comparing against your existing table.

        This is worth one build to settle, because the alternative is
        wondering about it for the rest of the sweep.
        """
        import torch

        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        before = (random.getstate(), np.random.get_state(),
                  torch.random.get_rng_state().clone())

        self.prepared()

        after = (random.getstate(), np.random.get_state(),
                 torch.random.get_rng_state())

        ok_py = before[0] == after[0]
        ok_np = all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                    for a, b in zip(before[1], after[1]))
        ok_pt = torch.equal(before[2], after[2])

        print(f"  python random untouched : {ok_py}")
        print(f"  numpy  random untouched : {ok_np}")
        print(f"  torch  random untouched : {ok_pt}")
        if ok_py and ok_np and ok_pt:
            print("  -> cached sweeps are bit-identical to uncached ones.")
        else:
            print("  -> stage 1 consumes randomness. Cached results stay")
            print("     internally consistent, but will not reproduce your")
            print("     earlier uncached numbers exactly. Re-run the baseline")
            print("     through this runner before comparing.")
        return ok_py and ok_np and ok_pt

    # -- timing ------------------------------------------------------------
    def time_stages(self, seed=0):
        """
        Measure B and T once, so you can budget the sweep instead of guessing.
        Prints the projected cost of a sweep at various sizes.
        """
        self.clear()
        t0 = time.time()
        prep = self.prepared()
        build = time.time() - t0

        t0 = time.time()
        self._one_run(prep, seed, None)
        train = time.time() - t0

        print(f"\n  stage 1 (build)   : {build:6.1f}s")
        print(f"  stage 2 (train)   : {train:6.1f}s")
        print(f"\n  {'runs':>6} {'old (rebuild each)':>20} {'cached':>10} "
              f"{'cached + 4 jobs':>16}")
        for n in (20, 100, 180, 200):
            old = n * (build + train)
            new = build + n * train
            par = build + n * train / 4
            print(f"  {n:6d} {old/60:17.0f}m {new/60:9.0f}m {par/60:15.0f}m")
        return {"build_seconds": build, "train_seconds": train}

    # -- stage 2 -----------------------------------------------------------
    def _one_run(self, prep, seed, overrides):
        original = copy.deepcopy(training_module.config)
        try:
            merged = _deep_update(original, overrides)
            merged["seed"] = seed
            training_module.config.clear()
            training_module.config.update(merged)
            return train_from_prepared(prep, seed=seed)
        finally:
            training_module.config.clear()
            training_module.config.update(original)

    def run_seeds(self, config_overrides=None, seeds=(0, 1, 2, 3, 4),
                  n_jobs=1):
        """
        Same contract as src.experiments.run_seeds: one row per seed.

        n_jobs > 1 runs seeds in parallel worker processes. See
        `run_parallel` for what that does and does not buy you.
        """
        # The build must happen in the parent, once, before any fork -- both
        # so it is shared and so a cache miss does not happen n_jobs times
        # concurrently.
        cfg = _deep_update(training_module.config, config_overrides)
        prep = self.prepared(cfg)

        if n_jobs > 1:
            out = _run_parallel(self, prep, config_overrides, seeds, n_jobs)
            if out is not None:
                return out

        rows = []
        for seed in seeds:
            result = self._one_run(prep, seed, config_overrides)
            rows.append(_row(result, seed))
            if self.verbose:
                r = rows[-1]
                print(f"  seed {seed}: valid {r.get('best_valid_loss'):.4f} "
                      f"@ epoch {r.get('best_epoch')} | "
                      f"medAE {r['winner_medAE']:.1f} | "
                      f"spearman {r['winner_spearman']:.3f} | "
                      f"sigma sat {r['sigma_saturation']:.2f}", flush=True)
        return pd.DataFrame(rows)


def _row(result, seed):
    row = summarize_predictions(result["val_predictions"], result["history"])
    row["seed"] = seed
    row["n_params"] = sum(p.numel() for p in result["model"].parameters()
                          if p.requires_grad)
    return row


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
#
# Worth reading before turning this on.
#
# The model is ~5,000 parameters over ~15,000 rows. That is far too small to
# saturate a GPU -- the run is dominated by Python-side DataLoader work and
# per-batch kernel launches, not by arithmetic. So:
#
#   * On CPU, parallel workers scale close to linearly up to the core count.
#     Kaggle gives 4 vCPUs, so n_jobs=4 with torch.set_num_threads(1) is the
#     sweet spot. More than that thrashes.
#   * On GPU, several processes share one device fine at this size (a CUDA
#     context is ~300MB against 16GB), but they serialise on kernel launches,
#     so the gain is smaller than on CPU. On a 2xT4 instance, pinning workers
#     to alternating devices helps.
#
# The prepared frames are passed to workers by pickling them once to
# /kaggle/working rather than through the Pool argument, because sending a
# ~15k-row frame per task through the pipe on every job is wasteful and
# `fork` start method is not available everywhere.

_WORKER_STATE = {}


def _worker_init(threads_per_worker, config_snapshot, prep):
    """
    Runs once per worker.

    Under `fork` -- which is what this uses -- `prep` and `config_snapshot`
    arrive through copy-on-write memory rather than a pickle, so the ~15k-row
    frames are not serialised at all and the workers start instantly.
    """
    import torch
    torch.set_num_threads(threads_per_worker)
    _WORKER_STATE["prep"] = prep
    _WORKER_STATE["config"] = config_snapshot


def _worker_run(job):
    seed, overrides = job
    import src.training as tm
    from src.training import train_from_prepared as _tfp

    merged = _deep_update(copy.deepcopy(_WORKER_STATE["config"]), overrides)
    merged["seed"] = seed
    tm.config.clear()
    tm.config.update(merged)

    return _row(_tfp(_WORKER_STATE["prep"], seed=seed), seed)


def _run_parallel(runner, prep, overrides, seeds, n_jobs):
    """
    Run seeds concurrently.

    `fork`, not `spawn`, and that is not a detail.
    -----------------------------------------------------------------
    `spawn` re-imports __main__ in each child, and a Jupyter kernel has no
    importable __main__ -- so on Kaggle, which is the only place this runs,
    spawn raises "An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase" before a single job
    starts. fork works in a notebook and additionally shares the prepared
    frames copy-on-write, so there is nothing to pickle.

    The cost of fork is that it is unsafe to touch CUDA in a child if the
    parent has already initialised a CUDA context. Workers are therefore
    pinned to CPU. That is not the sacrifice it sounds like: at ~5,000
    parameters over ~15,000 rows the run is dominated by Python-side
    DataLoader work and kernel-launch latency, not arithmetic, so four CPU
    workers beat one GPU process on wall clock for this model. If you scale
    the model up by an order of magnitude, revisit it.
    """
    import multiprocessing as mp

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("  fork unavailable on this platform -- running sequentially.")
        return None

    # Pin workers to CPU before forking, so no child inherits a live CUDA
    # context and none of them contend for the device.
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cfg = copy.deepcopy(training_module.config)
    threads = max(1, (os.cpu_count() or 4) // max(1, n_jobs))
    jobs = [(s, overrides) for s in seeds]

    try:
        with ctx.Pool(
            processes=min(n_jobs, len(jobs)),
            initializer=_worker_init,
            initargs=(threads, cfg, prep),
        ) as pool:
            rows = pool.map(_worker_run, jobs)
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev

    # Restore seed order; Pool.map preserves it, but be explicit.
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)