"""
The held-out evaluation protocol: fixing the epoch transfer WITHOUT
touching the loss weighting.

WHAT THIS DOES NOT DO ANY MORE
------------------------------
An earlier version of this module changed the validation weighting
(`valid_loss_weighting: uniform`) and then proposed selecting epochs
on a winner-only NLL. Both are reverted and neither should come back.
The reasons are worth writing down so they do not get re-invented:

  * `uniform` was measured on the real data and selected epoch 1.
    Train loss fell normally (4.14 -> 3.69 -> 3.15 -> 2.88 ...) while
    the uniform-weighted validation number ROSE from 1.2870 at epoch
    1 to 2.1662 at epoch 5, and had only recovered to 1.6777 by epoch
    30, at which point patience ended the run. It picked the
    least-trained model available.

  * winner-only selection was the wrong objective outright. The
    model's job is a valuation for EVERY (player, team) pair -- that
    is what makes it possible to say both what range a player will
    sell in AND which franchises are plausibly in the race. Most of
    those pairs are left-censored by construction: nine of ten teams
    do not buy any given player. The left rows are the majority of
    the TARGET, not noise to be selected away from. Scoring only the
    ~11% of rows with a completed sale optimises the smallest and
    least representative part of the problem.

The class-balanced weighting is what makes both objectives trainable
at once: it stops the 88% left majority from drowning out the 1,847
informative rows without discarding either. It stays exactly as it
was.

WHAT IS ACTUALLY WRONG WITH THE EPOCH TRANSFER
----------------------------------------------
Under the shipped (balanced) weighting the validation curve is smooth
and well behaved -- it descends, then flattens. The problem is that
where it flattens depends on which edition does the stopping, and the
protocol transfers that number to a different edition:

    epoch_used, 10 seeds (results-2.ipynb, cell 17)

           min    50%    max
    2025  45.0  144.5  150.0
    2026  20.0   25.5  149.0

The 2025 model is fit for a median of 144 epochs and the 2026 model
for 25. The mu head starts at log(mu_prior_lakh) = log(50) and learns
deviations from there, so a model stopped at 25 epochs is still close
to predicting 50 lakh for everyone -- which is the Livingstone /
Inglis story (875 -> 869 predicted in the 144-epoch fold; 1300 -> 124
in the 25-epoch one).

The flat region is genuinely flat, so the argmin inside it is noise;
that is why the spread is 20-149 within a single edition. It also
means "the best epoch" is not a well-posed question. What is needed
is an epoch BUDGET that is the same for both editions, chosen once,
on evidence.

Evidence already in hand, from the run this module was rewritten
after: report-2025 fit at 74 epochs scored medAE 68.9 / rho 0.842,
against 75.5 / 0.822 for the original 149-epoch fit. More epochs is
not monotonically better; 25 is clearly too few and 149 is past the
useful point.

WHAT THIS MODULE OFFERS
-----------------------
`run_fold(..., fixed_epochs=N)` -- the recommended path. Skips stage
1 entirely and trains stage 2 for exactly N epochs on every edition.
Both editions are then fit identically, which is the property that
was missing, at one run per edition instead of one plus a stage 1 per
seed.

`sweep_epoch_budget` -- how to choose N honestly, on BOTH objectives
at once: winner-subset accuracy alongside whole-frame interval
coverage, so a budget cannot be picked that sharpens the 77-182
winners while degrading the ~1,400 per-team valuation rows.

`run_fold(...)` without `fixed_epochs` keeps the original two-stage
behaviour, with the epoch pooled across seeds rather than taken from
one, for anyone who wants the old protocol de-noised rather than
replaced. It changes no weighting: stage 1 runs exactly as the repo
already runs it.

WHAT IS PRESERVED
-----------------
Stage 2 never early-stops against the reporting edition. That would
select a hyperparameter on the test set. The disjointness of stopping
and reporting editions was always correct and is untouched.
"""

from __future__ import annotations

import copy

import numpy as np


DEFAULT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)

# max(epoch)/min(epoch) across seeds, above which the stopping edition
# is not identifying an epoch at all. Only consulted on the two-stage
# path; `fixed_epochs` sidesteps the question entirely.
DEFAULT_MAX_SPREAD = 3.0


def _cfg_with(overrides, **training):
    cfg = copy.deepcopy(overrides) if overrides else {}
    if training:
        cfg.setdefault("training", {})
        cfg["training"].update(training)
    return cfg


def stage_one_epochs(run, report_year, seeds, years=DEFAULT_YEARS,
                     overrides=None, base_kwargs=None):
    """
    Fit the stopping model once per seed, return each run's best epoch.

    No weighting override of any kind: stage 1 runs under whatever
    configs/default.yaml says, which is `balanced`.
    """
    stop_year = report_year - 1
    train_years = [y for y in years if y < stop_year]
    if not train_years:
        raise ValueError(
            f"report_year={report_year} leaves no training years before "
            f"the stopping edition {stop_year}."
        )

    epochs = []
    for seed in seeds:
        cfg = _cfg_with(overrides)
        cfg.setdefault("seed", seed)
        result = run(train_years=train_years, val_years=[stop_year],
                     overrides=cfg, **(base_kwargs or {}))

        best = result["history"].get("best_epoch")
        if not best:
            # history["train"] has one entry per epoch actually run,
            # whatever ended the run -- a real count, unlike the
            # "epochs_run" key an earlier version of this fallback
            # read, which train() has never set (so it silently fell
            # back to 0).
            best = len(result["history"].get("train", [])) or 0
            print(f"    stage 1 (report {report_year}, seed {seed}): no "
                  f"best_epoch recorded, using {best} epochs actually run")
        epochs.append(int(best))

    return epochs


def epoch_disagreement_factor(epochs):
    """max/min across seeds. 1.0 is perfect agreement."""
    usable = [e for e in epochs if e and e > 0]
    return float(max(usable)) / float(min(usable)) if usable else float("inf")


def select_epoch(epochs, max_spread=DEFAULT_MAX_SPREAD, label=""):
    """
    One epoch from a set of per-seed argmins: the median.

    Raises when the seeds disagree by more than `max_spread` -- at
    that point the stopping edition is not identifying an epoch and
    transferring one transfers noise. max_spread=None downgrades this
    to a printed line.
    """
    usable = [int(e) for e in epochs if e and e > 0]
    if not usable:
        raise ValueError(f"{label}: no usable stage-1 epochs.")

    factor = epoch_disagreement_factor(usable)
    chosen = int(np.median(usable))
    message = (f"{label}: epochs across {len(usable)} seed(s) "
               f"{sorted(usable)} -> median {chosen} "
               f"(max/min = {factor:.2f})")

    if max_spread is not None and factor > max_spread:
        raise ValueError(
            message + f"\nSeed spread exceeds max_spread={max_spread}: the "
            f"stopping edition is not identifying an epoch, so this number "
            f"is noise. Prefer run_fold(..., fixed_epochs=N) -- one budget "
            f"for every edition -- or pass max_spread=None and report the "
            f"spread alongside the result."
        )

    print(f"  {message}")
    return chosen


def run_fold(run, report_year, seeds=(0,), years=DEFAULT_YEARS,
             overrides=None, base_kwargs=None,
             fixed_epochs=None, max_spread=DEFAULT_MAX_SPREAD,
             report_seed=None):
    """
    The protocol for one report year.

    fixed_epochs : int or None
        RECOMMENDED. Train stage 2 for exactly this many epochs and
        skip stage 1 entirely. Every edition is then fit identically,
        which is the property the original protocol lacked, and the
        cost is one run per edition. Choose the value with
        `sweep_epoch_budget` rather than by taste.

        None keeps the original two-stage transfer, with the epoch
        pooled across `seeds` instead of taken from one of them.

    Returns the stage-2 result dict plus:
        epoch_from_stop_season   epochs actually used
        stage_one_epochs         every seed's argmin, or None when
                                 fixed_epochs was given
    """
    if fixed_epochs is not None:
        epoch, stage_one = int(fixed_epochs), None
        print(f"  report {report_year}: fixed budget of {epoch} epochs "
              f"(stage 1 skipped; both editions fit identically)")
    else:
        stage_one = stage_one_epochs(run, report_year, seeds=seeds,
                                     years=years, overrides=overrides,
                                     base_kwargs=base_kwargs)
        epoch = select_epoch(stage_one, max_spread=max_spread,
                             label=f"report {report_year}")

    train_years = [y for y in years if y < report_year]
    assert report_year not in train_years

    cfg = _cfg_with(overrides, epochs=int(epoch),
                    early_stopping_patience=None,
                    restore_best_weights=False)
    cfg.setdefault("seed",
                   report_seed if report_seed is not None else list(seeds)[0])

    result = run(train_years=train_years, val_years=[report_year],
                 overrides=cfg, **(base_kwargs or {}))
    result["epoch_from_stop_season"] = int(epoch)
    result["stage_one_epochs"] = stage_one
    return result


def sweep_epoch_budget(run, report_years, budgets, seeds=(0,),
                       years=DEFAULT_YEARS, overrides=None,
                       base_kwargs=None, summarize=None):
    """
    Fit each report year at each epoch budget and report BOTH
    objectives, so the budget is chosen on evidence.

    This exists because the two things the model is for pull in
    different directions, and a single metric hides that:

      * winner_* columns -- accuracy on completed sales, i.e. "what
        will this player go for". 77-182 rows per edition.

      * overall_within_interval, and the left rows it is mostly made
        of -- whether the per-team valuations still land inside their
        observed bounds, i.e. "which franchises were plausibly in the
        race". ~1,400 rows per edition, and the majority of what the
        model actually outputs.

    A budget that improves the first while degrading the second is
    not an improvement for this project, and only a table carrying
    both columns makes that visible.

    `summarize` defaults to src.experiments.summarize_predictions.
    Returns a tidy DataFrame: one row per (report_year, seed, budget).
    """
    import pandas as pd

    if summarize is None:
        from src.experiments import summarize_predictions as summarize

    rows = []
    for budget in budgets:
        for year in report_years:
            for seed in seeds:
                r = run_fold(run, year, seeds=(seed,), years=years,
                             overrides=overrides, base_kwargs=base_kwargs,
                             fixed_epochs=budget, report_seed=seed)
                row = summarize(r["val_predictions"], r["history"])
                row.update(year=year, seed=seed, budget=budget)
                rows.append(row)
                print(f"    budget {budget:3d} | {year} seed {seed}: "
                      f"medAE {row.get('winner_medAE', float('nan')):.1f} | "
                      f"rho {row.get('winner_spearman', float('nan')):.3f} | "
                      f"overall_within_interval "
                      f"{row.get('overall_within_interval', float('nan')):.3f}",
                      flush=True)

    return pd.DataFrame(rows)


########################################################################
# Caching: build each (train_years, val_years) frame once, not once per
# call.
#
# The notebook's `_run` wrapper calls run_training_pipeline_with_holdout,
# which is exactly `prepared = prepare_holdout_data(...); return
# train_from_prepared(prepared, seed=...)` -- and prepare_holdout_data is
# the expensive half: a PlayerFeatureContext over ~2.4M deliveries,
# identity resolution across every roster in train_years+val_years, and
# one auction replay per year (the "Building 2018... Building 2019..."
# minutes in every log in this project). None of that depends on the
# seed or on model.*/training.* config -- see prepare_holdout_data's own
# docstring, which says so and was never wired to a cache until src/
# sweep.py added one.
#
# Every caller in THIS file re-triggers that full build on every single
# seed and every single epoch budget, because `run` is called fresh each
# time. Concretely, for one call to sweep_epoch_budget with 5 budgets and
# REPORT_YEARS=(2025, 2026):
#
#     without caching:  5 budgets x 2 years x 1 seed = 10 full builds
#     with caching:                        2 years            = 2 builds
#
# and the 2 come from there being two distinct (train_years, val_years)
# pairs -- one per report year -- not two per budget. A second seed
# multiplies the first number by 2 and leaves the second unchanged.
#
# make_cached_run wraps src.sweep.SweepRunner, which already implements
# exactly this cache (keyed on years plus the four data.* keys that can
# change the frame's shape -- see DATA_KEYS_AFFECTING_BUILD in
# src/sweep.py). This function only adds the (train_years, val_years)
# -> SweepRunner layer on top, and gives back something with the SAME
# call signature the notebook's hand-written `_run` already has, so it
# is a one-line swap:
#
#     from src.protocol import make_cached_run
#     _run = make_cached_run()
#
# every other cell -- fold(), run_fold(), sweep_epoch_budget() -- is
# unchanged, because they only ever call `run(train_years=, val_years=,
# overrides=, **base_kwargs)` and never inspect what `run` is.
########################################################################

def make_cached_run(verbose=True):
    """
    Build each distinct (train_years, val_years) pair once per Python
    session and reuse it for every seed / epoch budget that asks for
    the same pair.

    Returns a callable with the exact signature the notebook's `_run`
    already has: `run(train_years=, val_years=, overrides=, **kw)` ->
    the same dict run_training_pipeline_with_holdout returns (model,
    history, val_predictions, ...).

    The cache lives on the returned closure (`_run.runners`), not at
    module scope -- call make_cached_run() again for a clean cache
    (e.g. after editing the underlying CSVs) rather than restarting
    the kernel.
    """
    import src.training as training_module
    from src.experiments import _deep_update
    from src.sweep import SweepRunner
    from src.training import train_from_prepared

    runners = {}

    def _run(train_years=None, val_years=None, overrides=None, **base_kwargs):
        key = (tuple(sorted(train_years or ())),
               tuple(sorted(val_years or ())))

        if key not in runners:
            if verbose:
                print(f"  [cache miss] building {key[0][0]}-{key[0][-1]} "
                      f"-> {key[1]} (new SweepRunner)", flush=True)
            runners[key] = SweepRunner(
                train_years=list(train_years or ()),
                val_years=list(val_years or ()),
                verbose=verbose,
                **base_kwargs,
            )
        elif verbose:
            print(f"  [cache hit] {key[0][0]}-{key[0][-1]} -> {key[1]} "
                  f"(reusing build)", flush=True)

        runner = runners[key]

        original = copy.deepcopy(training_module.config)
        try:
            merged = _deep_update(original, overrides)
            training_module.config.clear()
            training_module.config.update(merged)
            prep = runner.prepared(training_module.config)
            return train_from_prepared(prep, seed=merged.get("seed"))
        finally:
            training_module.config.clear()
            training_module.config.update(original)

    _run.runners = runners  # inspect len(_run.runners) for a build count
    return _run

