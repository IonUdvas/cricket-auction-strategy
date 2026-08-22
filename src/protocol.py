"""
The held-out evaluation protocol, with the epoch transfer made honest.

WHAT WAS WRONG
--------------
The protocol in the notebooks is:

    stage 1   fit on years < stop_year, early-stop on stop_year,
              record best_epoch
    stage 2   refit on years < report_year for exactly best_epoch
              epochs, early stopping OFF, report on report_year

The stopping edition and the reporting edition are disjoint, which is
the property the protocol exists to guarantee, and that part is right.
What is not right is that `best_epoch` is not comparable between the
two report years, so the two reported models are fit for wildly
different lengths:

    epoch_used, 10 seeds (results-2.ipynb, cell 17)

           min    50%    max
    2025  45.0  144.5  150.0
    2026  20.0   25.5  149.0

A median of 144 epochs against a median of 25. The mu head is
zero-initialised with its bias at log(mu_prior_lakh) = log(50), so
training BEGINS by predicting 50 lakh for every player and learns
deviations from there. A model stopped at epoch 25 has barely left
that prior, and its predictions are compressed toward it.

That compression is the whole of the Livingstone / Inglis story:

    Liam Livingstone  2025 fold (149 ep)   875 actual    869 predicted
    Liam Livingstone  2026 fold ( 30 ep)  1300 actual    124 predicted
    Josh Inglis       2026 fold ( 30 ep)   860 actual     96 predicted

Same player, same feature pipeline, same identity resolution, same
archetype row, same last_salary (present, not missing, for both). The
only thing that differs is how long the model trained. The tails
confirm it: the largest UNDER-predictions are 10/15 from 2026 and the
largest OVER-predictions are 14/15 from 2025 -- one compressed
prediction distribution, not two failure modes. Spearman falls 0.822
-> 0.665 even though medAE improves, because medAE is dominated by the
many cheap winners that a prior-anchored model happens to get right.

WHY THE TWO EDITIONS DISAGREE SO BADLY
--------------------------------------
Two compounding reasons, both fixable here.

1. `valid_loss_weighting: balanced` refits class-balanced sample
   weights on each split's own interval midpoints. configs/default.yaml
   already says, in as many words, that "the reported validation NLL is
   not comparable between the 2025 and 2026 editions". The protocol
   then selects an epoch by minimising exactly that non-comparable
   quantity. 2024 as a stopping edition has 72 sales; 2025 has 182.
   Different strata, different weights, different curve, different
   argmin.

2. A single seed's argmin is noise. The spread WITHIN one report year
   (20 to 149 for 2026) is larger than the gap between the two report
   years' medians. results-2.ipynb already prints the test for this --
   "if max-min approaches the epoch median, the stop season is not
   identifying an epoch and the protocol is transferring noise" -- and
   for 2026 max-min is 129 against a median of 25.5.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
`select_epoch` pools the stage-1 runs across seeds and returns one
epoch per report year, and `run_fold` uses it. Three changes:

  * Epoch selection is scored under UNIFORM validation weighting, so
    the quantity being minimised means the same thing for a 72-sale
    stopping edition and a 182-sale one. This does not change how the
    model is TRAINED, only how the epoch is picked.

  * The epoch is the median across seeds, not one seed's argmin.

  * `epoch_disagreement_factor` reports max/min across seeds, and
    `run_fold` refuses to transfer an epoch silently when the spread
    exceeds `max_spread` -- that is the condition under which the
    number being transferred is noise, and it should stop the run
    rather than land in a results table.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not early-stop stage 2 against the report edition. That would
make the epoch a function of the held-out edition's own loss, i.e. it
would select a hyperparameter on the test set. The disjointness of the
stopping and reporting editions is the one property of the original
protocol that was already correct and it is preserved exactly.
"""

from __future__ import annotations

import copy

import numpy as np


DEFAULT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)

# Selecting the epoch under a weighting that is refitted per split
# makes the selection criterion a function of the split's label
# distribution. For SELECTION only, score unweighted.
SELECTION_WEIGHTING = "uniform"

# max(epoch)/min(epoch) across seeds, above which the stopping edition
# is not identifying an epoch at all. 3.0 is deliberately loose: the
# observed 2026 value is 149/20 = 7.45.
DEFAULT_MAX_SPREAD = 3.0


def _deep_update(base, overrides):
    """Recursive dict merge, overrides winning. Local copy so this
    module does not depend on src.experiments importing cleanly."""
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def stage_one_epochs(
    run,
    report_year,
    seeds,
    years=DEFAULT_YEARS,
    overrides=None,
    base_kwargs=None,
    selection_weighting=SELECTION_WEIGHTING,
):
    """
    Fit the stopping model once per seed and return the best epoch of
    each, as a list.

    `run` is a callable with the signature of
    src.training.run_training_pipeline_with_holdout, wrapped so that
    `overrides` are applied to the module-level config -- i.e. the
    notebook's `_run`.

    The stopping edition is report_year - 1 and the training years are
    everything strictly before it, so neither the stopping edition nor
    the report edition is trained on.
    """
    stop_year = report_year - 1
    train_years = [y for y in years if y < stop_year]

    if not train_years:
        raise ValueError(
            f"report_year={report_year} leaves no training years before "
            f"the stopping edition {stop_year}. The protocol needs at "
            f"least one edition before the one it stops on."
        )

    assert stop_year not in train_years
    assert report_year not in train_years

    epochs = []
    for seed in seeds:
        cfg = copy.deepcopy(overrides) if overrides else {}
        cfg.setdefault("seed", seed)

        # Score the stopping edition unweighted. This is the change
        # that makes 2024-as-stopper and 2025-as-stopper comparable.
        cfg.setdefault("training", {})
        cfg["training"].setdefault(
            "valid_loss_weighting", selection_weighting
        )

        result = run(
            train_years=train_years,
            val_years=[stop_year],
            overrides=cfg,
            **(base_kwargs or {}),
        )

        best = result["history"].get("best_epoch")
        if not best:
            # train() returns best_epoch=None when it never improved
            # or when early stopping was off and nothing was tracked.
            # Falling back to the cap is what the notebook did; keep
            # that, but make it visible rather than implicit.
            best = result["history"].get("epochs_run") or 0
            print(
                f"    stage 1 (report {report_year}, seed {seed}): no "
                f"best_epoch recorded, falling back to {best}"
            )
        epochs.append(int(best))

    return epochs


def epoch_disagreement_factor(epochs):
    """max/min across seeds. 1.0 is perfect agreement."""
    epochs = [e for e in epochs if e and e > 0]
    if not epochs:
        return float("inf")
    return float(max(epochs)) / float(min(epochs))


def select_epoch(epochs, max_spread=DEFAULT_MAX_SPREAD, label=""):
    """
    One epoch from a set of per-seed argmins.

    The median, not the mean: the distribution is bounded below by 1
    and above by the epoch cap, and the observed spreads are heavily
    skewed by seeds that run to the cap.

    Raises when the seeds disagree by more than `max_spread`, because
    at that point the stopping edition is not identifying an epoch and
    transferring one is transferring noise. Pass max_spread=None to
    downgrade this to a warning.
    """
    usable = [int(e) for e in epochs if e and e > 0]
    if not usable:
        raise ValueError(
            f"{label}: no usable stage-1 epochs. Every stage-1 run "
            f"returned best_epoch of None or 0."
        )

    factor = epoch_disagreement_factor(usable)
    chosen = int(np.median(usable))

    message = (
        f"{label}: epochs across {len(usable)} seed(s) "
        f"{sorted(usable)} -> median {chosen} "
        f"(max/min = {factor:.2f})"
    )

    if max_spread is not None and factor > max_spread:
        raise ValueError(
            message
            + f"\nThe stopping edition is not identifying an epoch: the "
            f"seed-to-seed spread exceeds max_spread={max_spread}. "
            f"Transferring one of these numbers to stage 2 transfers "
            f"noise, and the resulting model is fit for an arbitrary "
            f"length. Raise the seed count, or set max_spread=None to "
            f"proceed and report the spread alongside the result."
        )

    print(f"  {message}")
    return chosen


def run_fold(
    run,
    report_year,
    seeds=(0,),
    years=DEFAULT_YEARS,
    overrides=None,
    base_kwargs=None,
    max_spread=DEFAULT_MAX_SPREAD,
    report_seed=None,
):
    """
    The whole protocol for one report year.

    Returns the stage-2 result dict, with two extra keys:

        epoch_from_stop_season   the epoch actually used
        stage_one_epochs         every seed's argmin, for reporting

    `report_seed` is the seed the reported model is fit under; it
    defaults to the first seed. Stage 1 pools all of `seeds`, which is
    the point -- one seed's argmin is what was being transferred
    before.
    """
    stage_one = stage_one_epochs(
        run,
        report_year,
        seeds=seeds,
        years=years,
        overrides=overrides,
        base_kwargs=base_kwargs,
    )

    epoch = select_epoch(
        stage_one,
        max_spread=max_spread,
        label=f"report {report_year}",
    )

    train_years = [y for y in years if y < report_year]
    assert report_year not in train_years

    cfg = copy.deepcopy(overrides) if overrides else {}
    cfg.setdefault(
        "seed", report_seed if report_seed is not None else list(seeds)[0]
    )
    cfg.setdefault("training", {})
    cfg["training"].update({
        "epochs": int(epoch),
        "early_stopping_patience": None,
        "restore_best_weights": False,
    })

    result = run(
        train_years=train_years,
        val_years=[report_year],
        overrides=cfg,
        **(base_kwargs or {}),
    )

    result["epoch_from_stop_season"] = int(epoch)
    result["stage_one_epochs"] = stage_one
    return result
