"""
Multi-seed evaluation for the valuation model.

Why this module exists
----------------------

The validation split is one auction: 1,560 rows, of which 1,433 are
'left' (nobody bid) and only 77 are actual sales with a price. Every
metric worth caring about is computed on those 77 rows.

Re-running ONE config -- the proposed one -- under four different
seeds, changing nothing else, gives:

    best_valid_loss   2.746 .. 2.769   (spread 0.023)
    winner medAE      42.4  .. 68.5    (spread 26 lakh)
    winner spearman   0.609 .. 0.705   (spread 0.096)
    best_epoch        2     .. 31

Now compare that to the spread ACROSS configs at a fixed seed, over a
165x range of model size (513 to 84,083 parameters):

    best_valid_loss   2.688 .. 2.804   (spread 0.116)
    winner medAE      39.7  .. 68.8
    winner spearman   0.622 .. 0.698

The seed spread is the same size as the config spread. That means a
single run cannot tell you whether a change helped: most of what
looks like a hyperparameter effect on this data is seed noise. Any
comparison made from one run per config -- which is how the model has
been tuned so far -- is unreliable.

So: sweep with `run_seeds`, compare medians and spreads, and treat a
difference smaller than the seed spread as no difference.

    from src.experiments import run_seeds, compare

    base = dict(player_template=PT, bid_template=BT,
                train_years=[2018, ..., 2025], val_years=[2026],
                player_role_df=archetype_df_filtered)

    small = run_seeds(base, {"model": {"intrinsic_hidden_dims": [64, 32]}})
    big   = run_seeds(base, {"model": {"intrinsic_hidden_dims": [256, 128, 64]}})

    print(compare({"small": small, "big": big}))
"""

import copy

import numpy as np
import pandas as pd

import src.training as training_module
from src.training import run_training_pipeline_with_holdout


DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _deep_update(base, overrides):
    """Recursive dict merge, so a sweep can set one key without
    restating the whole config block."""

    out = copy.deepcopy(base)

    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value

    return out


def summarize_predictions(preds, history=None):
    """
    The metrics worth reporting, on the rows that carry information.

    `preds["within_interval"].mean()` over the whole frame -- the
    0.311 in the last report -- is dominated by the 1,433 'left'
    rows, whose "interval" is (0.01, basePrice) and whose only
    content is "nobody bid". It moves when the base-price
    distribution moves and barely responds to valuation quality.
    Winner-subset metrics are what to tune against.
    """

    won = preds[preds["winner"]]
    if "auctionPrice" in won.columns:
        won = won[won["auctionPrice"].notna()]

    error = won["predicted_median_value"] - won["auctionPrice"]

    # Ratio error, not absolute: being 200 lakh out on a 2,520 lakh
    # marquee signing and on a 300 lakh squad filler are not the same
    # mistake, and mean absolute error counts them the same. This is
    # also the scale the LogNormal actually models.
    log_ratio = np.log(
        np.clip(won["predicted_median_value"], 1e-3, None)
        / np.clip(won["auctionPrice"], 1e-3, None)
    )

    ################################################################
    # Bias and dispersion are separate numbers and must be computed
    # separately.
    #
    # `winner_mad_log_ratio` was median(|log_ratio|) -- the median
    # absolute log-ratio, taken about ZERO. That is not a median
    # absolute deviation, and it is not a dispersion measure at all:
    # a model that is uniformly 30% high with no scatter whatsoever
    # scores 0.262 on it, identical to a perfectly unbiased model
    # whose errors are +/-30%. It therefore double-counts whatever
    # `winner_median_log_ratio` already reports, and every comparison
    # made on it has been a comparison of bias and spread summed
    # together in unknown proportion.
    #
    # A true MAD is taken about the sample median, which is what the
    # per-subset table in the analysis notebook already computed --
    # so the two disagreed and the paper carried both under one name.
    #
    # Both are emitted now. The old quantity keeps its old definition
    # under a name that says what it is, so previously recorded
    # numbers remain checkable against `winner_median_abs_log_ratio`.
    ################################################################

    median_log_ratio = float(np.median(log_ratio))

    out = {
        "n_winners": int(len(won)),
        "winner_within_interval": float(won["within_interval"].mean()),
        "winner_medAE": float(error.abs().median()),
        "winner_meanAE": float(error.abs().mean()),
        "winner_median_log_ratio": median_log_ratio,
        # Dispersion about the median: |log_ratio - median|, medianed.
        "winner_mad_log_ratio": float(
            np.median(np.abs(log_ratio - median_log_ratio))
        ),
        # The pre-fix quantity, about zero: bias and spread combined.
        "winner_median_abs_log_ratio": float(np.median(np.abs(log_ratio))),
        "winner_spearman": float(
            won["predicted_median_value"].corr(
                won["auctionPrice"], method="spearman"
            )
        ),
        "overall_within_interval": float(preds["within_interval"].mean()),
        "sigma_mean": float(preds["predicted_sigma"].mean()),
        "sigma_max": float(preds["predicted_sigma"].max()),
    }

    ################################################################
    # Sigma saturation.
    #
    # sigma is the width of the predictive distribution. If the model
    # can satisfy the interval likelihood by making every prediction
    # vague rather than by making it right, it will -- and the
    # signature is sigma sitting at whatever ceiling you set.
    #
    # Measured: sigma_max 1.5 -> mean sigma 1.19; ceiling 1.0 -> mean
    # 0.99; ceiling 0.8 -> mean 0.79. It tracks the ceiling. Lowering
    # sigma_max does not stop the hedging, it just relocates it, and
    # validation loss gets worse as you squeeze (2.76 -> 2.87), which
    # says the width is genuinely doing work in the likelihood that
    # the mean is not.
    #
    # This is the number to watch after any change: it should come
    # DOWN on its own as the point estimates improve. If it stays
    # pinned, the model is still hedging.
    ################################################################

    # Prefer the ceiling the MODEL was built with, stamped onto the
    # frame by evaluate_predictions. The config fallback is for
    # frames produced before that change, and it is wrong whenever
    # the config has been restored since the run -- see the note in
    # evaluate_predictions.
    sigma_ceiling = preds.attrs.get("sigma_ceiling")
    if sigma_ceiling is None:
        sigma_ceiling = (
            training_module.config.get("model", {}).get("sigma_max", 1.5)
        )
        print(
            "  NOTE: predictions frame carries no sigma_ceiling; falling "
            "back to the live config's sigma_max "
            f"({sigma_ceiling}). If this run overrode sigma_max and the "
            "override has since been restored, sigma_saturation is being "
            "computed against the wrong denominator."
        )
    out["sigma_ceiling"] = float(sigma_ceiling)
    out["sigma_saturation"] = float(
        preds["predicted_sigma"].mean() / sigma_ceiling
    )

    if history is not None:
        out["best_epoch"] = history.get("best_epoch")
        out["best_valid_loss"] = history.get("best_valid_loss")
        if history.get("train") and history.get("best_epoch"):
            best = history["train"][history["best_epoch"] - 1]["loss"]
            out["train_loss_at_best"] = float(best)
            out["overfit_gap"] = float(
                history["best_valid_loss"] - best
            )
        if history.get("valid"):
            out["final_valid_loss"] = float(history["valid"][-1]["loss"])

    return out


def run_seeds(
    pipeline_kwargs,
    config_overrides=None,
    seeds=DEFAULT_SEEDS,
    verbose=True,
):
    """
    Run the holdout pipeline once per seed and return one row each.

    pipeline_kwargs  : passed straight to
                       run_training_pipeline_with_holdout
    config_overrides : nested dict merged into the loaded config,
                       e.g. {"model": {"dropout": 0.2}}
    """

    original = copy.deepcopy(training_module.config)
    rows = []

    try:
        for seed in seeds:
            merged = _deep_update(original, config_overrides)
            merged["seed"] = seed

            training_module.config.clear()
            training_module.config.update(merged)

            result = run_training_pipeline_with_holdout(**pipeline_kwargs)

            row = summarize_predictions(
                result["val_predictions"], result["history"]
            )
            row["seed"] = seed
            row["n_params"] = sum(
                p.numel()
                for p in result["model"].parameters()
                if p.requires_grad
            )
            rows.append(row)

            if verbose:
                print(
                    f"  seed {seed}: valid {row.get('best_valid_loss'):.4f} "
                    f"@ epoch {row.get('best_epoch')} | "
                    f"medAE {row['winner_medAE']:.1f} | "
                    f"spearman {row['winner_spearman']:.3f} | "
                    f"sigma sat {row['sigma_saturation']:.2f}"
                )

    finally:
        # Never leave a sweep's config behind for the next caller.
        training_module.config.clear()
        training_module.config.update(original)

    return pd.DataFrame(rows)


def compare(named_frames, metrics=None):
    """
    Median and spread per config, side by side.

    Spread is reported because it is the decision rule: if two
    configs' medians differ by less than the within-config spread,
    you have not measured a difference, you have measured a seed.
    """

    metrics = metrics or [
        "best_valid_loss",
        "winner_medAE",
        "winner_mad_log_ratio",
        "winner_spearman",
        "winner_within_interval",
        "sigma_saturation",
        "best_epoch",
        "n_params",
    ]

    rows = []

    for name, frame in named_frames.items():
        row = {"config": name, "n_seeds": len(frame)}
        for metric in metrics:
            if metric not in frame.columns:
                continue
            values = frame[metric].astype(float)
            row[f"{metric}_median"] = round(float(values.median()), 4)
            row[f"{metric}_spread"] = round(
                float(values.max() - values.min()), 4
            )
        rows.append(row)

    return pd.DataFrame(rows)


####################################################################
# The auction adjustment
#
# The model's central claim is a decomposition: an intrinsic valuation
# answering "who do we like", and a state-dependent multiplier phi
# answering "how desperate are we". That claim is only worth making if
# phi does something -- a model that learns log_phi ~ 0 everywhere has
# an auction network that is decoration, and one that learns a constant
# non-zero log_phi has merely rescaled mu.
#
# So the reportable facts about phi are, in order:
#   1. Does it vary at all?              (spread, and share at the bound)
#   2. Does it vary with auction state?  (trend across progress)
#   3. Does it vary the way theory says? (higher early, lower late)
#
# Only (3) is the interesting claim, and it is only interesting after
# (1) and (2) survive. Reporting a trend without first reporting the
# spread invites the reader to assume a range that may not exist.
####################################################################

def log_phi_report(preds, n_bins=5, max_log_phi=None, by_year=False):
    """
    Summarise the learned auction adjustment.

    preds : a predictions frame from evaluate_predictions, carrying
            predicted_log_phi and at least one progress column.
    max_log_phi : the tanh bound from the config, if you want the
            saturation diagnostic. The adjustment is bounded by
            max_log_phi * tanh(.), so mass piling up at +/- the bound
            means the bound is binding and the reported range is an
            artefact of the ceiling rather than a finding.

    Returns (summary_dict, progress_table).
    """
    import numpy as _np
    import pandas as _pd

    if "predicted_log_phi" not in preds.columns:
        raise ValueError(
            "predicted_log_phi is absent. It is added by "
            "evaluate_predictions; a frame built before that change must "
            "be regenerated, or derived as "
            "predicted_mu_effective - predicted_mu."
        )

    lp = preds["predicted_log_phi"].to_numpy(dtype=float)

    # The bound the model was actually built with, if the caller did
    # not name one. Same reasoning as sigma_ceiling above: passing
    # T.config["model"]["max_log_phi"] by hand is right only while the
    # config still matches the run.
    if max_log_phi is None:
        max_log_phi = preds.attrs.get("max_log_phi")

    summary = {
        "n_rows": int(len(lp)),
        "log_phi_mean": float(_np.mean(lp)),
        "log_phi_sd": float(_np.std(lp)),
        "log_phi_min": float(_np.min(lp)),
        "log_phi_p05": float(_np.percentile(lp, 5)),
        "log_phi_median": float(_np.median(lp)),
        "log_phi_p95": float(_np.percentile(lp, 95)),
        "log_phi_max": float(_np.max(lp)),
        # phi at the median, as a multiplier -- the interpretable form.
        "phi_median": float(_np.exp(_np.median(lp))),
        "phi_p05": float(_np.exp(_np.percentile(lp, 5))),
        "phi_p95": float(_np.exp(_np.percentile(lp, 95))),
    }

    if max_log_phi:
        # Within 1% of the bound counts as saturated. If this is more
        # than a few percent the tanh is clipping and the range below
        # is a property of the ceiling, not of the auction.
        at_bound = _np.mean(_np.abs(lp) > 0.99 * max_log_phi)
        summary["max_log_phi"] = float(max_log_phi)
        summary["frac_at_bound"] = float(at_bound)

    # ---------------------------------------------------------------
    # Trend across the auction.
    # ---------------------------------------------------------------

    progress_col = next(
        (c for c in ("players_completed", "auction_order", "players_remaining")
         if c in preds.columns),
        None,
    )
    if progress_col is None:
        return summary, None

    frame = preds[[progress_col, "predicted_log_phi"]].copy()
    if progress_col == "players_remaining":
        # Flip so that larger always means "later in the auction".
        frame[progress_col] = -frame[progress_col]
    if by_year and "auction_year" in preds.columns:
        frame["auction_year"] = preds["auction_year"].to_numpy()

    frame["progress_bin"] = _pd.qcut(
        frame[progress_col], q=n_bins, labels=False, duplicates="drop"
    )

    group = ["auction_year", "progress_bin"] if "auction_year" in frame else ["progress_bin"]
    table = (
        frame.groupby(group)["predicted_log_phi"]
        .agg(n="size", mean="mean", sd="std", median="median")
        .reset_index()
    )
    table["phi_median"] = _np.exp(table["median"])

    # Rank correlation rather than a slope: the theoretical claim is
    # monotone ("higher earlier, lower later"), not linear, and a
    # Pearson slope on a bounded tanh output is not the right test.
    from scipy.stats import spearmanr
    rho, p = spearmanr(frame[progress_col], frame["predicted_log_phi"])
    summary["progress_column"] = progress_col
    summary["spearman_progress_vs_log_phi"] = float(rho)
    summary["spearman_p"] = float(p)

    return summary, table
