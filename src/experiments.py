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

    out = {
        "n_winners": int(len(won)),
        "winner_within_interval": float(won["within_interval"].mean()),
        "winner_medAE": float(error.abs().median()),
        "winner_meanAE": float(error.abs().mean()),
        "winner_median_log_ratio": float(np.median(log_ratio)),
        "winner_mad_log_ratio": float(np.median(np.abs(log_ratio))),
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

    sigma_ceiling = (
        training_module.config.get("model", {}).get("sigma_max", 1.5)
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
