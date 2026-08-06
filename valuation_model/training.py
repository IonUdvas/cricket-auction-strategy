import numpy as np
import torch

from torch.utils.data import DataLoader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_grad_norm=5.0,
):
    """
    Train the model for one epoch.

    Parameters
    ----------
    max_grad_norm : float or None
        Clip the global gradient norm to this value before each
        optimizer step. This is the runaway-update guard for
        pathological samples (replacing the old nll.clamp(max=50),
        which zeroed gradients instead of bounding them -- see
        losses.py). Set to None to disable.

    Returns
    -------
    dict
        Dictionary containing average metrics over the epoch.
    """

    model.train()

    running = None

    for batch in loader:

        ########################################################
        # Move tensors to device
        ########################################################

        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        ########################################################
        # Forward pass
        ########################################################


        output = model(
            batch["player_features"],
            batch["role_features"],
            batch["team"],
            batch["team_state"],
            batch["auction_state"],
        )


        ########################################################
        # Loss
        ########################################################

        stats = criterion(
            mu=output["mu_effective"],
            sigma=output["sigma"],
            lower_bid=batch["lower_bid"],
            upper_bid=batch["upper_bid"],
            observation_type=batch["observation_type"],
            weight=batch["weight"],
        )

        loss = stats["loss"]

        ########################################################
        # Backpropagation
        ########################################################

        optimizer.zero_grad()

        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

        optimizer.step()

        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                print(name, "became NaN after optimizer step")
                raise RuntimeError

        ########################################################
        # Initialize metrics
        ########################################################

        if running is None:

            running = {
                key: 0.0
                for key in stats
            }

        ########################################################
        # Accumulate metrics
        ########################################################

        for key, value in stats.items():

            if torch.is_tensor(value):
                running[key] += value.item()
            else:
                running[key] += float(value)

    ############################################################
    # Average
    ############################################################

    for key in running:
        running[key] /= len(loader)

    return running


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate the model for one epoch.

    Returns
    -------
    dict
        Dictionary containing average validation metrics.
    """

    model.eval()

    running = None

    for batch in loader:

        ########################################################
        # Move tensors to device
        ########################################################

        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        ########################################################
        # Forward
        ########################################################

        output = model(
            batch["player_features"],
            batch["role_features"],
            batch["team"],
            batch["team_state"],
            batch["auction_state"],
        )

        ########################################################
        # Loss
        ########################################################

        stats = criterion(
            mu=output["mu_effective"],
            sigma=output["sigma"],
            lower_bid=batch["lower_bid"],
            upper_bid=batch["upper_bid"],
            observation_type=batch["observation_type"],
            weight=batch["weight"],
        )

        ########################################################
        # Initialize metrics
        ########################################################

        if running is None:

            running = {
                key: 0.0
                for key in stats
            }

        ########################################################
        # Accumulate metrics
        ########################################################

        for key, value in stats.items():

            if torch.is_tensor(value):
                running[key] += value.item()
            else:
                running[key] += float(value)

    ############################################################
    # Average
    ############################################################

    for key in running:
        running[key] /= len(loader)

    return running


@torch.no_grad()
def predict(
    model,
    dataset,
    device,
    batch_size=256,
):
    """
    Run the model over every row of `dataset`, in original row order.

    Returns
    -------
    dict of np.ndarray, each length == len(dataset):
        "mu", "sigma", "mu_effective"
    """

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    mus, sigmas, mu_effectives = [], [], []

    for batch in loader:

        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        output = model(
            batch["player_features"],
            batch["role_features"],
            batch["team"],
            batch["team_state"],
            batch["auction_state"],
        )

        mus.append(output["mu"].squeeze(-1).cpu())
        sigmas.append(output["sigma"].squeeze(-1).cpu())
        mu_effectives.append(output["mu_effective"].squeeze(-1).cpu())

    return {
        "mu": torch.cat(mus).numpy(),
        "sigma": torch.cat(sigmas).numpy(),
        "mu_effective": torch.cat(mu_effectives).numpy(),
    }


def evaluate_predictions(
    model,
    dataset,
    device,
    batch_size=256,
):
    """
    Compare model predictions against the actual observed auction
    outcome for every row in `dataset`, aligned 1:1 with
    dataset.training_df.

    Since mu / mu_effective live in log-space (the model outputs the
    parameters of a LogNormal), predicted point estimates are
    converted back to real bid units here:

        median value = exp(mu_effective)
        mean value   = exp(mu_effective + sigma^2 / 2)

    Returns
    -------
    pd.DataFrame
        One row per (player, team) observation, with predicted
        valuation columns alongside the actual lower/upper bid
        interval that was observed in that auction.
    """

    preds = predict(
        model,
        dataset,
        device,
        batch_size=batch_size,
    )

    mu_effective = preds["mu_effective"]
    sigma = preds["sigma"]

    predicted_median_value = np.exp(mu_effective)

    # Clip the exponent, not the inputs: with a still-mispriced model
    # (or sigma near its 3.0 ceiling) mu_effective + 0.5*sigma^2 can
    # be large enough that np.exp overflows to inf and warns. 700 is
    # comfortably past float64's ~709 overflow point, so this only
    # ever affects display of already-nonsensical predictions.
    mean_log_value = np.clip(mu_effective + 0.5 * sigma ** 2, a_min=None, a_max=700.0)
    predicted_mean_value = np.exp(mean_log_value)

    ########################################################
    # Metadata carried through to the comparison frame.
    #
    # The auction-progress and team-state columns are here so that
    # log_phi can be analysed against the state it is a function of.
    # Without them the adjustment is a number with nothing to plot it
    # against, and the claim that it rises early and falls late is
    # untestable from the predictions frame alone.
    #
    # auction_year is needed to pool several holdout seasons into one
    # frame and still separate them.
    #
    # Every name is guarded by the membership test below, so listing a
    # column that a given build does not produce is harmless.
    ########################################################

    metadata_columns = [
        c for c in [
            "playerName",
            "team",
            "role",
            "basePrice",
            "auctionPrice",
            "playsForTeam",
            "auctionStatus",
            "observation_type",

            "auction_year",

            # Auction progress -- the x-axis for any log_phi analysis.
            "auction_order",
            "players_completed",
            "players_remaining",

            # Team state -- the "how desperate are we" side of phi.
            "remaining_purse",
            "remaining_slots",
            "players_bought",

            # Whether the player is priceable at all. A debutant with no
            # senior record and no prior auction price is not a player the
            # model got wrong; he is a player no performance-conditioned
            # model can price, and pooling him into an error statistic
            # understates the model on players it can actually see. These
            # are the flags that split that subset out.
            "last_salary_is_missing",
            "age_is_missing",
            "last_salary",
            "age",
            "cappedStatus",
            "isPlayerOverseas",
            "untagged",
        ]
        if c in dataset.training_df.columns
    ]

    comparison = (
        dataset.training_df[metadata_columns]
        .copy()
        .reset_index(drop=True)
    )

    comparison["winner"] = dataset.winner.numpy()
    comparison["lower_bid"] = dataset.lower_bid.numpy()
    comparison["upper_bid"] = dataset.upper_bid.numpy()

    comparison["predicted_mu"] = preds["mu"]
    comparison["predicted_mu_effective"] = mu_effective

    ########################################################
    # The auction adjustment itself.
    #
    # log_phi = mu_effective - mu by construction, so this adds no
    # information -- but it is the quantity the model is built around
    # ("who do we like" vs "how desperate are we"), and requiring every
    # analysis to re-derive it invites sign errors. phi is the
    # multiplicative form: phi > 1 means bidding above intrinsic value.
    ########################################################

    comparison["predicted_log_phi"] = mu_effective - preds["mu"]
    comparison["predicted_phi"] = np.exp(comparison["predicted_log_phi"])
    comparison["predicted_sigma"] = sigma
    comparison["predicted_median_value"] = predicted_median_value
    comparison["predicted_mean_value"] = predicted_mean_value

    ########################################################
    # Predicted (mu - 3*sigma, mu + 3*sigma) band, mapped back
    # to real price units. This is a check on the whole
    # predictive distribution, not just the median point
    # estimate -- under a LogNormal this band covers ~99.7% of
    # the distribution's mass, so a well-calibrated model should
    # have the true outcome fall inside it the large majority of
    # the time. Clip the exponent for the same overflow reason
    # as predicted_mean_value above.
    ########################################################

    band_low_log = np.clip(mu_effective - 3.0 * sigma, a_min=-700.0, a_max=700.0)
    band_high_log = np.clip(mu_effective + 3.0 * sigma, a_min=-700.0, a_max=700.0)

    comparison["predicted_band_low"] = np.exp(band_low_log)
    comparison["predicted_band_high"] = np.exp(band_high_log)

    # What actually happened: the real sale price for winners,
    # otherwise the midpoint of the observed censoring interval
    # as the best available stand-in for "true value".
    actual_reference = comparison["lower_bid"].to_numpy(dtype=np.float64)
    if "auctionPrice" in comparison.columns:
        actual_reference = np.where(
            comparison["winner"].to_numpy()
            & comparison["auctionPrice"].notna().to_numpy(),
            comparison["auctionPrice"].to_numpy(dtype=np.float64),
            (
                comparison["lower_bid"].to_numpy(dtype=np.float64)
                + comparison["upper_bid"].to_numpy(dtype=np.float64)
            )
            / 2.0,
        )

    comparison["actual_reference_value"] = actual_reference

    comparison["within_3sigma_band"] = (
        (actual_reference >= comparison["predicted_band_low"])
        & (actual_reference <= comparison["predicted_band_high"])
    )

    # Did the predicted median valuation fall inside the observed
    # (lower, upper) bid interval for that team/player?
    comparison["within_interval"] = (
        (comparison["predicted_median_value"] >= comparison["lower_bid"])
        & (comparison["predicted_median_value"] <= comparison["upper_bid"])
    )

    # For rows where the team actually won the player, this is a
    # direct point-estimate error against the real auction price.
    if "auctionPrice" in comparison.columns:
        comparison["error_vs_auction_price"] = (
            comparison["predicted_median_value"] - comparison["auctionPrice"]
        )

    ########################################################
    # The bounds THIS model was built with, carried on the frame.
    #
    # sigma_saturation is mean(sigma) / sigma_max and log_phi
    # saturation is measured against max_log_phi, so both diagnostics
    # need a ceiling. Reading it from the live global config at
    # reporting time -- which is what summarize_predictions used to
    # do -- gives the right answer only if the config still holds the
    # values the model was constructed under. It does not in the one
    # case that matters: an ablation that overrides sigma_max inside
    # a try/finally restores the config before the metrics are
    # computed, so a run with a ceiling of 1.0 was scored against
    # 1.5 and its saturation came out as 0.65 when the model was in
    # fact pinned at 0.98 of its own ceiling. The ablation that was
    # supposed to detect hedging reported the opposite.
    #
    # attrs travels with the frame, so the number is now the model's
    # own and cannot drift from it.
    ########################################################

    intrinsic = getattr(model, "intrinsic", model)
    comparison.attrs["sigma_min"] = float(getattr(intrinsic, "sigma_min", 0.05))
    comparison.attrs["sigma_ceiling"] = float(getattr(intrinsic, "sigma_max", 1.5))
    auction = getattr(model, "auction", None)
    if auction is not None:
        comparison.attrs["max_log_phi"] = float(
            getattr(auction, "max_log_phi", 1.5)
        )

    return comparison


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs,
    valid_loader=None,
    max_grad_norm=5.0,
    patience=None,
    min_delta=0.0,
    restore_best=True,
):
    """
    Complete training loop.

    Parameters
    ----------
    max_grad_norm : float or None
        Passed through to `train_one_epoch` each epoch -- see there
        for why this replaced the old nll clamp.
    patience : int or None
        Stop after this many epochs with no improvement in validation
        loss. None disables early stopping.

        This is not optional polish. Validation loss on this task
        bottoms out within the first handful of epochs and then climbs
        while training loss keeps falling, so a fixed 20-epoch run
        returns a model from well past its best point -- and, worse,
        makes feature work unreadable: a new feature that genuinely
        helps and a model that simply memorised more both show up as
        lower training loss.
    min_delta : float
        Improvement smaller than this does not reset the counter.
    restore_best : bool
        Load the best-validation weights back into `model` before
        returning. Without this, early stopping only saves time; the
        model you evaluate is still the overfit one from the last
        epoch.

    Returns
    -------
    history : dict
        Metric history, plus "best_epoch" and "best_valid_loss".
    """

    history = {
        "train": [],
        "valid": [],
        "best_epoch": None,
        "best_valid_loss": None,
    }

    best_loss = float("inf")
    best_state = None
    best_epoch = None
    since_improvement = 0

    for epoch in range(epochs):

        ########################################################
        # Training
        ########################################################

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
        )

        history["train"].append(train_stats)

        ########################################################
        # Validation
        ########################################################

        if valid_loader is not None:

            valid_stats = validate_one_epoch(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
            )

            history["valid"].append(valid_stats)

            improved = valid_stats["loss"] < best_loss - min_delta

            if improved:
                best_loss = valid_stats["loss"]
                best_epoch = epoch + 1
                since_improvement = 0
                if restore_best:
                    best_state = {
                        k: v.detach().clone()
                        for k, v in model.state_dict().items()
                    }
            else:
                since_improvement += 1

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_stats['loss']:.4f} | "
                f"Valid Loss: {valid_stats['loss']:.4f}"
                f"{'  *' if improved else ''}"
            )

            if patience is not None and since_improvement >= patience:
                print(
                    f"Early stop: no improvement for {patience} epochs. "
                    f"Best was epoch {best_epoch} at {best_loss:.4f}."
                )
                break

        else:

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_stats['loss']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Restored weights from epoch {best_epoch} "
            f"(valid loss {best_loss:.4f})."
        )

    history["best_epoch"] = best_epoch
    history["best_valid_loss"] = best_loss if best_epoch else None

    return history