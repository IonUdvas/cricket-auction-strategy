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
            batch["role"],
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
            batch["role"],
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
            batch["role"],
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
    comparison["predicted_sigma"] = sigma
    comparison["predicted_median_value"] = predicted_median_value
    comparison["predicted_mean_value"] = predicted_mean_value

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
):
    """
    Complete training loop.

    Parameters
    ----------
    max_grad_norm : float or None
        Passed through to `train_one_epoch` each epoch -- see there
        for why this replaced the old nll clamp.

    Returns
    -------
    history : dict
        Dictionary containing metric history.
    """

    history = {
        "train": [],
        "valid": [],
    }

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

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_stats['loss']:.4f} | "
                f"Valid Loss: {valid_stats['loss']:.4f}"
            )

        else:

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_stats['loss']:.4f}"
            )

    return history