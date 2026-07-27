import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

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
        )

        loss = stats["loss"]

        ########################################################
        # Backpropagation
        ########################################################

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

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


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs,
    valid_loader=None,
):
    """
    Complete training loop.

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