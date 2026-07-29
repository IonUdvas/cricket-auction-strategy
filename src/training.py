from input_creation_2.auction_dataset_utils import build_training_samples, build_encoders
from input_creation_2.auction_dataset import IPLAuctionDataset
from valuation_model.models import *
from valuation_model.losses import *
from valuation_model.training import *
from torch.utils.data import DataLoader

import pandas as pd
import yaml

with open("/kaggle/working/cricket-auction-strategy/configs/default.yaml","r") as f:
    config = yaml.safe_load(f)

AUCTION_DATES = {
    2018: "2018-01-27",
    2019: "2018-12-18",
    2020: "2019-12-19",
    2021: "2021-02-18",
    2022: "2022-02-12",
    2023: "2022-12-23",
    2024: "2023-12-19",
    2025: "2024-11-24",
    2026: "2025-12-16",
}

AUCTION_MAX_PURSES = {
    2018: 8000,
    2019: 8200,
    2020: 8500,
    2021: 8500,
    2022: 9000,
    2023: 9500,
    2024: 10000,
    2025: 12000,
    2026: 12500,
}

def build_training_df(
        player_template,
        bid_template,
        bbb_parquet_PATH,
        years=None,
):
    """
    years : iterable of int, optional
        Subset of AUCTION_DATES keys to build. Defaults to all years.
    """

    training_dfs = {}

    selected_years = (
        AUCTION_DATES
        if years is None
        else {y: AUCTION_DATES[y] for y in years}
    )

    for year, auction_date in selected_years.items():

        print(f"Building {year}...")

        player_df_PATH = player_template.format(year=year)

        bid_df_PATH = bid_template.format(year=year)

        training_df = build_training_samples(
            player_df_PATH,
            bid_df_PATH,
            bbb_parquet_PATH,
            auction_date,
            auction_max_purse = AUCTION_MAX_PURSES[year]
        )

        training_dfs[year] = training_df

        print(
            f"Finished {year}: "
            f"{len(training_df)} training rows"
        )

    full_training_df = pd.concat(
        training_dfs.values(),
        ignore_index=True
    )

    return full_training_df

def load_and_encode_data(full_training_df):
    encoder_manager = build_encoders(full_training_df)

    dataset = IPLAuctionDataset(
        full_training_df,
        encoder_manager
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    return encoder_manager, dataset, loader

def run_training_pipeline(
        player_template,
        bid_template,
        parquet_path
):
    
    full_training_df = build_training_df(player_template, bid_template, parquet_path)
    encoder_manager, dataset, loader = load_and_encode_data(full_training_df)

    config["model"]["player_dim"] = len(
        full_training_df.attrs["player_feature_columns"]
    )

    config["model"]["team_state_dim"] = len(
        full_training_df.attrs["team_state_columns"]
    )

    config["model"]["auction_state_dim"] = len(
        full_training_df.attrs["auction_state_columns"]
    )

    config["model"]["num_archetypes"] = len(
        encoder_manager.get_encoder("role").classes_
    )

    config["model"]["num_teams"] = len(
        encoder_manager.get_encoder("team").classes_
    )

    model = ValuationModel(
        player_dim=config["model"]["player_dim"],
        team_state_dim=config["model"]["team_state_dim"],
        auction_state_dim=config["model"]["auction_state_dim"],
        num_archetypes=config["model"]["num_archetypes"],
        num_teams=config["model"]["num_teams"],
        embedding_dim=config["model"]["embedding_dim"]

    )

    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            print(name, "contains NaNs/Infs")

    criterion = IntervalCensoredLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    history = train(
        model=model,
        train_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["training"]["epochs"],
    )

    return model, history, encoder_manager, dataset, loader, full_training_df


def run_training_pipeline_with_holdout(
        player_template,
        bid_template,
        parquet_path,
        train_years,
        val_years,
):
    """
    Same as run_training_pipeline, but builds train_years and val_years
    as separate auctions, trains only on train_years, and evaluates
    (no gradient updates) on val_years each epoch.

    Example
    -------
    train_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    val_years = [2026]
    """

    ################################################################
    # Build train / val dataframes as separate auction replays
    ################################################################

    train_df = build_training_df(
        player_template, bid_template, parquet_path, years=train_years
    )

    val_df = build_training_df(
        player_template, bid_template, parquet_path, years=val_years
    )

    ################################################################
    # Fit encoders on the UNION of train + val categories.
    # This only fixes team/role/observation_type vocabulary --
    # it does not leak any auction outcome / target information --
    # but is required because the encoder has no "unknown" bucket
    # and will crash on a category it has never seen.
    ################################################################

    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    encoder_manager = build_encoders(combined_df)

    train_dataset = IPLAuctionDataset(train_df, encoder_manager)
    val_dataset = IPLAuctionDataset(val_df, encoder_manager)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    ################################################################
    # Model dims come from train_df; train/val share schema since
    # both come from build_training_df / build_training_samples.
    ################################################################

    config["model"]["player_dim"] = len(
        train_df.attrs["player_feature_columns"]
    )

    config["model"]["team_state_dim"] = len(
        train_df.attrs["team_state_columns"]
    )

    config["model"]["auction_state_dim"] = len(
        train_df.attrs["auction_state_columns"]
    )

    config["model"]["num_archetypes"] = len(
        encoder_manager.get_encoder("role").classes_
    )

    config["model"]["num_teams"] = len(
        encoder_manager.get_encoder("team").classes_
    )

    model = ValuationModel(
        player_dim=config["model"]["player_dim"],
        team_state_dim=config["model"]["team_state_dim"],
        auction_state_dim=config["model"]["auction_state_dim"],
        num_archetypes=config["model"]["num_archetypes"],
        num_teams=config["model"]["num_teams"],
        embedding_dim=config["model"]["embedding_dim"],
    )

    criterion = IntervalCensoredLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    history = train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["training"]["epochs"],
        valid_loader=val_loader,
    )

    val_predictions = evaluate_predictions(
        model=model,
        dataset=val_dataset,
        device=device,
    )

    return {
        "model": model,
        "history": history,
        "encoder_manager": encoder_manager,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_df": train_df,
        "val_df": val_df,
        "val_predictions": val_predictions,
    }