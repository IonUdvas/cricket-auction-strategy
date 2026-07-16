from input_creation.auction_dataset_utils import build_training_samples, build_encoders
from input_creation.auction_dataset import IPLAuctionDataset
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

def build_training_df(
        player_template,
        bid_template,
        bbb_parquet_PATH
):
    training_dfs = {}
    
    for year, auction_date in AUCTION_DATES.items():

        print(f"Building {year}...")

        player_df_PATH = player_template.format(year=year)

        bid_df_PATH = bid_template.format(year=year)

        training_df = build_training_samples(
            player_df_PATH,
            bid_df_PATH,
            bbb_parquet_PATH,
            auction_date
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

    return encoder_manager, loader

def run_training_pipeline(
        player_template,
        bid_template,
        parquet_path
):
    
    full_training_df = build_training_df(player_template, bid_template, parquet_path)
    encoder_manager, loader = load_and_encode_data(full_training_df)

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

    return model, history, encoder_manager, full_training_df