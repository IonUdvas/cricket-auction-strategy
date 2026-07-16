import pandas as pd
from input_creation.player_features.player_features import PlayerStatsAggregator, PlayerFeatureBuilder
from input_creation.auction_state.auction_state import AuctionReplayEngine
from input_creation.auction_state.utils import build_bid_summary
from .auction_state.utils import build_bid_summary

def build_training_samples(
    player_df_PATH,
    bid_df_PATH,
    bbb_data_parquet_PATH,
    auction_date
):
    bbb_data_df = pd.read_parquet(bbb_data_parquet_PATH).sort_values("match_date").reset_index(drop=True)
    player_feature_builder = PlayerFeatureBuilder(PlayerStatsAggregator(bbb_data_df))

    bid_df = pd.read_csv(bid_df_PATH)
    player_df = pd.read_csv(player_df_PATH)

    engine = AuctionReplayEngine(
        bid_df,
        player_df,
        initial_purse=800
    )

    auction_state_df, team_state_df = engine.replay()
    ############################################################
    # 1. Player Features
    ############################################################

    player_features = (
        player_feature_builder
        .build_feature_table(
            player_df["playerName"].tolist(),
            auction_date
        )
    )

    print("Player Features Done", player_features.shape)

    
    ############################################################
    # 2. Player Role
    ############################################################
    
    roles = player_df[
        ["playerId", "role"]
    ].copy()
    
    ############################################################
    # Only keep players for whom features exist
    ############################################################
    
    valid_players = set(
        player_features["playerName"]
    )
    
    ############################################################
    # Bid Summaries
    ############################################################
    
    summaries = []
    
    for player_name, player_bid_df in bid_df.groupby("playerName"):
    
        if player_name not in valid_players:
            continue
    
        summaries.append(
            build_bid_summary(player_bid_df)
        )
    
    bid_summary = pd.concat(
        summaries,
        ignore_index=True
    )
    ############################################################
    # 4. Merge everything
    ############################################################

    training_df = (
        bid_summary
        .merge(
            player_features,
            on=["playerName"],
            how="left"
        )
        .merge(
            roles,
            on="playerId",
            how="left"
        )
    )

    training_df = (
        training_df
        .merge(
            auction_state_df,
            on=["playerId", "playerName"],
            how="left"
        )
        .merge(
            team_state_df,
            on=["playerId", "playerName", "team", "auction_order"],
            how="left"
        )
    )

    training_df.attrs["player_feature_columns"] = list(player_features.columns.drop("playerName"))

    training_df.attrs["auction_state_columns"] = [
        c for c in auction_state_df.columns
        if c not in ["playerId", "playerName"]
    ]
    
    training_df.attrs["team_state_columns"] = [
        c for c in team_state_df.columns
        if c not in [
            "playerId",
            "playerName",
            "team",
            "auction_order"
        ]
    ]

    return training_df