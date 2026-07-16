import pandas as pd
from input_creation.player_features.player_features import PlayerStatsAggregator, PlayerFeatureBuilder
from input_creation.auction_state.auction_state import AuctionReplayEngine
from input_creation.auction_state.utils import build_bid_summary
from .auction_state.utils import build_bid_summary

class LabelEncoder:

    def __init__(self):

        self.label_to_idx = {}
        self.idx_to_label = {}

    def fit(self, values):

        values = (
            pd.Series(values)
            .dropna()
            .unique()
        )

        values = sorted(values)

        self.label_to_idx = {
            label: idx
            for idx, label in enumerate(values)
        }

        self.idx_to_label = {
            idx: label
            for label, idx in self.label_to_idx.items()
        }

        return self

    def transform(self, values):

        return (
            pd.Series(values)
            .map(self.label_to_idx)
            .astype(int)
        )

    def fit_transform(self, values):

        self.fit(values)

        return self.transform(values)

    def inverse_transform(self, values):

        return (
            pd.Series(values)
            .map(self.idx_to_label)
        )

    @property
    def classes_(self):

        return list(self.label_to_idx.keys())
    
class EncoderManager:

    def __init__(self):

        self.encoders = {}

    def fit(self, df, columns):

        for column in columns:

            encoder = LabelEncoder()

            encoder.fit(df[column])

            self.encoders[column] = encoder

        return self

    def transform(self, df):

        df = df.copy()

        for column, encoder in self.encoders.items():

            df[column] = encoder.transform(df[column])

        return df

    def fit_transform(self, df, columns):

        self.fit(df, columns)

        return self.transform(df)

    def get_encoder(self, column):

        return self.encoders[column]
    


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

def build_encoders(training_df):

    manager = EncoderManager()

    manager.fit(
        training_df,
        [
            "team",
            "role"
        ]
    )

    return manager