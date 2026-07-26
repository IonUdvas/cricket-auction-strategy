import pandas as pd
from input_creation_2.player_features.player_features import PlayerStatsAggregator, PlayerFeatureBuilder
from input_creation_2.auction_replay_engine import AuctionReplayEngine

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
    

def build_encoders(training_df):

    manager = EncoderManager()

    manager.fit(
        training_df,
        [
            "team",
            "role",
            "observation_type",
        ],
    )

    return manager

def build_training_samples(
    player_df_PATH,
    bid_df_PATH,
    bbb_data_parquet_PATH,
    auction_date,
    auction_max_purse,
    player_role_df=None,
):
    """
    Build the complete training dataframe.

    Parameters
    ----------
    player_df_PATH : str

    bid_df_PATH : str

    bbb_data_parquet_PATH : str

    auction_date : str or datetime

    auction_max_purse : float

    player_role_df : pd.DataFrame, optional

        Expected columns:

            playerName
            role
    """

    ############################################################
    # Load historical cricket data
    ############################################################

    bbb_data_df = (
        pd.read_parquet(bbb_data_parquet_PATH)
        .sort_values("match_date")
        .reset_index(drop=True)
    )

    ############################################################
    # Feature Builder
    ############################################################

    player_feature_builder = PlayerFeatureBuilder(
        PlayerStatsAggregator(bbb_data_df)
    )

    ############################################################
    # Load auction data
    ############################################################

    bid_df = pd.read_csv(bid_df_PATH)

    player_df = pd.read_csv(player_df_PATH)

    ############################################################
    # Replay Auction
    ############################################################

    engine = AuctionReplayEngine(
        bid_df=bid_df,
        player_df=player_df,
        auction_max_purse=auction_max_purse,
    )

    outputs = engine.replay()

    training_df = outputs["training"]

    auction_state_df = outputs["auction_state"]

    team_state_df = outputs["team_state"]

    bid_summary_df = outputs["bid_summary"]

    ############################################################
    # Player Features
    ############################################################

    player_features = (
        player_feature_builder
        .build_feature_table(
            player_df["playerName"].tolist(),
            auction_date,
        )
    )

    print(
        "Player Features Done:",
        player_features.shape
    )

    training_df = training_df.merge(
        player_features,
        on="playerName",
        how="left"
    )

    ############################################################
    # Player Roles / Archetypes
    ############################################################

    if player_role_df is not None:

        training_df = training_df.drop(
            columns=["role"],
            errors="ignore"
        )

        training_df = training_df.merge(
            player_role_df,
            on="playerName",
            how="left"
        )

    ############################################################
    # Store Feature Groups
    ############################################################

    metadata_columns = {

        "playerId",
        "playerName",
        "team",

        "country",
        "countryId",
        "cappedStatus",

        "isPlayerOverseas",

        "basePrice",
        "auctionPrice",

        "auctionStatus",

        "playsForTeam",

        "role",

    }

    training_df.attrs["player_feature_columns"] = [

        c

        for c in player_features.columns

        if c != "playerName"

    ]

    training_df.attrs["auction_state_columns"] = [

        c

        for c in auction_state_df.columns

        if c not in metadata_columns

    ]

    training_df.attrs["team_state_columns"] = [

        c

        for c in team_state_df.columns

        if c not in metadata_columns

    ]

    training_df.attrs["bid_summary_columns"] = [

        c

        for c in bid_summary_df.columns

        if c not in metadata_columns

    ]

    return training_df