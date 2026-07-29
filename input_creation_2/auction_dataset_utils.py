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
            # "role" is intentionally not label-encoded: it's a
            # numeric multi-hot block (see `role_columns` in
            # training_df.attrs), not a single mutually-exclusive
            # category. See build_training_samples / build_role_table.
            "observation_type",
        ],
    )

    return manager

def build_role_table(
    training_df,
    player_role_df=None,
):
    """
    Build the player-role feature block as a numeric multi-hot table,
    regardless of source. Returns (role_frame, role_columns):

        role_frame   : pd.DataFrame indexed like training_df, one
                        0/1 float column per role tag.
        role_columns : list of str, the names of those columns
                        (stash this in training_df.attrs["role_columns"]).

    Two sources, unified into the same contract:

    1. player_role_df given (the curated multi-label table): a
       player can be RHB *and* pace *and* death_overs_bowler *and*
       bowling_allrounder at once, so this is fundamentally a
       multi-hot block, not a single mutually-exclusive category.
       Matched on playerId when available (robust to name variants
       across auction years), falling back to playerName otherwise.

       Identity columns that duplicate what the auction-replay
       engine already provides on training_df (country, capped
       status, overseas status) are dropped from the merge rather
       than kept side-by-side under different names/encodings --
       keeping both would just double-count the same signal under
       two different columns.

    2. player_role_df is None (legacy path): training_df already has
       a single categorical "role" string from the auction dataset
       (BATTER / BOWLER / ALL-ROUNDER / WICKETKEEPER). This is
       one-hot encoded here so downstream code sees the exact same
       "numeric multi-hot block" shape either way -- no separate
       code path needed in the dataset or model for the legacy case.
    """

    ################################################################
    # Identity fields the auction-replay engine already attaches to
    # every row of training_df -- if player_role_df carries its own
    # versions of these, they're redundant (and often differently
    # encoded, e.g. "capped"/"uncapped" strings vs. a boolean), so
    # they're dropped from the merge rather than merged in under a
    # colliding or shadow name.
    ################################################################

    DUPLICATE_IDENTITY_COLUMNS = {
        "country": "country",
        "capped_status": "cappedStatus",
        "is_overseas": "isPlayerOverseas",
    }

    if player_role_df is not None:

        role_df = player_role_df.copy()

        ############################################################
        # Normalize the id column and pick a merge key. playerId is
        # preferred -- it's stable across auction years, unlike
        # playerName (spelling variants, retirements/comebacks under
        # slightly different listed names, etc).
        ############################################################

        if "player_id" in role_df.columns:
            role_df = role_df.rename(columns={"player_id": "playerId"})

        if "playerId" in role_df.columns:
            merge_key = "playerId"
            role_df["playerId"] = pd.to_numeric(
                role_df["playerId"], errors="coerce"
            )
        elif "playerName" in role_df.columns:
            merge_key = "playerName"
        else:
            raise ValueError(
                "player_role_df must contain a 'playerId' "
                "(or 'player_id') or 'playerName' column to merge on."
            )

        drop_cols = [
            col
            for col in DUPLICATE_IDENTITY_COLUMNS
            if col in role_df.columns
        ]

        if drop_cols:
            role_df = role_df.drop(columns=drop_cols)

        tag_columns = [
            c for c in role_df.columns if c not in (merge_key, "playerName")
        ]

        ############################################################
        # Coerce every tag column to 0/1 float. Booleans cast
        # directly; anything unexpectedly non-numeric (e.g. a
        # leftover string category) is one-hot expanded rather than
        # silently dropped or crashing on .astype(float).
        ############################################################

        numeric_frame = pd.DataFrame(index=role_df.index)
        role_columns = []

        for col in tag_columns:

            series = role_df[col]

            if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
                numeric_frame[col] = series.astype(float)
                role_columns.append(col)
            else:
                dummies = pd.get_dummies(series, prefix=col, dummy_na=False)
                dummies = dummies.astype(float)
                numeric_frame = pd.concat([numeric_frame, dummies], axis=1)
                role_columns.extend(dummies.columns.tolist())

        role_df = pd.concat(
            [role_df[[merge_key]], numeric_frame], axis=1
        )

        merged = training_df[[merge_key]].merge(
            role_df,
            on=merge_key,
            how="left",
        )

        assert len(merged) == len(training_df), (
            f"player_role_df appears to have duplicate '{merge_key}' "
            f"values -- merging it produced {len(merged)} rows from "
            f"{len(training_df)} input rows. Dedupe player_role_df on "
            f"'{merge_key}' before passing it in."
        )

        role_frame = merged[role_columns].reset_index(drop=True)

        return role_frame, role_columns

    ################################################################
    # Legacy path: one-hot the existing single "role" column so the
    # rest of the pipeline never has to know which source it came
    # from.
    ################################################################

    dummies = pd.get_dummies(
        training_df["role"], prefix="role", dummy_na=False
    ).astype(float)

    return dummies.reset_index(drop=True), dummies.columns.tolist()


def build_training_samples(
    player_df_PATH,
    bid_df_PATH,
    bbb_data_parquet_PATH,
    auction_date,
    auction_max_purse,
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

    Note: player-role handling (the curated multi-tag table, or the
    legacy single "role" column) is intentionally not done here --
    see build_role_table and the note further down. It's applied
    once, after all years are concatenated together, by
    build_training_df / run_training_pipeline_with_holdout.
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
    #
    # Deliberately left as the single raw "role" string column here.
    # Turning it into the final numeric role block (via
    # build_role_table) happens one level up, in build_training_df,
    # *after* every year's frame has been concatenated together --
    # not per-year here. If it were done per-year, each year could
    # independently discover a different set of role columns (e.g.
    # a role tag that happens not to appear in one year's slice, or
    # -- for the legacy one-hot fallback -- a "role" string value
    # missing from that year), and concatenating frames with
    # different columns afterwards would silently produce a
    # misaligned/inconsistent feature set. Building the role table
    # once, after concatenation, guarantees one shared vocabulary --
    # exactly the same reasoning that already applies to the
    # team/observation_type encoders (fit once on the full
    # concatenated frame, not per year).
    ############################################################

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