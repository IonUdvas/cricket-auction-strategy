import os

import pandas as pd
from input_creation_2.player_features.player_features import PlayerStatsAggregator, PlayerFeatureBuilder
from input_creation_2.player_features.identity import PlayerIdentityResolver
from input_creation_2.player_features.squad_index import SquadIndex
from input_creation_2.auction_replay_engine import AuctionReplayEngine


class PlayerFeatureContext:
    """
    Owns everything that is shared across auction years: the ball-by-ball
    aggregator, the feature flattener, and the auction-roster -> person_id
    resolution map.

    Two things drive this design.

    First, the aggregator is as-of queryable, so it carries no year-specific
    state; building it once instead of once per year removes eight redundant
    passes over 2.4M deliveries.

    Second, identity has to be resolved ONCE for all years together.  Resolving
    per year lets the same auction playerId land on different people in
    different years -- the roster spells "Shivam Dubey" one year and "Shivam
    Dube" the next, and the name tiers answer differently.  A playerId is one
    human being; that has to be true by construction, not by luck.
    """

    def __init__(self, bbb_dir, competitions=None, overrides=None,
                 resolution=None, use_squads=True, verbose=True):
        need = ("deliveries.parquet", "fielding.parquet", "people.parquet")
        # The old pipeline took a path to one flat parquet, so this is the
        # first thing anyone upgrading gets wrong.  Say so, instead of listing
        # three files as "missing" from something that is not a directory.
        if os.path.isfile(bbb_dir) or str(bbb_dir).endswith(".parquet"):
            raise NotADirectoryError(
                f"{bbb_dir} is a single parquet file. This pipeline now needs "
                f"the DIRECTORY that data.build_bbb writes, holding "
                f"{', '.join(need)}. Use the repo's data/bbb, or rebuild with "
                f"`python -m data.build_bbb --out-dir data/bbb`."
            )
        missing = [f for f in need if not os.path.exists(os.path.join(bbb_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"{bbb_dir} is missing {missing}. Build it with "
                f"`python -m data.build_bbb --download --out-dir {bbb_dir}`."
            )

        deliveries = pd.read_parquet(os.path.join(bbb_dir, "deliveries.parquet"))
        fielding = pd.read_parquet(os.path.join(bbb_dir, "fielding.parquet"))
        people = pd.read_parquet(os.path.join(bbb_dir, "people.parquet"))

        # `fielding=` is not optional in practice: without it every field_*
        # column is silently zero for every player, which the model reads as
        # "nobody ever took a catch" rather than "unknown".
        self.aggregator = PlayerStatsAggregator(
            deliveries, competitions=competitions, fielding=fielding
        )
        self.builder = PlayerFeatureBuilder(self.aggregator)
        self.resolver = PlayerIdentityResolver(
            people,
            overrides=overrides,
            resolution=resolution,
            squad_index=SquadIndex(deliveries) if use_squads else None,
        )
        self.verbose = verbose
        self.person_by_player = {}
        self.identity_conflicts = []

    def register_rosters(self, rosters):
        """
        rosters : {auction_year: player_df}

        Resolve every roster row once, then collapse to one person_id per
        playerId.  A playerId that resolves two different ways across years is
        NOT silently majority-voted: the disagreement is itself evidence that
        at least one match is wrong, so it is recorded and left unresolved.
        """
        frames = []
        for year, df in rosters.items():
            f = df[["playerId", "playerName", "playsForTeam"]].copy()
            f["season_year"] = year
            frames.append(f)
        every = pd.concat(frames, ignore_index=True)

        resolved = self.resolver.resolve(every)
        got = resolved.dropna(subset=["person_id"])

        for pid_, grp in got.groupby("playerId"):
            people_hit = set(grp["person_id"])
            if len(people_hit) == 1:
                self.person_by_player[pid_] = next(iter(people_hit))
            else:
                self.identity_conflicts.append(
                    (pid_, grp["playerName"].iloc[0], sorted(people_hit))
                )

        if self.verbose:
            n_ids = every["playerId"].nunique()
            print(f"identity: {len(self.person_by_player)}/{n_ids} distinct "
                  f"playerIds resolved "
                  f"({len(self.person_by_player) / n_ids:.1%})")
            # The resolver now settles each playerId once, pooling every
            # spelling and every (season, franchise) it ever carried, so this
            # loop is a belt-and-braces check that should never fire.  The
            # disagreements it used to catch are caught one level down, in
            # `resolver.conflicts`, where the evidence still exists.
            if self.identity_conflicts:
                print(f"  {len(self.identity_conflicts)} playerIds resolved "
                      f"inconsistently across years and were left unresolved:")
                for pid_, name, hits in self.identity_conflicts[:10]:
                    print(f"    {name!r} (id {pid_}) -> {hits}")
            if self.resolver.conflicts:
                print(f"  {len(self.resolver.conflicts)} playerIds whose "
                      f"spellings disagreed and were left unresolved:")
                for pid_, name, hits in self.resolver.conflicts[:10]:
                    print(f"    {name!r} (id {pid_}) -> {hits}")
            if self.resolver.cache_misses:
                print(f"  {len(self.resolver.cache_misses)} cached cricinfo ids "
                      f"have no T20 ball data")
        return self.person_by_player

    def features_for(self, player_df, auction_date):
        """
        One feature row per auction `playerId`, as of `auction_date`.

        Deliberately not `build_feature_table`: that keys on and de-duplicates
        by person_id, and every unresolved player has person_id = None, so
        merging back onto playerId would collapse them into a single row and
        then match none of them, because null never equals null in a join.
        Flattening per roster row keeps the frame keyed on playerId, which is
        what the training rows actually carry.
        """
        roster = player_df[["playerId"]].drop_duplicates()
        rows = [
            self.builder.flatten(
                self.aggregator.get_player_stats(
                    self.person_by_player.get(pid_), auction_date
                )
            )
            for pid_ in roster["playerId"]
        ]
        features = pd.DataFrame(rows)
        features.insert(0, "playerId", roster["playerId"].to_numpy())

        assert len(features) == len(roster), "feature table lost or gained rows"
        assert features["playerId"].is_unique, "duplicate playerId in features"
        return features

    @staticmethod
    def feature_column_names(features):
        """The numeric block only -- never the join keys."""
        return [c for c in features.columns if c not in ("playerId", "playerName")]

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
    feature_context,
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

    # The roster's Cricbuzz playerId is NOT the ball data's Cricsheet
    # person_id.  features_for() goes through the resolved identity map;
    # passing playerId straight to the aggregator (the old behaviour) gave
    # every single player an empty career.
    player_features = feature_context.features_for(
        player_df,
        auction_date,
    )

    print(
        "Player Features Done:",
        player_features.shape
    )

    before = len(training_df)
    training_df = training_df.merge(
        player_features,
        on="playerId",
        how="left",
        validate="many_to_one",     # <-- crashes instead of silently fanning out
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

    # The id column must not leak into the feature block: this used to filter
    # out only "playerName", leaving the raw Cricbuzz playerId to be cast to
    # float32 and fed to the model as a numeric feature.
    training_df.attrs["player_feature_columns"] = (
        PlayerFeatureContext.feature_column_names(player_features)
    )

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