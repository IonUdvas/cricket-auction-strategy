import os

import numpy as np
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

####################################################################
# Columns in player_role_df that must never become model features.
#
# These are not roles. They are player identity (a name, a date of
# birth, the ball-by-ball name the matcher landed on) or bookkeeping
# about the archetype table itself. One-hot expanding them, which is
# what the `else` branch below used to do to any non-numeric column,
# produced 3,409 of 3,467 "role" features -- an ~800-way one-hot
# lookup on the player's own name, three more copies of it, and the
# raw semicolon-joined tag string as a 222-way category on top of the
# individual tags it is built from.
#
# That block is a memorisation channel, not a feature: a player with
# two or three auctions in the whole dataset gets a private column
# whose only job is to carry his training price. 65% of validation
# rows land on a name column the model saw in training, which is
# exactly the shape of a train loss that keeps falling (2.84 -> 2.28)
# while validation loss more than doubles (2.68 -> 6.25).
####################################################################

ROLE_IDENTITY_COLUMNS = {
    # Player identity, in four spellings.
    "auction_name",
    "cricbuzz_name",
    "bbb_player",
    "date_of_birth",
    # The tag strings the individual boolean tag columns are parsed
    # from. Keeping both double-counts every tag and adds a
    # high-cardinality category for each distinct combination.
    "archetypes",
    "performance_archetypes",
    # Bookkeeping about how the archetype table was built.
    "match_method",
}

####################################################################
# Columns computed ACROSS the whole archetype table, i.e. as of the
# most recent auction in it, and therefore leaking the future into
# every earlier auction's rows.
#
# `last_auction` and `n_auctions` say how long a player's IPL career
# will turn out to run -- in a 2018 row, that is information from
# 2026. `age_at_last_auction` is an age as of the wrong date.
# `first_auction` is safe backwards but combines with the others.
#
# age is genuinely useful and date_of_birth is available; the right
# fix is to compute it against each row's own auction date in
# PlayerFeatureBuilder rather than to recover it from here.
####################################################################

ROLE_LEAKY_COLUMNS = {
    "first_auction",
    "last_auction",
    "n_auctions",
    "age_at_last_auction",
}


def build_role_table(
    training_df,
    player_role_df=None,
    max_role_cardinality=24,
    drop_identity_columns=True,
    drop_leaky_columns=True,
    verbose=True,
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

    Parameters
    ----------
    max_role_cardinality : int or None
        A non-numeric column is one-hot expanded only if it has at
        most this many distinct values. Above it, the column is a
        near-unique key rather than a category and is dropped with a
        warning. This is the general guard behind the specific
        ROLE_IDENTITY_COLUMNS list: a new identity-like column added
        to the archetype table later cannot silently reintroduce the
        same failure. Set to None to disable (not recommended).
    drop_identity_columns : bool
        Drop ROLE_IDENTITY_COLUMNS -- player names and the raw tag
        strings.
    drop_leaky_columns : bool
        Drop ROLE_LEAKY_COLUMNS -- fields computed as of the most
        recent auction in the table rather than as of each row's own
        auction date.
    verbose : bool
        Print what was dropped and the final block width. Worth
        leaving on: the block silently going from ~50 to ~3,500
        columns is the single most expensive thing that can happen
        here and there is otherwise nothing that reports it.
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

        ############################################################
        # Drop identity and as-of-wrong-date columns before anything
        # decides how to encode them.
        ############################################################

        blocked = set()
        if drop_identity_columns:
            blocked |= ROLE_IDENTITY_COLUMNS
        if drop_leaky_columns:
            blocked |= ROLE_LEAKY_COLUMNS

        blocked_present = sorted(blocked & set(role_df.columns))

        if blocked_present:
            role_df = role_df.drop(columns=blocked_present)
            if verbose:
                print(
                    f"  build_role_table: dropped "
                    f"{len(blocked_present)} identity/leaky columns: "
                    f"{blocked_present}"
                )

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
        skipped_high_cardinality = []

        for col in tag_columns:

            series = role_df[col]

            if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
                numeric_frame[col] = series.astype(float)
                role_columns.append(col)
            else:
                ####################################################
                # A category worth one-hot encoding is one many
                # players share (batting_style: 2 values,
                # bowling_style: 10). A column with hundreds of
                # distinct values across ~800 players is a key, and
                # one-hot encoding a key gives the model a private
                # column per player to memorise into. Refuse rather
                # than expand.
                ####################################################

                n_unique = series.nunique(dropna=True)

                if (
                    max_role_cardinality is not None
                    and n_unique > max_role_cardinality
                ):
                    skipped_high_cardinality.append((col, int(n_unique)))
                    continue

                dummies = pd.get_dummies(series, prefix=col, dummy_na=False)
                dummies = dummies.astype(float)
                numeric_frame = pd.concat([numeric_frame, dummies], axis=1)
                role_columns.extend(dummies.columns.tolist())

        if skipped_high_cardinality and verbose:
            detail = ", ".join(
                f"{c} ({n} distinct)" for c, n in skipped_high_cardinality
            )
            print(
                f"  build_role_table: skipped {len(skipped_high_cardinality)} "
                f"column(s) above max_role_cardinality="
                f"{max_role_cardinality}: {detail}"
            )

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

        ############################################################
        # A player absent from the archetype table merges to all-NaN,
        # which the dataset fills with 0 -- and an all-zero multi-hot
        # is indistinguishable from "a player with no tags at all".
        # An explicit flag lets the model tell "role unknown" from
        # "role known to be nothing", the same way the player feature
        # block already carries its *_is_missing flags.
        ############################################################

        role_missing = role_frame.isna().all(axis=1).astype(float)
        role_frame = role_frame.copy()
        role_frame["role_is_missing"] = role_missing
        role_columns = role_columns + ["role_is_missing"]

        if verbose:
            print(
                f"  build_role_table: {len(role_columns)} role features; "
                f"{int(role_missing.sum())} of {len(role_frame)} rows "
                f"unmatched in player_role_df"
            )

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
    player_context_columns=None,
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

    training_df = add_player_context_features(
        training_df,
        columns=player_context_columns,
    )

    return training_df


####################################################################
# Pre-auction facts about the player that the replay engine carries
# on every row but that metadata_columns excluded from every feature
# block, so they reached the model through no path at all.
#
# basePrice is the one that matters. It is the player's own
# reservation price, announced before the auction opens, so it is
# legitimately available at prediction time -- this is not
# hindsight. On winners its Spearman correlation with the realised
# auction price is 0.733, against roughly 0.68 for the whole trained
# model, so it is the single strongest signal in the dataset and
# nothing was reading it.
#
# cappedStatus and isPlayerOverseas fell through a gap: they are in
# metadata_columns here, AND build_role_table drops the archetype
# table's capped_status / is_overseas as "duplicates of what the
# auction-replay engine already provides". Both sides deferred to
# the other and neither supplied them.
#
# THE CAVEAT ON basePrice, because it is real:
#
#   For a 'left' row the label interval is literally
#   (0.01, basePrice). Handing basePrice to the model as an input
#   means that on 88% of training rows the upper bound of the target
#   is readable straight off a feature. A model that just predicts
#   slightly under basePrice for everyone scores well on those rows
#   without learning anything about valuation.
#
#   That is a shortcut, not leakage -- the value is genuinely known
#   pre-auction -- but a shortcut that costs winner accuracy is
#   still a loss. So this is behind a config flag, defaults are set
#   from a measured comparison rather than from taste, and the
#   winner-subset metrics are what decide it. See the sweep in
#   CHANGES.md.
####################################################################

DEFAULT_PLAYER_CONTEXT_COLUMNS = (
    "basePrice",
    "cappedStatus",
    "isPlayerOverseas",
)


####################################################################
# The boolean-ish spellings these columns actually arrive in.
#
# The auction API returns cappedStatus as CAPPED/UNCAPPED and
# isPlayerOverseas as a JSON true/false, and the scraper writes both
# straight to CSV. Neither is guaranteed to survive the round trip:
# ONE blank cell anywhere in isPlayerOverseas makes pd.read_csv give
# back dtype object holding True/False/nan, which is neither
# is_bool_dtype nor is_numeric_dtype.
#
# That matters because the fallback branch below used to be a
# hardcoded `== "CAPPED"` comparison applied to ANY non-numeric
# column. So an object-dtype isPlayerOverseas became 0.0 for every
# player: present in player_feature_columns, present in the tensor,
# carrying nothing. Measured on a synthetic frame with a single blank
# cell -- 1 distinct value, std 0.000, correlation with the raw
# column undefined. Nothing else in the pipeline reports a constant
# input, so it would have trained that way indefinitely.
#
# Mapping through an explicit vocabulary instead means an
# unrecognised spelling becomes NaN -> flagged missing -> printed,
# rather than silently becoming False.
####################################################################

CONTEXT_TRUE_TOKENS = {
    "TRUE", "T", "YES", "Y", "1", "1.0",
    "CAPPED",
    "OVERSEAS",
}

CONTEXT_FALSE_TOKENS = {
    "FALSE", "F", "NO", "N", "0", "0.0",
    "UNCAPPED",
    "INDIAN", "DOMESTIC",
}


def _context_to_binary(series):
    """
    Map a boolean / boolean-spelled-as-text column to 0.0 / 1.0 / NaN.

    Everything non-numeric goes through here, real bool dtype
    included -- True stringifies to "TRUE" and hits the vocabulary --
    so there is one code path rather than one per dtype pandas might
    have inferred from the CSV.

    Returns (values, unmapped), where `unmapped` is the sorted list of
    non-null tokens the vocabulary did not recognise.
    """

    text = series.astype("string").str.strip().str.upper()

    values = pd.Series(np.nan, index=series.index, dtype=float)
    values[text.isin(CONTEXT_TRUE_TOKENS)] = 1.0
    values[text.isin(CONTEXT_FALSE_TOKENS)] = 0.0

    known = text.isin(CONTEXT_TRUE_TOKENS) | text.isin(CONTEXT_FALSE_TOKENS)
    unmapped = sorted(set(text[text.notna() & ~known].unique()))

    return values, unmapped


def add_player_context_features(training_df, columns=None, verbose=True):
    """
    Promote pre-auction player context into the player feature block.

    Returns training_df with the numeric versions appended and
    attrs["player_feature_columns"] extended. Columns already present
    in the block, or absent from the frame, are skipped -- so calling
    this twice is a no-op, and an older frame missing one of these
    columns still builds.

    Every promoted column gets a `ctx_<name>_is_missing` companion
    UNCONDITIONALLY, even when nothing is missing.

    The flag used to be emitted only `if values.isna().any()`, which
    makes the WIDTH of player_feature_columns a function of the data.
    train_years and val_years are built by two separate
    build_training_df calls, so one unparseable base price in one
    split and not the other gives the two frames different feature
    blocks, and nothing compares them: with scalers on, the val
    frame's extra column is silently dropped (BlockScaler transforms
    the columns it was FIT on, i.e. train's); with scalers off it is
    a shape mismatch inside the first nn.Linear. Within a single
    split it is less dangerous only in that build_training_df's
    cross-year attrs check turns it into a hard build failure.

    A constant-zero flag costs three weights and makes the schema a
    property of the code instead of a property of the data.
    """

    if columns is None:
        columns = DEFAULT_PLAYER_CONTEXT_COLUMNS

    columns = list(columns or [])

    if not columns:
        return training_df

    existing = list(training_df.attrs.get("player_feature_columns", []))
    added = []
    summary = []

    for column in columns:

        if column not in training_df.columns:
            summary.append(f"{column}: NOT IN FRAME, skipped")
            continue

        target = f"ctx_{column}"

        if target in existing:
            continue

        series = training_df[column]
        unmapped = []

        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            values = pd.to_numeric(series, errors="coerce").astype(float)
        else:
            values, unmapped = _context_to_binary(series)

        training_df[target] = values
        added.append(target)

        ############################################################
        # basePrice is never missing in the current data (0.000 NaN
        # rate across all nine auctions), but a '--' base price for a
        # traded or retained player is representable and parses to
        # NaN. The flag follows the same *_is_missing convention the
        # player feature block already uses, which BlockScaler and
        # the dataset both fill to 1 rather than 0.
        ############################################################

        flag = f"{target}_is_missing"
        training_df[flag] = values.isna().astype(float)
        added.append(flag)

        ############################################################
        # A promoted column that came out constant reached the model
        # in name only. Say so here, rather than leaving it to be
        # inferred from a validation metric that did not move.
        ############################################################

        n_distinct = int(values.nunique(dropna=True))
        n_missing = int(values.isna().sum())

        line = (
            f"{column} -> {target}: {n_distinct} distinct, "
            f"{n_missing} missing "
            f"({n_missing / max(len(values), 1):.1%})"
        )

        if unmapped:
            line += f" | UNRECOGNISED {unmapped[:5]} -> NaN"

        if n_distinct <= 1:
            line += "  <-- CONSTANT, carries no signal"

        summary.append(line)

    if added:
        training_df.attrs["player_feature_columns"] = existing + added

    if verbose and summary:
        print("  add_player_context_features:")
        for line in summary:
            print(f"    {line}")

    return training_df