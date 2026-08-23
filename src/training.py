import os
import random
import sys

# Path bootstrap FIRST, before any project import.
#
# This block used to sit below the imports and read REPO_ROOT five lines
# before REPO_ROOT was assigned, so importing this module raised
# `NameError: name 'REPO_ROOT' is not defined` -- every time, on every
# machine. Even once that is fixed the order still matters: on Kaggle the
# notebook's working directory is /kaggle/working, not the clone, so
# `input_creation_2` is not importable until the repo root is on sys.path,
# and the imports below are what need it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import yaml

import data_sources as ds

from input_creation_2.auction_dataset_utils import (
    build_training_samples,
    build_encoders,
    build_role_table,
    build_demographic_features,
    PlayerFeatureContext,
    DEFAULT_PLAYER_CONTEXT_COLUMNS,
)
from input_creation_2.auction_dataset import IPLAuctionDataset
from input_creation_2.verification import (
    verify_year,
    verify_feature_monotonicity,
)
from valuation_model.models import *
from valuation_model.losses import *
from valuation_model.training import *
from valuation_model.scaling import fit_scalers
from torch.utils.data import DataLoader

# configs/ is code, not data: it is small, versioned, and belongs next to the
# thing it configures. It is the one path in this module resolved against the
# repo rather than through data_sources.
CONFIG_PATH = os.environ.get(
    "CRICKET_CONFIG", os.path.join(REPO_ROOT, "configs", "default.yaml")
)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def set_seed(seed=None):
    """
    Seed python / numpy / torch.

    configs/default.yaml has carried `seed: 42` with a comment saying
    it was "now actually applied, via set_seed()". No such function
    existed anywhere in the repo, so every run to date has had a
    different init, a different shuffle order and a different
    train/val curve. That matters more than usual here: with a
    validation set of 1,560 rows and 77 winners, run-to-run noise is
    comparable to the effect sizes being compared.
    """

    if seed is None:
        seed = config.get("seed")
    if seed is None:
        return None

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def _model_dims(frame, encoder_manager):
    """Runtime dims, read off the built frame rather than the config."""

    return {
        "player_dim": len(frame.attrs["player_feature_columns"]),
        "team_state_dim": len(frame.attrs["team_state_columns"]),
        "auction_state_dim": len(frame.attrs["auction_state_columns"]),
        "num_role_features": len(frame.attrs["role_columns"]),
        "num_teams": len(encoder_manager.get_encoder("team").classes_),
    }


def build_optimizer(model, cfg=None):
    """
    Adam, with weight decay applied only where it means something.

    Decaying biases, the sigma head and the two embedding tables is
    not regularisation of the same kind as decaying a weight matrix:
    it pulls the mu-head bias away from its log(mu_prior)
    initialisation and shrinks every team embedding toward a shared
    zero, which is a prior nobody asked for. The weight matrices are
    what actually overfit.
    """

    cfg = (cfg or config)["training"]
    decay_value = cfg.get("weight_decay", 0.0)

    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.Adam(
        [
            {"params": decay, "weight_decay": decay_value},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.get("learning_rate", 3e-4),
    )


def _train_kwargs(cfg=None):
    """
    Early-stopping settings, read from the config.

    `train()` has taken patience / min_delta / restore_best since it
    was written and neither pipeline ever passed them, so the three
    early_stopping_* keys in configs/default.yaml did nothing. The
    150-epoch run in the last log is the visible consequence: it ran
    to completion, restored epoch 27 out of luck because
    restore_best defaults to True, and spent 123 epochs after the
    best one driving validation loss from 2.68 to 6.25.
    """

    cfg = (cfg or config)["training"]

    return {
        "epochs": cfg.get("epochs", 150),
        "patience": cfg.get("early_stopping_patience"),
        "min_delta": cfg.get("early_stopping_min_delta", 0.0),
        "restore_best": cfg.get("restore_best_weights", True),
        "max_grad_norm": cfg.get("max_grad_norm", 5.0),
    }

# The ball data is a DIRECTORY of parquets written by pipelines.build_bbb
# (deliveries / fielding / people / wickets / matches), not the single flat
# parquet this pipeline used to take.
#
# Every one of these is resolved LAZILY, through data_sources, at the moment
# it is needed -- never at import, and never against the repo. Two reasons,
# and both have already cost a run:
#
#   1. Module-level path constants freeze whatever was visible at import
#      time. On Kaggle the repo is cloned before the datasets are guaranteed
#      to be mounted, so a constant computed at import can be a path to
#      nothing while the real mount appears a second later.
#   2. A repo-relative default is worse than no default. It resolves,
#      silently, to a stale local copy, and the run reports numbers that
#      cannot be reproduced from any dataset version.
#
# There are therefore no DEFAULT_BBB_DIR / DEFAULT_RESOLUTION constants. Call
# the functions.
def default_bbb_dir():
    """The bbb parquet set, from the inputs dataset or this session's build."""
    return ds.bbb_dir()


def default_resolution():
    """
    The hand-verified identity cache, or None.

    None is a legitimate answer -- the pipeline runs without it -- so this
    does NOT raise. It is `required=False` inside data_sources and the caller
    decides. What it must never do is fall back to a repo path.
    """
    return ds.resolution_path()


def default_player_template():
    """completed_players_{year}.csv, from the auction dataset."""
    return ds.player_template()


def default_bid_template():
    """auction_trail_{year}.csv, from the auction dataset."""
    return ds.bid_template()


def default_archetype_df():
    """
    The RAW player_archetypes.csv, for the replay engine.

    Distinct from the `player_role_df` passed to build_role_table. That
    one is filtered down to the tag columns that become the model's role
    vector; this one must still carry `pace`, `RA` and `LA`, because
    build_archetype_tags derives right_arm_pace / left_arm_pace from
    their conjunction and asserts that every paced player has exactly
    one arm set. Hand it the filtered frame and it raises.

    Loaded by default rather than left to the caller: the engine's
    archetype_df argument has existed since the archetype counters were
    written and no call site ever passed it, so every model to date
    trained on the three legacy role counters instead. A default that
    has to be opted out of fails in the safer direction.
    """
    return pd.read_csv(ds.archetypes_path())

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
        player_template=None,
        bid_template=None,
        bbb_dir=None,
        years=None,
        competitions=None,
        overrides=None,
        resolution=None,
        feature_context=None,
        player_context_columns=None,
        archetype_df=None,
        verify=None,
        verify_strict=None,
):
    """
    player_template / bid_template : paths containing "{year}".  Default to
                 the auction Kaggle dataset via data_sources.
    years      : iterable of int, optional. Defaults to all of AUCTION_DATES.
    bbb_dir    : directory holding deliveries/fielding/people parquet, as
                 written by pipelines.build_bbb.  Was previously a path to one
                 parquet file.  Defaults to the inputs Kaggle dataset.
    resolution : the cricinfo identity cache, resolved from the inputs Kaggle
                 dataset.  It used to default to None, which meant every
                 hand-verified identity in that file was silently ignored by
                 the training pipeline and players like Rohit Sharma trained
                 with an empty career.
    feature_context : an already-built PlayerFeatureContext with rosters
                 already registered.  When given, identity is the caller's
                 responsibility and is NOT re-resolved here.  This is how a
                 train/val split shares one identity map; resolving each split
                 separately lets one playerId become two different cricketers,
                 which is the whole thing PlayerFeatureContext exists to stop.
    player_context_columns : iterable of str, or None.
                 Pre-auction facts on the replay-engine row to promote into the
                 player feature block -- basePrice, cappedStatus,
                 isPlayerOverseas by default. None reads
                 data.player_context_columns from the config; an explicit empty
                 list turns the promotion off entirely, which is the ablation
                 the basePrice caveat in auction_dataset_utils calls for.
                 build_training_samples has taken this argument since the
                 promotion was added and nothing ever passed it, so the
                 "behind a config flag" in that comment was not true.
    verify     : bool or None. Run input_creation_2.verification after each
                 year and the cross-year monotonicity check after all of
                 them. None reads data.verify from the config (default True).

                 verification.py's docstring said this function called it.
                 It did not -- nothing imported the module at all, so the
                 checks that exist specifically to catch "right shape, wrong
                 numbers" had never run on a single build. They are cheap
                 (seconds against minutes for the build itself) and print
                 only findings, so the default is on.
    verify_strict : bool or None. Turn `error`-severity findings into a
                 ValueError instead of a printed line. None reads
                 data.verify_strict from the config (default False).

                 Off by default deliberately: an error-severity finding here
                 means the frame is wrong, but some of these checks have
                 never been run against real data, so the first few builds
                 should report rather than halt. Turn it on per the module
                 docstring once a year is known good.
    """
    if player_template is None:
        player_template = default_player_template()
    if bid_template is None:
        bid_template = default_bid_template()
    if bbb_dir is None:
        bbb_dir = default_bbb_dir()
    if resolution is None:
        resolution = default_resolution()

    selected_years = (
        AUCTION_DATES
        if years is None
        else {y: AUCTION_DATES[y] for y in years}
    )

    if feature_context is None:
        # Built once, not once per year.
        feature_context = PlayerFeatureContext(
            bbb_dir,
            competitions=competitions,
            overrides=overrides,
            resolution=resolution,
        )

        # Identity is resolved across ALL years at once, before any year is
        # built, so one playerId cannot become two different cricketers.
        rosters = {
            year: pd.read_csv(player_template.format(year=year))
            for year in selected_years
        }
        feature_context.register_rosters(rosters)

    if player_context_columns is None:
        player_context_columns = (config.get("data", {}) or {}).get(
            "player_context_columns",
            list(DEFAULT_PLAYER_CONTEXT_COLUMNS),
        )

    ####################################################################
    # The archetype table defaults to loading rather than to None.
    #
    # None is what every call site has effectively passed since the
    # archetype counters were written, and the engine treats it as
    # "use the legacy three role counters" without complaint. That
    # default is the reason the supply/demand/scarcity block has never
    # appeared in a training frame. Pass archetype_df=False to opt out
    # deliberately -- distinct from None, which now means "load it".
    ####################################################################

    if archetype_df is None:
        archetype_df = default_archetype_df()
    elif archetype_df is False:
        archetype_df = None

    data_cfg = config.get("data", {}) or {}
    if verify is None:
        verify = data_cfg.get("verify", True)
    if verify_strict is None:
        verify_strict = data_cfg.get("verify_strict", False)

    training_dfs = {}
    findings = []
    for year, auction_date in selected_years.items():
        print(f"Building {year}...")
        training_df = build_training_samples(
            player_template.format(year=year),
            bid_template.format(year=year),
            feature_context,
            auction_date,
            auction_max_purse=AUCTION_MAX_PURSES[year],
            player_context_columns=player_context_columns,
            archetype_df=archetype_df,
            # The SEASON, not the calendar year of auction_date. The
            # two differ for every edition auctioned in December --
            # see the note in build_training_samples.
            auction_season=year,
        )
        training_dfs[year] = training_df
        print(f"Finished {year}: {len(training_df)} training rows")

        ############################################################
        # Verify here, on the single-year frame, not later on the
        # concatenation. Half these checks are only meaningful per
        # auction: "rows vs players x teams" and "winner rows vs sold
        # players" both compare against that year's roster, and both
        # are trivially satisfied by a concatenated frame no matter
        # how wrong any individual year is.
        ############################################################

        if verify:
            frame = verify_year(
                year,
                training_df,
                engine_report=training_df.attrs.get("engine_report"),
            )
            findings.append(frame)

    if verify and findings:
        all_findings = pd.concat(findings, ignore_index=True)

        # Cross-year as-of check. Needs every year at once, so it can
        # only run here -- and it is the one check that can catch as-of
        # leakage without a second source of truth.
        verify_feature_monotonicity(training_dfs)

        errors = all_findings[all_findings["severity"] == "error"]
        if len(errors) and verify_strict:
            lines = "\n".join(
                f"  {r.year} {r.check}: {r.value} {r.detail}"
                for r in errors.itertuples()
            )
            raise ValueError(
                f"verification found {len(errors)} error-severity "
                f"finding(s) and data.verify_strict is on:\n{lines}"
            )
        if len(errors):
            print(
                f"\n  VERIFICATION: {len(errors)} error-severity finding(s) "
                f"across {errors['year'].nunique()} year(s). Listed above. "
                f"Set data.verify_strict to make these raise."
            )

    # pandas silently drops attrs to {} when concatenated frames disagree, and
    # IPLAuctionDataset then dies with a bare KeyError far from the cause.
    frames = list(training_dfs.values())
    reference = dict(frames[0].attrs)
    for year, frame in zip(selected_years, frames):
        for key in ("player_feature_columns", "auction_state_columns",
                    "team_state_columns", "bid_summary_columns"):
            if frame.attrs.get(key) != reference.get(key):
                a = set(reference.get(key, []))
                b = set(frame.attrs.get(key, []))
                raise ValueError(
                    f"{year} disagrees with {list(selected_years)[0]} on "
                    f"attrs[{key!r}]: only in first {sorted(a - b)}, "
                    f"only in {year} {sorted(b - a)}"
                )

    full_training_df = pd.concat(training_dfs.values(), ignore_index=True)

    # engine_report is a per-auction diagnostic. `reference` is year one's
    # attrs, so carrying it through would label the whole frame with the
    # first auction's replay stats -- a number that looks authoritative and
    # describes one ninth of the rows. It has already been consumed by
    # verify_year above.
    reference.pop("engine_report", None)

    full_training_df.attrs = reference

    return full_training_df

def load_and_encode_data(full_training_df, scalers=None):
    encoder_manager = build_encoders(full_training_df)

    dataset = IPLAuctionDataset(
        full_training_df,
        encoder_manager,
        scalers=scalers,
    )

    # Was hardcoded to 64 while configs/default.yaml said 256, so the
    # single-frame pipeline and the holdout pipeline trained at
    # different batch sizes and only one of them was the configured
    # one.
    loader = DataLoader(
        dataset,
        batch_size=config["training"].get("batch_size", 256),
        shuffle=True
    )

    return encoder_manager, dataset, loader

def run_training_pipeline(
        player_template=None,
        bid_template=None,
        bbb_dir=None,
        player_role_df=None,
        archetype_df=None,
):
    
    set_seed()

    # Left as None deliberately: build_training_df resolves them, so there is
    # exactly one place that knows the defaults.
    full_training_df = build_training_df(
        player_template, bid_template, bbb_dir, archetype_df=archetype_df,
    )

    ################################################################
    # Player roles, built once on the full concatenated frame -- see
    # the note in build_training_samples for why this doesn't happen
    # per-year.
    ################################################################

    data_cfg = config.get("data", {}) or {}

    role_frame, role_columns = build_role_table(
        full_training_df,
        player_role_df=player_role_df,
        max_role_cardinality=data_cfg.get("max_role_cardinality", 24),
        drop_identity_columns=data_cfg.get("drop_role_identity_columns", True),
        drop_leaky_columns=data_cfg.get("drop_role_leaky_columns", True),
    )

    # Age and last salary. Same position as the role table and for the
    # same reason -- last_salary needs every year present at once, and
    # build_training_df has just concatenated them.
    demo_frame, demo_columns = build_demographic_features(
        full_training_df,
        player_role_df=player_role_df,
    )

    # pd.concat(axis=1) returns a NEW frame and does not carry .attrs across,
    # so the column-group contract set in build_training_samples would be lost
    # here and IPLAuctionDataset would fail on player_feature_columns. Capture
    # and restore it explicitly.
    _attrs = dict(full_training_df.attrs)

    full_training_df = pd.concat(
        [full_training_df.reset_index(drop=True), role_frame, demo_frame],
        axis=1,
    )

    full_training_df.attrs = _attrs
    full_training_df.attrs["role_columns"] = role_columns
    full_training_df.attrs["player_feature_columns"] = (
        list(_attrs["player_feature_columns"]) + demo_columns
    )

    # No holdout here, so there is no split to leak across: the
    # scalers are fit on the one frame that exists.
    scalers = (
        fit_scalers(full_training_df)
        if data_cfg.get("scale_features", True)
        else None
    )

    encoder_manager, dataset, loader = load_and_encode_data(
        full_training_df, scalers=scalers
    )

    dims = _model_dims(full_training_df, encoder_manager)
    config["model"].update(dims)

    model = ValuationModel.from_config(config, dims)

    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            print(name, "contains NaNs/Infs")

    criterion = IntervalCensoredLoss()

    optimizer = build_optimizer(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    history = train(
        model=model,
        train_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        **_train_kwargs(),
    )

    return model, history, encoder_manager, dataset, loader, full_training_df


def prepare_holdout_data(
        player_template=None,
        bid_template=None,
        bbb_dir=None,
        train_years=None,
        val_years=None,
        player_role_df=None,
        archetype_df=None,
        competitions=None,
        overrides=None,
        resolution=None,
):
    """
    STAGE 1 of the holdout pipeline: everything that does NOT depend on the
    seed, the model config, or the training config.

    Returns (train_df, val_df, encoder_manager, role_columns).

    This is split out because it is the expensive half and it is constant
    across a sweep. One call reads deliveries.parquet (~2.4M rows), resolves
    identity across all nine auction rosters, and runs nine auction replays --
    minutes of work -- and the ONLY config keys that reach it are:

        data.player_context_columns
        data.max_role_cardinality
        data.drop_role_identity_columns
        data.drop_role_leaky_columns

    Nothing under model.* or training.*, and not the seed. So a sweep of 36
    model/training configs x 5 seeds needs ONE call to this function, not 180.
    See src/sweep.py, which caches on exactly those four keys.

    `data.scale_features` deliberately lives in stage 2 instead: scalers are
    cheap to fit and keeping them out of the cache key means toggling scaling
    does not force a rebuild.
    """

    ################################################################
    # Build train / val dataframes as separate auction replays --
    # but off ONE feature context, not two.
    #
    # Two contexts means two independent identity resolutions over two
    # different sets of years, which is precisely the per-year splitting
    # PlayerFeatureContext was written to prevent: the same auction playerId
    # can land on one cricketer in train and another in val, and the model is
    # then evaluated on a player it never saw. It also read and aggregated 2.4M
    # deliveries twice for no benefit.
    ################################################################

    # No set_seed() here. Stage 1 is seed-independent by construction --
    # replaying an auction and aggregating deliveries draws no random
    # numbers -- and seeding inside it would be misleading about that.
    # The wrapper below seeds before stage 2, where it matters.

    if player_template is None:
        player_template = default_player_template()
    if bid_template is None:
        bid_template = default_bid_template()
    if bbb_dir is None:
        bbb_dir = default_bbb_dir()
    if resolution is None:
        resolution = default_resolution()

    feature_context = PlayerFeatureContext(
        bbb_dir,
        competitions=competitions,
        overrides=overrides,
        resolution=resolution,
    )

    # Registered over train + val together, so identity is one map across the
    # whole split.
    all_years = list(train_years) + list(val_years)
    feature_context.register_rosters({
        year: pd.read_csv(player_template.format(year=year))
        for year in all_years
    })

    # archetype_df goes to BOTH splits or the team/auction state blocks
    # come out different widths and the cross-split attrs check below
    # fires. It is threaded explicitly rather than left to each call's
    # default so that opting out (archetype_df=False) opts both out.
    train_df = build_training_df(
        player_template, bid_template, bbb_dir, years=train_years,
        feature_context=feature_context, archetype_df=archetype_df,
    )

    val_df = build_training_df(
        player_template, bid_template, bbb_dir, years=val_years,
        feature_context=feature_context, archetype_df=archetype_df,
    )

    ################################################################
    # Train and val are two separate build_training_df calls, and
    # build_training_df only checks the attrs contract ACROSS YEARS
    # WITHIN its own call -- nothing compared the two splits.
    #
    # It matters now that the player block carries columns derived
    # from the auction row rather than only from the ball data: a
    # column whose presence depends on the data (an is_missing flag
    # emitted only when something was missing, which is what
    # add_player_context_features used to do) can differ between the
    # splits. With scalers on, the divergence is invisible --
    # BlockScaler transforms the columns it was fit on, so the val
    # frame's extra column is silently dropped and the model trains
    # on a val block that is not the val block. This check is cheap
    # and turns that into a message.
    ################################################################

    for key in ("player_feature_columns", "team_state_columns",
                "auction_state_columns"):
        if train_df.attrs.get(key) != val_df.attrs.get(key):
            a = set(train_df.attrs.get(key, []))
            b = set(val_df.attrs.get(key, []))
            raise ValueError(
                f"train and val disagree on attrs[{key!r}]: "
                f"only in train {sorted(a - b)}, only in val {sorted(b - a)}. "
                f"The two splits must present the same feature block."
            )

    ################################################################
    # Fit encoders on the UNION of train + val categories.
    # This only fixes team/observation_type vocabulary -- it does
    # not leak any auction outcome / target information -- but is
    # required because the encoder has no "unknown" bucket and will
    # crash on a category it has never seen.
    ################################################################

    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    encoder_manager = build_encoders(combined_df)

    ################################################################
    # Player roles, built once on the same train+val union and then
    # split back by position. player_role_df itself is a player-level
    # table (no auction outcome in it), so this isn't a leakage
    # concern the way the encoder fit above is -- the point here is
    # just guaranteeing train and val land on one shared set of role
    # columns, the same way they share one team/observation_type
    # vocabulary. Building it separately per split could otherwise
    # give each split a different column set (most visibly for the
    # legacy one-hot fallback, e.g. val_years happening not to
    # contain any WICKETKEEPER-listed player that auction).
    ################################################################

    data_cfg = config.get("data", {}) or {}

    role_frame, role_columns = build_role_table(
        combined_df,
        player_role_df=player_role_df,
        max_role_cardinality=data_cfg.get("max_role_cardinality", 24),
        drop_identity_columns=data_cfg.get("drop_role_identity_columns", True),
        drop_leaky_columns=data_cfg.get("drop_role_leaky_columns", True),
    )

    ################################################################
    # Age and last salary, built on the same train+val union and for
    # a stronger reason than the role table: last_salary is defined
    # against the auctions BEFORE this row's, so a 2026 val row needs
    # the 2025 train auction to have a value at all. Built per split,
    # every val player would be a debutant.
    #
    # The join is strictly backward, so this direction of sharing
    # does not leak: no row of either split can see its own auction's
    # prices, and no train row can see anything from a val year.
    ################################################################

    ################################################################
    # The earnings tables are the salary source, when they resolve.
    #
    # data_sources has exposed earnings_template() all along and
    # nothing called it. The auction trail alone cannot see a retained
    # player's money -- the replay engine removes retentions from the
    # pool before emitting rows -- so last_salary was missing for
    # exactly the established players it matters most for.
    ################################################################

    earnings_frames = None
    try:
        template = ds.earnings_template(required=False)
    except Exception:
        template = None

    if template:
        seasons = sorted(set(list(train_years) + list(val_years)))
        loaded = []
        for season in seasons:
            path = template.format(year=season)
            if os.path.exists(path):
                loaded.append(pd.read_csv(path))
        if loaded:
            earnings_frames = loaded
        else:
            print(
                "  earnings template resolved but no file matched; "
                "falling back to auction-trail prices, which cannot see "
                "retentions."
            )
    else:
        print(
            "  no earnings tables found -- last_salary will come from the "
            "auction trail only and every retained player will look like a "
            "debutant."
        )

    ################################################################
    # Cricbuzz debut dates -> capped status, when available.
    #
    # Optional and silent when absent: the pipeline runs without it,
    # just with `capped` constant and flagged missing. Resolved the
    # same way every other data file is, so nothing needs a path.
    ################################################################

    ################################################################
    # Cricbuzz debut dates -> capped status, when available.
    #
    # Resolved the same way every other data file is: by FILENAME
    # through data_sources, so it can live in the inputs dataset
    # (identity/cricbuzz_debuts.csv) and needs no path passed. Falls
    # back to /kaggle/working for the session that first scrapes it,
    # before it has been uploaded. CRICBUZZ_DEBUTS overrides both.
    #
    # Optional and silent when absent: the pipeline runs without it,
    # just with `capped` constant and flagged missing.
    ################################################################

    debut_df = None
    debut_path = os.environ.get("CRICBUZZ_DEBUTS")

    if not debut_path:
        try:
            debut_path = ds.find_file("cricbuzz_debuts", extensions=(".csv",),
                                      required=False)
        except Exception:
            debut_path = None

    if not debut_path:
        for cand in ("/kaggle/working/cricbuzz_debuts.csv",
                     "cricbuzz_debuts.csv"):
            if os.path.exists(cand):
                debut_path = cand
                break

    if debut_path and os.path.exists(debut_path):
        try:
            debut_df = pd.read_csv(debut_path)
            print(f"  debut table: {len(debut_df)} players from {debut_path}")
        except Exception as exc:
            print(f"  debut table at {debut_path} could not be read ({exc}); "
                  f"capped will be all-missing")
    else:
        print(
            "  no cricbuzz debut table found -- `capped` will be constant. "
            "Build one with pipelines/scrape_cricbuzz_profiles.py, then put "
            "cricbuzz_debuts.csv in the inputs dataset under identity/."
        )

    demo_frame, demo_columns = build_demographic_features(
        combined_df,
        player_role_df=player_role_df,
        earnings_frames=earnings_frames,
        debut_df=debut_df,
    )

    # Preserve existing attrs 
    train_attrs = dict(train_df.attrs) 
    val_attrs = dict(val_df.attrs)

    train_roles = role_frame.iloc[: len(train_df)].reset_index(drop=True) 
    val_roles = role_frame.iloc[len(train_df):].reset_index(drop=True)

    train_demo = demo_frame.iloc[: len(train_df)].reset_index(drop=True)
    val_demo = demo_frame.iloc[len(train_df):].reset_index(drop=True)

    train_df = pd.concat( [train_df.reset_index(drop=True), train_roles, train_demo], axis=1, )

    val_df = pd.concat( [val_df.reset_index(drop=True), val_roles, val_demo], axis=1, )

    # Restore attrs and add role columns 
    train_df.attrs = train_attrs 
    val_df.attrs = val_attrs 
    train_df.attrs["role_columns"] = role_columns 
    val_df.attrs["role_columns"] = role_columns

    ################################################################
    # Demographics join the PLAYER block rather than becoming a block
    # of their own: they are per-player facts known pre-auction, and
    # last_salary is in lakhs, so it needs the log compression
    # BlockScaler already applies to the player block's career totals.
    # A separate block would need its own scaler and its own model
    # input head for four columns.
    ################################################################

    for frame in (train_df, val_df):
        frame.attrs["player_feature_columns"] = (
            list(frame.attrs["player_feature_columns"]) + demo_columns
        )

    return train_df, val_df, encoder_manager, role_columns


def train_from_prepared(prepared, seed=None, scale_features=None):
    """
    STAGE 2: everything that DOES depend on the seed and on model/training
    config. Cheap by comparison -- a few thousand parameters over ~15k rows.

    `prepared` is the tuple returned by prepare_holdout_data.

    On the seed: set_seed() is called HERE, immediately before anything
    stochastic. run_training_pipeline_with_holdout still calls it at the top
    as it always did, so calling that function directly behaves exactly as
    before. The two orderings agree as long as stage 1 draws no random
    numbers -- which it should not, but "should not" is not "does not", so
    src/sweep.py ships check_build_rng_neutral() to prove it on your data
    before you trust a cached sweep.
    """
    train_df, val_df, encoder_manager, role_columns = prepared
    set_seed(seed)

    data_cfg = config.get("data", {}) or {}

    ################################################################
    # Input scaling: fit on TRAIN only, apply to both.
    #
    # Not optional on this data. The player block holds bat_runs up
    # to 10,673 and bowl_runs up to 12,080 in the same nn.Linear as
    # 0/1 missing-flags and rates in [0, 1]; team_state holds
    # remaining_purse up to 11,050. At default Kaiming init the
    # pre-activation spread is set by whichever column is largest,
    # so the first layer reads three or four career-total columns
    # and treats the other ~85 as noise.
    ################################################################

    if scale_features is None:
        scale_features = data_cfg.get("scale_features", True)

    scalers = fit_scalers(train_df) if scale_features else None

    if scalers is None:
        print(
            "WARNING: scale_features is off -- raw career totals and "
            "purses go straight into nn.Linear. See valuation_model/"
            "scaling.py."
        )

    ################################################################
    # Validation weighting.
    #
    # The train dataset is always class-balanced -- that is what the
    # weighting is for. The validation dataset has been balanced too,
    # by nothing more than both datasets going through the same
    # constructor, and that is a reporting bug rather than a choice:
    # the strata are cut on each split's own interval midpoints, and
    # a winner's interval is [P, kP), so the weights are a function
    # of the labels and are refitted per split. "Validation NLL" has
    # therefore been a label-reweighted number that is not comparable
    # across editions, and early stopping has been minimising it.
    #
    # Default stays "balanced" so that nothing changes silently
    # underneath existing results. Set
    #
    #     training:
    #       valid_loss_weighting: uniform
    #
    # to stop and report on the plain held-out likelihood, which is
    # what the paper means by validation NLL.
    ################################################################

    valid_weighting = (
        config.get("training", {}).get("valid_loss_weighting", "balanced")
    )

    train_dataset = IPLAuctionDataset(train_df, encoder_manager, scalers=scalers)
    val_dataset = IPLAuctionDataset(
        val_df, encoder_manager, scalers=scalers, weighting=valid_weighting,
    )

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

    dims = _model_dims(train_df, encoder_manager)
    config["model"].update(dims)

    model = ValuationModel.from_config(config, dims)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_rows = len(train_dataset)
    n_informative = int(
        (train_dataset.training_df["observation_type"] != "left").sum()
    )

    print(
        f"model: {n_params:,} trainable parameters | "
        f"{n_rows:,} training rows ({n_informative:,} non-left) | "
        f"{n_params / max(n_informative, 1):.1f} params per informative row"
    )

    criterion = IntervalCensoredLoss()

    optimizer = build_optimizer(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    history = train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        valid_loader=val_loader,
        **_train_kwargs(),
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

def run_training_pipeline_with_holdout(
        player_template=None,
        bid_template=None,
        bbb_dir=None,
        train_years=None,
        val_years=None,
        player_role_df=None,
        archetype_df=None,
        competitions=None,
        overrides=None,
        resolution=None,
        scale_features=None,
        prepared=None,
):
    """
    Build train_years and val_years as separate auctions off ONE feature
    context, train on train_years, evaluate on val_years each epoch.

    Signature and return value are unchanged; this is now a thin wrapper over
    prepare_holdout_data + train_from_prepared, which exist so a sweep can
    call the first once and the second many times.

    prepared : the tuple from prepare_holdout_data, optional.
        Pass it to skip the rebuild. Everything else about the run is
        identical. This is the entire reason a 5-seed sweep does not need to
        read 2.4M deliveries five times.

    scale_features : bool or None
        Fit BlockScalers on the TRAIN split and apply to both. None reads
        data.scale_features from the config (default True).
        valuation_model/scaling.py has existed, with a docstring explaining
        why raw career totals in the thousands cannot share an nn.Linear with
        0/1 flags, and nothing in the repo ever called fit_scalers -- both
        datasets were built with scalers=None, the path its own docstring
        calls "only ever right for a smoke test".

    Example
    -------
    train_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    val_years = [2026]
    """
    seed = set_seed()
    if seed is not None:
        print(f"seed: {seed}")

    if prepared is None:
        prepared = prepare_holdout_data(
            player_template=player_template,
            bid_template=bid_template,
            bbb_dir=bbb_dir,
            train_years=train_years,
            val_years=val_years,
            player_role_df=player_role_df,
            archetype_df=archetype_df,
            competitions=competitions,
            overrides=overrides,
            resolution=resolution,
        )

    return train_from_prepared(prepared, seed=seed,
                               scale_features=scale_features)