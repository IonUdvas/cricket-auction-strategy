from input_creation_2.auction_dataset_utils import (
    build_training_samples,
    build_encoders,
    build_role_table,
    PlayerFeatureContext,
)
from input_creation_2.auction_dataset import IPLAuctionDataset
from valuation_model.models import *
from valuation_model.losses import *
from valuation_model.training import *
from valuation_model.scaling import fit_scalers
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import yaml

import os
import random

# Resolve relative to this file rather than a hardcoded Kaggle path, so the
# module imports on a laptop, in CI and on Kaggle alike.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# The ball data is a DIRECTORY of parquets written by data.build_bbb
# (deliveries / fielding / people / wickets / matches), not the single flat
# parquet this pipeline used to take.  Defaulting to the repo's own copy means
# a caller never has to know that.
DEFAULT_BBB_DIR = os.path.join(REPO_ROOT, "data", "bbb")
DEFAULT_RESOLUTION = os.path.join(
    REPO_ROOT, "data", "identity", "cricinfo_resolution.csv"
)

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
        bbb_dir=None,
        years=None,
        competitions=None,
        overrides=None,
        resolution=None,
        feature_context=None,
):
    """
    years      : iterable of int, optional. Defaults to all of AUCTION_DATES.
    bbb_dir    : directory holding deliveries/fielding/people parquet, as
                 written by data.build_bbb.  Was previously a path to one
                 parquet file.  Defaults to the repo's own data/bbb.
    resolution : the cricinfo identity cache.  Defaults to the repo's own
                 data/identity/cricinfo_resolution.csv -- it used to default to
                 None, which meant every hand-verified identity in that file was
                 silently ignored by the training pipeline and players like
                 Rohit Sharma trained with an empty career.
    feature_context : an already-built PlayerFeatureContext with rosters
                 already registered.  When given, identity is the caller's
                 responsibility and is NOT re-resolved here.  This is how a
                 train/val split shares one identity map; resolving each split
                 separately lets one playerId become two different cricketers,
                 which is the whole thing PlayerFeatureContext exists to stop.
    """
    if bbb_dir is None:
        bbb_dir = DEFAULT_BBB_DIR
    if resolution is None and os.path.exists(DEFAULT_RESOLUTION):
        resolution = DEFAULT_RESOLUTION

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

    training_dfs = {}
    for year, auction_date in selected_years.items():
        print(f"Building {year}...")
        training_df = build_training_samples(
            player_template.format(year=year),
            bid_template.format(year=year),
            feature_context,
            auction_date,
            auction_max_purse=AUCTION_MAX_PURSES[year],
        )
        training_dfs[year] = training_df
        print(f"Finished {year}: {len(training_df)} training rows")

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
        player_template,
        bid_template,
        bbb_dir=None,
        player_role_df=None,
):
    
    set_seed()

    full_training_df = build_training_df(player_template, bid_template, bbb_dir)

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

    # pd.concat(axis=1) returns a NEW frame and does not carry .attrs across,
    # so the column-group contract set in build_training_samples would be lost
    # here and IPLAuctionDataset would fail on player_feature_columns. Capture
    # and restore it explicitly.
    _attrs = dict(full_training_df.attrs)

    full_training_df = pd.concat(
        [full_training_df.reset_index(drop=True), role_frame],
        axis=1,
    )

    full_training_df.attrs = _attrs
    full_training_df.attrs["role_columns"] = role_columns

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


def run_training_pipeline_with_holdout(
        player_template,
        bid_template,
        bbb_dir=None,
        train_years=None,
        val_years=None,
        player_role_df=None,
        competitions=None,
        overrides=None,
        resolution=None,
        scale_features=None,
):
    """
    Same as run_training_pipeline, but builds train_years and val_years
    as separate auctions, trains only on train_years, and evaluates
    (no gradient updates) on val_years each epoch.

    scale_features : bool or None
        Fit BlockScalers on the TRAIN frame and apply them to both
        splits. None reads data.scale_features from the config
        (default True). valuation_model/scaling.py has existed, with
        a docstring explaining why raw career totals in the thousands
        cannot share an nn.Linear with 0/1 flags, and nothing in the
        repo ever called fit_scalers -- both datasets were built with
        scalers=None, the path its own docstring calls "only ever
        right for a smoke test".

    Example
    -------
    train_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    val_years = [2026]
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

    seed = set_seed()
    if seed is not None:
        print(f"seed: {seed}")

    if bbb_dir is None:
        bbb_dir = DEFAULT_BBB_DIR
    if resolution is None and os.path.exists(DEFAULT_RESOLUTION):
        resolution = DEFAULT_RESOLUTION

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

    train_df = build_training_df(
        player_template, bid_template, bbb_dir, years=train_years,
        feature_context=feature_context,
    )

    val_df = build_training_df(
        player_template, bid_template, bbb_dir, years=val_years,
        feature_context=feature_context,
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

    # Preserve existing attrs 
    train_attrs = dict(train_df.attrs) 
    val_attrs = dict(val_df.attrs)

    train_roles = role_frame.iloc[: len(train_df)].reset_index(drop=True) 
    val_roles = role_frame.iloc[len(train_df):].reset_index(drop=True)

    train_df = pd.concat( [train_df.reset_index(drop=True), train_roles], axis=1, )

    val_df = pd.concat( [val_df.reset_index(drop=True), val_roles], axis=1, )

    # Restore attrs and add role columns 
    train_df.attrs = train_attrs 
    val_df.attrs = val_attrs 
    train_df.attrs["role_columns"] = role_columns 
    val_df.attrs["role_columns"] = role_columns

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

    train_dataset = IPLAuctionDataset(train_df, encoder_manager, scalers=scalers)
    val_dataset = IPLAuctionDataset(val_df, encoder_manager, scalers=scalers)

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