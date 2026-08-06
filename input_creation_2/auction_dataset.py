import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class IPLAuctionDataset(Dataset):

    def __init__(
        self,
        training_df,
        encoder_manager,
        scalers=None,
        weighting="balanced",
    ):
        """
        scalers : dict of BlockScaler, keyed by attrs group name, as
                  returned by valuation_model.scaling.fit_scalers.
                  Fit on the TRAIN frame and passed to both datasets.
                  None keeps the old raw-feature behaviour, which is
                  only ever right for a smoke test -- see scaling.py.

        weighting : "balanced" or "uniform".

            "balanced" is the class-balanced scheme documented below
            and is what training wants.

            "uniform" gives every row weight 1.0, and is what a
            VALIDATION set wants. The balanced weights are derived
            from price brackets cut on each dataset's OWN interval
            midpoints -- and for a winner the interval is [P, kP), so
            the bracket is a function of the label. Applying them to
            the validation set makes the reported "validation NLL" a
            label-reweighted quantity whose weights are refitted on
            every split, which has two consequences worth knowing
            before quoting it:

              1. It is not comparable across editions. A 182-sale
                 mega auction and a 77-sale small one produce
                 different strata and therefore different weights, so
                 2.51 on one and 2.66 on the other are not two
                 measurements of the same thing.

              2. Early stopping is driven by it. The epoch chosen is
                 the epoch that minimises a reweighting of the
                 validation labels, not the validation likelihood.

            That is a large part of why validation NLL has ranked
            configurations differently from every point-estimate
            metric. Set training.valid_loss_weighting: uniform in the
            config to score and stop on the plain held-out likelihood
            instead.
        """

        if weighting not in ("balanced", "uniform"):
            raise ValueError(
                f"weighting must be 'balanced' or 'uniform', got "
                f"{weighting!r}"
            )
        self.weighting = weighting

        self.training_df = training_df.copy()

        ########################################################
        # Drop rows with no real signal.
        #
        # "unknown" rows (e.g. every non-retaining team on a
        # retained player) have lower=NaN, upper=NaN by design --
        # they mean "we don't know this team's valuation", not
        # "this team valued the player near zero". Keeping them
        # and fillna(0)-ing the bounds would silently turn every
        # one of these into a fake (0.001, 0.002) interval, which
        # trains the model to think most team/player pairs are
        # worthless. So they're excluded here, before any tensors
        # are built.
        ########################################################

        self.training_df = (
            self.training_df[
                self.training_df["observation_type"] != "unknown"
            ]
            .reset_index(drop=True)
        )

        self.encoder_manager = encoder_manager

        ########################################################
        # Column groups
        ########################################################

        self.player_feature_columns = (
            training_df.attrs["player_feature_columns"]
        )

        self.team_state_columns = (
            training_df.attrs["team_state_columns"]
        )

        self.auction_state_columns = (
            training_df.attrs["auction_state_columns"]
        )

        ########################################################
        # Numerical Features
        ########################################################

        self.scalers = scalers

        def numeric_block(columns, scaler_key):
            """
            Scaled when a fitted scaler is supplied, raw otherwise.

            Raw is kept only so an unscaled run stays reproducible for
            comparison; it is not a supported training path.  The
            feature blocks span career totals in the thousands,
            purses up to 12500 and rates in [0, 1] all at once, and an
            nn.Linear reads that as "the biggest column is the only
            column".
            """
            if scalers is not None:
                return scalers[scaler_key].transform(self.training_df)

            fill = {
                c: (1.0 if c.endswith("_is_missing") else 0.0)
                for c in columns
            }

            return (
                self.training_df[columns]
                .fillna(fill)
                .to_numpy(dtype=np.float32)
            )

        self.player_features = torch.tensor(
            numeric_block(
                self.player_feature_columns,
                "player_feature_columns",
            ),
            dtype=torch.float32,
        )

        self.team_state = torch.tensor(
            numeric_block(
                self.team_state_columns,
                "team_state_columns",
            ),
            dtype=torch.float32,
        )

        self.auction_state = torch.tensor(
            numeric_block(
                self.auction_state_columns,
                "auction_state_columns",
            ),
            dtype=torch.float32,
        )

        ########################################################
        # Encoded categorical inputs
        ########################################################

        self.team = torch.tensor(

            self.encoder_manager
            .get_encoder("team")
            .transform(
                self.training_df["team"]
            )
            .values,

            dtype=torch.long,
        )

        self.role_columns = (
            training_df.attrs["role_columns"]
        )

        self.role_features = torch.tensor(

            self.training_df[
                self.role_columns
            ]
            .fillna(0)
            .to_numpy(dtype=np.float32),

            dtype=torch.float32,
        )

        self.observation_type = torch.tensor(

            self.encoder_manager
            .get_encoder("observation_type")
            .transform(
                self.training_df["observation_type"]
            )
            .values,

            dtype=torch.long,
        )

        ########################################################
        # Class-balanced sample weights
        #
        # "left" (never-bid) rows are typically the large majority
        # -- every team that didn't bid on a player gets one -- so
        # an unweighted mean loss mostly teaches the model "most
        # team/player pairs are worth very little".
        #
        # Balancing by observation_type alone still leaves a
        # second imbalance inside "right"/"interval": a handful of
        # marquee, high-price sales get outvoted by many
        # ordinary-priced ones of the *same* type, so the model
        # keeps regressing expensive players toward the pack.
        #
        # So weights are stratified on the *joint* key
        # (observation_type, price_bracket) rather than type alone.
        # This is deliberately NOT type_weight * price_weight --
        # multiplying two independently-balanced weights would let
        # a rare-type + rare-price row (e.g. one very expensive
        # winner) dominate a whole batch. Balancing the compound
        # key directly keeps every (type, price bracket) bucket
        # contributing equally, without that runaway effect.
        ########################################################

        if self.weighting == "uniform":
            # Every row counts once. See the docstring: the balanced
            # weights below are cut on label-derived price brackets,
            # which is right for training and wrong for scoring.
            self.sample_weight = torch.ones(
                len(self.training_df), dtype=torch.float32
            )
        else:
            log_midpoint = np.log(
                np.clip(
                    (
                        self.training_df["lower"].to_numpy(dtype=np.float64)
                        + self.training_df["upper"].to_numpy(dtype=np.float64)
                    )
                    / 2.0,
                    1e-3,
                    None,
                )
            )

            num_price_bins = 6

            def _bin_within_type(group):
                try:
                    return pd.qcut(
                        group,
                        q=num_price_bins,
                        labels=False,
                        duplicates="drop",
                    )
                except (ValueError, IndexError):
                    # Too few rows/unique values in this observation_type
                    # to form quantile bins (e.g. a small validation
                    # split) -- collapse to a single price bracket for
                    # that type.
                    return pd.Series(
                        np.zeros(len(group), dtype=int),
                        index=group.index,
                    )

            price_bin = (
                pd.Series(log_midpoint, index=self.training_df.index)
                .groupby(self.training_df["observation_type"])
                .transform(_bin_within_type)
            )

            strata = (
                self.training_df["observation_type"].astype(str)
                + "_"
                + pd.Series(price_bin, index=self.training_df.index).astype(str)
            )

            strata_counts = strata.value_counts()

            num_strata = len(strata_counts)
            total_rows = len(self.training_df)

            strata_weight = {
                key: total_rows / (num_strata * count)
                for key, count in strata_counts.items()
            }

            self.sample_weight = torch.tensor(
                strata.map(strata_weight).to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )

        ########################################################
        # Targets
        ########################################################

        ########################################################
        # Bounds are NOT filled.
        #
        # fillna(0) here was the last line of defence that stopped a
        # broken interval from ever being visible: a losing-bidder row
        # whose upper bound the replay engine could not determine
        # arrived as NaN, became 0, and the loss then clamped it up to
        # lower + 1e-3 -- a label asserting that team's valuation to
        # three decimal places, carrying a huge NLL, on a row that
        # should not have existed at all.
        #
        # "unknown" rows are already gone by this point, so anything
        # still NaN is a genuine defect upstream and should be fixed
        # there, not papered over here.
        ########################################################

        lower = self.training_df["lower"].to_numpy(dtype=np.float64)
        upper = (
            self.training_df["upper"]
            .replace(np.inf, np.finfo(np.float32).max)
            .to_numpy(dtype=np.float64)
        )

        bad = (
            ~np.isfinite(lower)
            | ~np.isfinite(upper)
            | (upper <= lower)
            | (lower <= 0)
        )

        if bad.any():
            offenders = (
                self.training_df.loc[
                    bad, ["playerName", "team", "observation_type"]
                ]
                .assign(lower=lower[bad], upper=upper[bad])
                .head(10)
            )
            raise ValueError(
                f"{int(bad.sum())} of {len(self.training_df)} rows have an "
                f"unusable (lower, upper) interval. These used to be "
                f"silently filled with 0. First offenders:\n{offenders}"
            )

        self.lower_bid = torch.tensor(lower, dtype=torch.float32)
        self.upper_bid = torch.tensor(upper, dtype=torch.float32)

        ########################################################
        # Useful metadata
        ########################################################

        self.winner = torch.tensor(

            self.training_df["winner"]
            .fillna(False)
            .astype(bool)
            .to_numpy(),

            dtype=torch.bool,
        )

    def __len__(self):

        return len(self.training_df)

    def __getitem__(self, idx):

        return {

            ####################################################
            # Numerical inputs
            ####################################################

            "player_features":
                self.player_features[idx],

            "team_state":
                self.team_state[idx],

            "auction_state":
                self.auction_state[idx],

            ####################################################
            # Categorical inputs
            ####################################################

            "team":
                self.team[idx],

            "role_features":
                self.role_features[idx],

            ####################################################
            # Targets
            ####################################################

            "lower_bid":
                self.lower_bid[idx],

            "upper_bid":
                self.upper_bid[idx],

            "observation_type":
                self.observation_type[idx],

            "weight":
                self.sample_weight[idx],

            ####################################################
            # Metadata
            ####################################################

            "winner":
                self.winner[idx],
        }