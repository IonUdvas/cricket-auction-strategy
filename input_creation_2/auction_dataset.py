import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class IPLAuctionDataset(Dataset):

    def __init__(
        self,
        training_df,
        encoder_manager,
    ):

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

        self.player_features = torch.tensor(

            self.training_df[
                self.player_feature_columns
            ]
            .fillna(0)
            .to_numpy(dtype=np.float32),

            dtype=torch.float32,
        )

        self.team_state = torch.tensor(

            self.training_df[
                self.team_state_columns
            ]
            .fillna(0)
            .to_numpy(dtype=np.float32),

            dtype=torch.float32,
        )

        self.auction_state = torch.tensor(

            self.training_df[
                self.auction_state_columns
            ]
            .fillna(0)
            .to_numpy(dtype=np.float32),

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

        self.role = torch.tensor(

            self.encoder_manager
            .get_encoder("role")
            .transform(
                self.training_df["role"]
            )
            .values,

            dtype=torch.long,
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

        self.lower_bid = torch.tensor(

            self.training_df["lower"]
            .fillna(0)
            .to_numpy(dtype=np.float32),

            dtype=torch.float32,
        )

        self.upper_bid = torch.tensor(

            self.training_df["upper"]
            .fillna(0)
            .replace(
                np.inf,
                np.finfo(np.float32).max,
            )
            .to_numpy(dtype=np.float32),

            dtype=torch.float32,
        )

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

            "role":
                self.role[idx],

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