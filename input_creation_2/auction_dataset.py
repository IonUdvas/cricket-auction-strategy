import numpy as np
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
        # team/player pairs are worth very little". These weights
        # rescale each observation_type to contribute equally in
        # aggregate, so the comparatively rare but far more
        # informative "interval"/"right" (actual bid/sale) rows
        # aren't drowned out.
        ########################################################

        type_counts = self.training_df["observation_type"].value_counts()

        num_classes = len(type_counts)
        total_rows = len(self.training_df)

        class_weight = {
            obs_type: total_rows / (num_classes * count)
            for obs_type, count in type_counts.items()
        }

        self.sample_weight = torch.tensor(
            self.training_df["observation_type"]
            .map(class_weight)
            .to_numpy(dtype=np.float32),

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