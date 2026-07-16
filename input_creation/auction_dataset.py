import numpy as np
import torch

from torch.utils.data import Dataset


class IPLAuctionDataset(Dataset):

    def __init__(
        self,
        training_df,
        encoder_manager
    ):

        self.training_df = training_df.copy()

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
        # Numerical tensors
        ########################################################

        self.player_features = torch.tensor(

            self.training_df[
                self.player_feature_columns
            ]
            .fillna(0)
            .to_numpy(dtype=np.float32),

            dtype=torch.float32
        )

        self.team_state = torch.tensor(

            self.training_df[
                self.team_state_columns
            ]
            .fillna(0)
            .to_numpy(dtype=np.float32),

            dtype=torch.float32
        )

        self.auction_state = torch.tensor(

            self.training_df[
                self.auction_state_columns
            ]
            .fillna(0)
            .to_numpy(dtype=np.float32),

            dtype=torch.float32
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

            dtype=torch.long
        )

        self.archetype = torch.tensor(

            self.encoder_manager
            .get_encoder("role")
            .transform(
                self.training_df["role"]
            )
            .values,

            dtype=torch.long
        )

        ########################################################
        # Targets
        ########################################################

        self.lower_bid = torch.tensor(

            self.training_df["lower"]
            .to_numpy(dtype=np.float32),

            dtype=torch.float32
        )

        self.upper_bid = torch.tensor(

            self.training_df["upper"]
            .replace(
                np.inf,
                np.finfo(np.float32).max
            )
            .to_numpy(dtype=np.float32),

            dtype=torch.float32
        )

        self.is_winner = torch.tensor(

            self.training_df["winner"]
            .astype(bool)
            .to_numpy(),

            dtype=torch.bool
        )

    def __len__(self):

        return len(self.training_df)

    def __getitem__(self, idx):

        return {

            "player_features":
                self.player_features[idx],

            "team_state":
                self.team_state[idx],

            "auction_state":
                self.auction_state[idx],

            "team":
                self.team[idx],

            "archetype":
                self.archetype[idx],

            "lower_bid":
                self.lower_bid[idx],

            "upper_bid":
                self.upper_bid[idx],

            "winner":
                self.is_winner[idx]
        }