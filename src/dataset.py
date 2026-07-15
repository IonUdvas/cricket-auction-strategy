import torch
from torch.utils.data import Dataset

class DummyAuctionDataset(Dataset):

    def __init__(
        self,
        n_samples=5000,
        player_dim=120,
        team_state_dim=30,
        auction_state_dim=25,
        num_archetypes=8,
        num_teams=10
    ):

        self.player_features = torch.randn(n_samples, player_dim)

        self.team_state = torch.randn(n_samples, team_state_dim)

        self.auction_state = torch.randn(n_samples, auction_state_dim)

        self.team = torch.randint(0, num_teams, (n_samples,))

        self.archetype = torch.randint(0, num_archetypes, (n_samples,))

        # Dummy interval data
        self.lower_bid = torch.rand(n_samples) * 100
        self.upper_bid = self.lower_bid + 0.5

        # Random winner indicator
        self.is_winner = torch.randint(0,2,(n_samples,)).bool()

    def __len__(self):
        return len(self.team)

    def __getitem__(self, idx):

        return {
            "player_features": self.player_features[idx],
            "team_state": self.team_state[idx],
            "auction_state": self.auction_state[idx],
            "team": self.team[idx],
            "archetype": self.archetype[idx],
            "lower_bid": self.lower_bid[idx],
            "upper_bid": self.upper_bid[idx],
            "winner": self.is_winner[idx]
        }