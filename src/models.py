import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrinsicValuationNetwork(nn.Module):

    def __init__(
        self,
        player_dim,
        num_archetypes,
        num_teams,
        embedding_dim=16,
    ):
        super().__init__()

        self.archetype_embedding = nn.Embedding(
            num_archetypes,
            embedding_dim
        )

        self.team_embedding = nn.Embedding(
            num_teams,
            embedding_dim
        )

        input_dim = player_dim + 2 * embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(64,1)
        self.sigma_head = nn.Linear(64,1)

    def forward(
        self,
        player_features,
        archetype,
        team
    ):

        a = self.archetype_embedding(archetype)
        t = self.team_embedding(team)

        x = torch.cat(
            [player_features,a,t],
            dim=1
        )

        h = self.network(x)

        mu = self.mu_head(h)

        sigma = F.softplus(
            self.sigma_head(h)
        ) + 1e-6

        return mu,sigma

class AuctionAdjustmentNetwork(nn.Module):

    def __init__(
        self,
        team_state_dim,
        auction_state_dim
    ):
        super().__init__()

        input_dim = (
            team_state_dim
            + auction_state_dim
        )

        self.network = nn.Sequential(

            nn.Linear(input_dim,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,1)

        )

    def forward(
        self,
        team_state,
        auction_state
    ):

        x = torch.cat(
            [team_state,auction_state],
            dim=1
        )

        log_phi = self.network(x)

        return log_phi

class ValuationModel(nn.Module):

    def __init__(
        self,
        player_dim,
        team_state_dim,
        auction_state_dim,
        num_archetypes,
        num_teams,
        embedding_dim=16
    ):
        super().__init__()

        # Intrinsic valuation network
        self.intrinsic = IntrinsicValuationNetwork(
            player_dim=player_dim,
            num_archetypes=num_archetypes,
            num_teams=num_teams,
            embedding_dim=embedding_dim
        )

        # Auction adjustment network
        self.auction = AuctionAdjustmentNetwork(
            team_state_dim=team_state_dim,
            auction_state_dim=auction_state_dim
        )

    def forward(
        self,
        player_features,
        archetype,
        team,
        team_state,
        auction_state
    ):
    
        # Intrinsic valuation
        mu, sigma = self.intrinsic(
            player_features,
            archetype,
            team
        )
    
        # Auction adjustment
        log_phi = self.auction(
            team_state,
            auction_state
        )
    
        # Effective valuation
        mu_effective = mu + log_phi
    
        return {
            "mu": mu,
            "sigma": sigma,
            "log_phi": log_phi,
            "mu_effective": mu + log_phi
        }