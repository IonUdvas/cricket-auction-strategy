import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrinsicValuationNetwork(nn.Module):

    def __init__(
        self,
        player_dim,
        num_role_features,
        num_teams,
        embedding_dim=16,
    ):
        super().__init__()

        # A player's role is a multi-hot vector (a player can be
        # RHB *and* pace *and* death_overs_bowler *and*
        # bowling_allrounder at once) rather than one mutually
        # exclusive category, so this is a Linear projection rather
        # than an nn.Embedding lookup. Note this is a strict
        # generalization, not a behavior change for the old
        # single-role case: a Linear layer (no bias) applied to a
        # one-hot vector is mathematically identical to an embedding
        # lookup -- each column of the weight matrix is exactly the
        # "embedding" for that role/tag -- and for a multi-hot input
        # it naturally sums the tag embeddings of every active tag.
        self.role_proj = nn.Linear(
            num_role_features,
            embedding_dim,
            bias=False,
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
        role_features,
        team
    ):

        a = self.role_proj(role_features)
        t = self.team_embedding(team)

        x = torch.cat(
            [player_features,a,t],
            dim=1
        )

        h = self.network(x)

        # h = x

        # for i, layer in enumerate(self.network):

        #     h = layer(h)

        #     print(i, layer, torch.isfinite(h).all())

        #     if not torch.isfinite(h).all():
        #         print(h)
        #         raise RuntimeError

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
        num_role_features,
        num_teams,
        embedding_dim=16
    ):
        super().__init__()
        self.intrinsic = IntrinsicValuationNetwork(
            player_dim=player_dim,
            num_role_features=num_role_features,
            num_teams=num_teams,
            embedding_dim=embedding_dim
        )

        self.auction = AuctionAdjustmentNetwork(
            team_state_dim=team_state_dim,
            auction_state_dim=auction_state_dim
        )

    def forward(
        self,
        player_features,
        role_features,
        team,
        team_state,
        auction_state
    ):

        assert torch.isfinite(player_features).all(), "player_features contains NaN/Inf"
        assert torch.isfinite(team_state).all(), "team_state contains NaN/Inf"
        assert torch.isfinite(auction_state).all(), "auction_state contains NaN/Inf"

        assert torch.isfinite(role_features).all()
        assert torch.isfinite(team.float()).all()
    
        mu, sigma = self.intrinsic(
            player_features,
            role_features,
            team
        )
    
        log_phi = self.auction(
            team_state,
            auction_state
        )
    
        mu_effective = mu + log_phi

        if not torch.isfinite(mu).all():
            raise RuntimeError("mu is NaN")

        if not torch.isfinite(log_phi).all():
            raise RuntimeError("log_phi is NaN")

        if not torch.isfinite(sigma).all():
            raise RuntimeError("sigma is NaN")

        mu_effective = mu + log_phi

        if not torch.isfinite(mu_effective).all():
            raise RuntimeError("mu_effective is NaN")
    
        return {
            "mu": mu,
            "sigma": sigma,
            "log_phi": log_phi,
            "mu_effective": mu_effective
        }