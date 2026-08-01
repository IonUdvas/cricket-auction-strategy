import math

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
        sigma_min=0.05,
        sigma_max=1.5,
        mu_prior=50.0,
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

        # Bound sigma smoothly in [sigma_min, sigma_max] via a
        # sigmoid, rather than an unbounded softplus that only gets
        # clamped later inside the loss. torch.clamp has zero
        # gradient outside its range, so once raw sigma drifted past
        # the loss's clamp ceiling there was nothing pulling it back
        # down -- and nothing stopping it from growing further, since
        # doing so cost nothing once already clamped. A wide sigma
        # then makes almost any interval look plausible regardless of
        # mu, which is a cheap way to reduce loss without learning
        # accurate point valuations. Sigmoid bounding removes that
        # escape hatch structurally: sigma literally cannot leave
        # [sigma_min, sigma_max], and the gradient stays nonzero
        # everywhere (just small at the extremes), so there's always
        # some pull back toward the interior.
        self.sigma_min = sigma_min
        # sigma is the standard deviation in LOG space, so 3.0 spans
        # e^(6*3) ~= 6e7 across a 3-sigma band -- wide enough that a
        # median of 279,497 still puts real mass on an actual sale at
        # 1,575, and the interval likelihood barely notices.  That's
        # the hedging escape hatch the sigmoid bounding was meant to
        # close, just at a ceiling too high to close it: it shows up
        # as within_3sigma_band ~= 0.94 sitting next to
        # within_interval ~= 0.09.  1.5 still allows a ~8000x
        # 3-sigma band, which is ample for auction prices spanning
        # 30 lakh to 27 crore.
        self.sigma_max = sigma_max

        ############################################################
        # Anchor mu where prices actually live.
        #
        # With a zeroed final weight and the bias at log(typical
        # price), the network starts by predicting the same sensible
        # median for everyone and learns deviations from there,
        # instead of starting at whatever a random projection of the
        # feature block happens to produce and having to travel back.
        ############################################################

        nn.init.zeros_(self.mu_head.weight)
        nn.init.constant_(self.mu_head.bias, math.log(mu_prior))

        nn.init.zeros_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)

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

        mu = self.mu_head(h)

        sigma = self.sigma_min + (
            self.sigma_max - self.sigma_min
        ) * torch.sigmoid(self.sigma_head(h))

        return mu,sigma

class AuctionAdjustmentNetwork(nn.Module):

    def __init__(
        self,
        team_state_dim,
        auction_state_dim,
        max_log_phi=1.5,
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

        )

        self.head = nn.Linear(64,1)

        ############################################################
        # log_phi is a *correction* to an intrinsic valuation, not a
        # valuation in its own right.  Two changes enforce that.
        #
        # Zero init: the model starts at phi = 1 (no adjustment) and
        # has to earn every departure from it.  Previously log_phi
        # was a random projection of remaining_purse (0..12500) and
        # auction_order (1..400), which at initialisation has a
        # spread of hundreds of nats against a target range of about
        # 12.5 -- so exp(mu + log_phi) was astronomically wrong
        # before a single gradient step, and the first thing the
        # optimiser had to do was spend its budget dragging this head
        # back toward zero.
        #
        # tanh bound: caps the adjustment at e^1.5, i.e. phi in
        # roughly [0.22, 4.5].  Team context can plausibly move a
        # valuation by a factor of a few -- a team with a full purse
        # and one slot left will overpay -- but not by a factor of
        # 100, and leaving the head unbounded let it absorb the
        # entire prediction instead of adjusting one.
        ############################################################

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.max_log_phi = max_log_phi

    def forward(
        self,
        team_state,
        auction_state
    ):

        x = torch.cat(
            [team_state,auction_state],
            dim=1
        )

        log_phi = self.max_log_phi * torch.tanh(
            self.head(self.network(x))
        )

        return log_phi

class ValuationModel(nn.Module):

    def __init__(
        self,
        player_dim,
        team_state_dim,
        auction_state_dim,
        num_role_features,
        num_teams,
        embedding_dim=16,
        sigma_min=0.05,
        sigma_max=1.5,
        mu_prior=50.0,
        max_log_phi=1.5,
    ):
        super().__init__()
        self.intrinsic = IntrinsicValuationNetwork(
            player_dim=player_dim,
            num_role_features=num_role_features,
            num_teams=num_teams,
            embedding_dim=embedding_dim,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            mu_prior=mu_prior,
        )

        self.auction = AuctionAdjustmentNetwork(
            team_state_dim=team_state_dim,
            auction_state_dim=auction_state_dim,
            max_log_phi=max_log_phi,
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