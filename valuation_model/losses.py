import torch
import torch.nn as nn
from torch.distributions import LogNormal


class IntervalCensoredLoss(nn.Module):

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu,
        sigma,
        lower_bid,
        upper_bid,
        winner
    ):

        # Remove trailing dimension if present
        mu = mu.squeeze(-1)
        sigma = sigma.squeeze(-1)

        dist = LogNormal(mu, sigma)

        loser_prob = (
            dist.cdf(upper_bid)
            -
            dist.cdf(lower_bid)
        )

        winner_prob = (
            1
            -
            dist.cdf(lower_bid)
        )

        likelihood = torch.where(
            winner,
            winner_prob,
            loser_prob
        )

        likelihood = torch.clamp(
            likelihood,
            min=self.eps
        )

        loss = -torch.log(
            likelihood
        ).mean()

        return {

            "loss": loss,

            "likelihood": likelihood.mean(),

            "winner_probability": winner_prob.mean(),

            "loser_probability": loser_prob.mean()

        }