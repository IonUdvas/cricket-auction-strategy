import torch
import torch.nn as nn

from torch.distributions import LogNormal


class IntervalCensoredLoss(nn.Module):
    """
    Unified interval likelihood loss.

    Every observation is treated as a finite interval:

        left:      (0.01, basePrice)
        interval:  (lastBid, nextBid)
        right:     (winningBid, 2 * winningBid)

    The likelihood for every sample is:

        P(lower <= V < upper) = F(upper) - F(lower)

    where V follows a LogNormal(mu, sigma).
    """

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu,
        sigma,
        lower_bid,
        upper_bid,
        observation_type=None,  # kept for backward compatibility
    ):

        ########################################################
        # Shapes
        ########################################################

        mu = mu.squeeze(-1)
        sigma = sigma.squeeze(-1)

        ########################################################
        # Numerical safety
        ########################################################

        sigma = torch.clamp(sigma, min=0.05, max=3.0)

        lower_bid = torch.clamp(lower_bid, min=1e-3)
        upper_bid = torch.clamp(upper_bid, min=1e-3)

        ########################################################
        # Ensure valid intervals
        ########################################################

        upper_bid = torch.maximum(
            upper_bid,
            lower_bid + 1e-3
        )

        ########################################################
        # Distribution
        ########################################################

        dist = LogNormal(mu, sigma)

        ########################################################
        # Unified interval likelihood
        ########################################################

        upper_cdf = dist.cdf(upper_bid)
        lower_cdf = dist.cdf(lower_bid)

        interval_prob = upper_cdf - lower_cdf

        ########################################################
        # Numerical stability
        ########################################################

        interval_prob = torch.clamp(
            interval_prob,
            min=self.eps,
            max=1.0
        )

        ########################################################
        # Negative log-likelihood
        ########################################################

        nll = -torch.log(interval_prob)

        # Prevent rare pathological samples from exploding
        nll = torch.clamp(nll, max=50.0)

        loss = nll.mean()

        ########################################################
        # Return
        ########################################################

        return {

            ####################################################
            # Optimization target
            ####################################################

            "loss": loss,

            ####################################################
            # Overall likelihood
            ####################################################

            "likelihood": interval_prob.mean(),

            ####################################################
            # Mean negative log-likelihood
            ####################################################

            "nll": nll.mean(),

            ####################################################
            # Diagnostics
            ####################################################

            "interval_probability": interval_prob.mean(),

            "interval_loss": nll.mean(),

            ####################################################
            # Batch size
            ####################################################

            "num_samples": torch.tensor(
                float(interval_prob.numel()),
                device=interval_prob.device,
            ),
        }