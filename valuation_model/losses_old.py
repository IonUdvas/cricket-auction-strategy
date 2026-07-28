import torch
import torch.nn as nn

from torch.distributions import LogNormal


class IntervalCensoredLoss(nn.Module):

    OBS_LEFT = 0
    OBS_INTERVAL = 1
    OBS_RIGHT = 2
    OBS_UNKNOWN = 3

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu,
        sigma,
        lower_bid,
        upper_bid,
        observation_type,
    ):

        ########################################################
        # Remove trailing dimensions
        ########################################################

        mu = mu.squeeze(-1)
        sigma = sigma.squeeze(-1)

        ########################################################
        # Distribution
        ########################################################

        # print("Printing mus")
        # print(mu.min(), mu.max())
        # print(sigma.min(), sigma.max())
        # print(lower_bid.min(), lower_bid.max())
        # print(upper_bid.min(), upper_bid.max())

        dist = LogNormal(mu, sigma)

        ########################################################
        # Observation likelihoods
        ########################################################

        interval_prob = (
            dist.cdf(upper_bid)
            -
            dist.cdf(lower_bid)
        )

        left_prob = (
            dist.cdf(upper_bid)
        )

        right_prob = (
            1.0
            -
            dist.cdf(lower_bid)
        )

        ########################################################
        # Numerical stability
        ########################################################

        interval_prob = torch.clamp(
            interval_prob,
            min=self.eps
        )

        left_prob = torch.clamp(
            left_prob,
            min=self.eps
        )

        right_prob = torch.clamp(
            right_prob,
            min=self.eps
        )

        ########################################################
        # Masks
        ########################################################

        interval_mask = (
            observation_type == self.OBS_INTERVAL
        )

        left_mask = (
            observation_type == self.OBS_LEFT
        )

        right_mask = (
            observation_type == self.OBS_RIGHT
        )

        unknown_mask = (
            observation_type == self.OBS_UNKNOWN
        )

        valid_mask = ~unknown_mask

        ########################################################
        # Per-sample likelihood
        ########################################################

        likelihood = torch.ones_like(mu)

        likelihood[interval_mask] = (
            interval_prob[interval_mask]
        )

        likelihood[left_mask] = (
            left_prob[left_mask]
        )

        likelihood[right_mask] = (
            right_prob[right_mask]
        )

        ########################################################
        # Overall Loss
        ########################################################

        loss = -torch.log(
            likelihood[valid_mask]
        ).mean()

        ########################################################
        # Debug metrics
        ########################################################

        def masked_mean(x, mask):

            if mask.any():
                return x[mask].mean()

            return torch.tensor(
                float("nan"),
                device=x.device,
            )

        def masked_nll(x, mask):

            if mask.any():
                return -torch.log(
                    x[mask]
                ).mean()

            return torch.tensor(
                float("nan"),
                device=x.device,
            )

        ########################################################
        # Return
        ########################################################

        return {

            ####################################################
            # Optimization target
            ####################################################

            "loss":
                loss,

            ####################################################
            # Overall likelihood
            ####################################################

            "likelihood":
                masked_mean(
                    likelihood,
                    valid_mask
                ),

            ####################################################
            # Backward compatibility
            ####################################################

            "winner_probability":
                masked_mean(
                    right_prob,
                    right_mask
                ),

            "loser_probability":
                masked_mean(
                    interval_prob,
                    interval_mask
                ),

            ####################################################
            # New metrics
            ####################################################

            "interval_probability":
                masked_mean(
                    interval_prob,
                    interval_mask
                ),

            "left_probability":
                masked_mean(
                    left_prob,
                    left_mask
                ),

            "right_probability":
                masked_mean(
                    right_prob,
                    right_mask
                ),

            ####################################################
            # Individual losses
            ####################################################

            "interval_loss":
                masked_nll(
                    interval_prob,
                    interval_mask
                ),

            "left_loss":
                masked_nll(
                    left_prob,
                    left_mask
                ),

            "right_loss":
                masked_nll(
                    right_prob,
                    right_mask
                ),

            ####################################################
            # Batch composition
            ####################################################

            "num_interval":
                interval_mask.sum().float(),

            "num_left":
                left_mask.sum().float(),

            "num_right":
                right_mask.sum().float(),

            "num_unknown":
                unknown_mask.sum().float(),
        }