import math

import torch
import torch.nn as nn


########################################################################
# Numerically-stable standard-normal tail machinery.
#
# The rest of this file needs log P(lower <= V < upper) for
# V ~ LogNormal(mu, sigma), i.e. log(Phi(z_upper) - Phi(z_lower)) in
# terms of standardized z-scores. The naive way -- compute Phi(z) in
# linear space, subtract, clamp to eps, then take log -- is exactly
# what silently kills gradients on badly-mispriced samples: once a
# z-score is extreme enough that Phi(z) underflows to 0.0 in float32,
# the clamp makes every sample past that point look identical to the
# loss (interval_prob == eps everywhere), so d(loss)/d(mu) is exactly
# 0 no matter how wrong the prediction is.
#
# The fix is to never materialize Phi(z) in linear space for extreme
# z. log Phi(z) is computed directly, using the standard asymptotic
# tail expansion once erfc-based evaluation would lose precision, and
# interval log-probabilities are computed via stable log-space
# subtraction (from whichever side keeps both cdf evaluations away
# from 0 or 1).
########################################################################

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_INV_SQRT2 = 0.7071067811865476  # 1 / sqrt(2)

# Below this z, erfc(-z/sqrt2) is being asked to resolve a number
# that's already ~1e-89 or smaller -- still representable in float64,
# but close enough to the edge that precision degrades before it
# underflows outright. The asymptotic series below is accurate to
# ~1e-13 (relative) by z=-20 (verified against mpmath), so it takes
# over well before that happens.
_LOG_NDTR_LOWER_THRESHOLD = -20.0


def _ndtr(z):
    """Standard normal CDF, elementwise (linear space)."""
    return 0.5 * torch.erfc(-z * _INV_SQRT2)


def _log_ndtr_asymptotic(z, series_order=6):
    """
    Asymptotic expansion of log Phi(z) for z << 0 (Mills-ratio tail
    series):

        Phi(z) ~ phi(z) / (-z) * [1 - 1/z^2 + 3/z^4 - 15/z^6 + ...]

        log Phi(z) = -z^2/2 - log(-z) - 0.5*log(2*pi) + log(series)

    This never forms Phi(z) itself, so it stays finite -- and keeps
    producing a correctly-scaled gradient -- for arbitrarily large
    |z|, unlike torch.log(Phi(z).clamp(min=eps)).
    """

    z2 = z * z
    series = torch.ones_like(z)
    term = torch.ones_like(z)
    sign = -1.0
    double_fact = 1.0

    for k in range(1, series_order + 1):
        double_fact *= (2 * k - 1)
        term = sign * double_fact / z2.pow(k)
        series = series + term
        sign = -sign

    # Defensive only: by the time this branch is reached (z past
    # _LOG_NDTR_LOWER_THRESHOLD) the series is comfortably positive;
    # this just keeps log() safe if a caller ever changes the
    # threshold without re-checking series convergence there.
    series = torch.clamp(series, min=1e-12)

    return -0.5 * z2 - torch.log(-z) - _LOG_SQRT_2PI + torch.log(series)


def log_ndtr(z, series_order=6):
    """
    Numerically stable log(Phi(z)), accurate and finite (with a
    correctly-scaled gradient) for arbitrarily large |z|.
    """

    direct_mask = z > _LOG_NDTR_LOWER_THRESHOLD

    # Route each branch's *input* through torch.where too, not just
    # the output -- otherwise the unused branch can still produce
    # inf/nan internally (e.g. 1/z^2 for z near 0 in the asymptotic
    # branch) and poison gradients even though its value is discarded.
    z_direct = torch.where(direct_mask, z, torch.full_like(z, -1.0))
    z_asym = torch.where(
        direct_mask,
        torch.full_like(z, 2.0 * _LOG_NDTR_LOWER_THRESHOLD),
        z,
    )

    direct_val = torch.log(_ndtr(z_direct))
    asym_val = _log_ndtr_asymptotic(z_asym, series_order=series_order)

    return torch.where(direct_mask, direct_val, asym_val)


def _log_diff_ndtr(z_lo, z_hi):
    """
    log(Phi(z_hi) - Phi(z_lo)) for z_lo <= z_hi, via stable log-space
    subtraction. Never forms Phi(z_hi) or Phi(z_lo) directly when
    either would underflow.
    """

    log_lo = log_ndtr(z_lo)
    log_hi = log_ndtr(z_hi)

    # log_hi >= log_lo always (Phi is increasing); clamp defensively
    # against floating-point noise right at equality.
    diff_log = torch.clamp(log_lo - log_hi, max=0.0)

    # Only bites for near-zero-width intervals (log_lo ~= log_hi);
    # bounds the result instead of ever producing log(0) = -inf.
    exp_diff = torch.clamp(torch.exp(diff_log), max=1.0 - 1e-12)

    return log_hi + torch.log1p(-exp_diff)


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

    Unlike a naive implementation, the log-probability is computed
    directly in log-space rather than by forming F(upper) - F(lower)
    in linear space and clamping the result away from 0. That clamp
    is a zero-gradient trap: once a sample is mispriced badly enough
    that the linear-space probability underflows, every worse
    prediction looks identical to the loss, so no gradient signal
    reaches it (26x reweighting a zero gradient is still zero).
    With this formulation, gradient magnitude keeps scaling with how
    wrong the prediction is, all the way into the deep tail.
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
        weight=None,
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
        # Work in float64 for the tail machinery. A badly-mispriced
        # sample is exactly the case where z-scores get large -- and
        # that's the case we most need a real gradient for -- so the
        # extra range here isn't optional precision polish, it's load
        # bearing for the fix.
        ########################################################

        mu64 = mu.double()
        sigma64 = sigma.double()

        z_lower = (torch.log(lower_bid.double()) - mu64) / sigma64
        z_upper = (torch.log(upper_bid.double()) - mu64) / sigma64

        ########################################################
        # Log-probability of the interval, computed from whichever
        # side keeps both cdf evaluations away from a 0 or 1 that
        # would lose precision on subtraction:
        #
        #   - interval entirely at/below mu -> lower-tail log_ndtr
        #     directly
        #   - interval entirely at/above mu -> mirror onto the lower
        #     tail via the survival function Phi(-z)
        #   - interval straddles mu -> plain subtraction is already
        #     well-conditioned (neither cdf value is extreme)
        ########################################################

        mask_left = z_upper <= 0
        mask_right = z_lower >= 0

        log_prob_left = _log_diff_ndtr(z_lower, z_upper)
        log_prob_right = _log_diff_ndtr(-z_upper, -z_lower)
        log_prob_straddle = torch.log(
            torch.clamp(_ndtr(z_upper) - _ndtr(z_lower), min=self.eps)
        )

        log_prob = torch.where(
            mask_left,
            log_prob_left,
            torch.where(mask_right, log_prob_right, log_prob_straddle),
        )

        ########################################################
        # Negative log-likelihood
        #
        # No clamp(max=...) here on purpose: that clamp was the same
        # zero-gradient trap as the linear-space one, just relocated
        # -- anything past the cap flattened back to zero gradient.
        # Runaway-update protection now lives at the optimizer step
        # (gradient-norm clipping in train_one_epoch), which bounds
        # the size of the update without ever zeroing an individual
        # sample's contribution to it.
        ########################################################

        nll = (-log_prob).to(mu.dtype)

        if weight is not None:
            loss = (nll * weight).sum() / weight.sum().clamp(min=self.eps)
        else:
            loss = nll.mean()

        ########################################################
        # Diagnostics only -- detached, not part of the loss graph.
        ########################################################

        with torch.no_grad():
            interval_prob = torch.exp(log_prob).to(mu.dtype)

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