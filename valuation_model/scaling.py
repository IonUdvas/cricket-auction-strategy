"""
Input scaling for the valuation model.

Nothing in this pipeline normalised its numeric inputs before this
module existed.  The three numeric blocks that reach the network are:

    player_feature_columns   raw career totals (bat_runs in the
                             thousands, bat_balls in the thousands)
                             sitting next to rates in [0, 1] and 0/1
                             missing-flags
    team_state_columns       remaining_purse, 0 .. 12500 (lakh)
    auction_state_columns    auction_order / players_remaining,
                             0 .. ~400

Feeding those straight into `nn.Linear` with default (Kaiming-uniform,
fan_in) init gives a pre-activation whose spread is set by the largest
column, not by the target.  Measured at initialisation on
representative magnitudes, `log_phi` has a standard deviation of
roughly 300 nats -- while the entire target range is log(0.01) = -4.6
to log(2700) = 7.9, a span of 12.5 nats.  exp() of that is where the
279,497-lakh predictions come from.

Scaling is fit on TRAIN ONLY and applied to validation, the same way
the encoders are fit once and shared.  Unlike the encoder vocabulary
(which carries no outcome information), a scaler fit on train+val
would leak the validation year's feature distribution, so it is not
done here.
"""

import numpy as np
import pandas as pd


class BlockScaler:
    """
    Per-column: optional log1p compression, then z-score.

    log1p is applied to heavy-tailed non-negative count columns
    (career runs, balls, purse, auction_order) because a z-score alone
    leaves them skewed enough that a handful of Kohli-sized careers
    still dominate the first layer.  Columns already on a bounded
    scale -- rates, shares, 0/1 flags -- are z-scored only.

    The decision is made from the TRAIN distribution and then frozen,
    so a column can never be compressed in train and not in val.
    """

    def __init__(self, log_threshold=50.0):
        self.log_threshold = float(log_threshold)
        self.columns = None
        self.log_mask = None
        self.mean = None
        self.std = None

    def fit(self, frame, columns):
        self.columns = list(columns)

        raw = self._raw(frame)

        # Heavy-tailed and non-negative -> compress.  Two conditions,
        # both from percentiles rather than the max, so one outlier
        # row cannot flip a column's treatment:
        #   * large enough to matter at all (p99 above the threshold)
        #   * actually skewed (p99 far above the median) -- this is
        #     what separates bat_runs, where the top of the range is
        #     ~50x the middle, from bat_strike_rate, where a p99 of
        #     ~180 sits close to a median of ~130 and log() would
        #     only throw away resolution.
        p99 = np.nanpercentile(raw, 99, axis=0)
        p50 = np.nanpercentile(raw, 50, axis=0)
        non_negative = np.nanmin(raw, axis=0) >= 0.0

        self.log_mask = (
            non_negative
            & (p99 > self.log_threshold)
            & (p99 > 3.0 * (p50 + 1.0))
        )

        compressed = self._compress(raw)

        self.mean = np.nanmean(compressed, axis=0)
        std = np.nanstd(compressed, axis=0)

        # A column that is constant in train carries no information;
        # dividing by ~0 would turn float noise into a huge input.
        self.std = np.where(std < 1e-6, 1.0, std)

        return self

    def transform(self, frame):
        if self.columns is None:
            raise RuntimeError("BlockScaler.transform called before fit")

        raw = self._raw(frame)
        compressed = self._compress(raw)

        scaled = (compressed - self.mean) / self.std

        # Val can hold values outside anything train saw (a career
        # that kept growing, a purse size that changed).  Clip rather
        # than let one row re-scale a whole batch.
        return np.clip(scaled, -10.0, 10.0).astype(np.float32)

    def fit_transform(self, frame, columns):
        return self.fit(frame, columns).transform(frame)

    # ------------------------------------------------------------------

    def _raw(self, frame):
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"scaler fit on columns absent from frame: {missing}")

        block = frame[self.columns]

        # `*_is_missing` flags mean "this metric was undefined".  A
        # left-merge miss (an unresolved playerId) leaves the whole
        # row NaN, and filling those flags with 0 asserts the metric
        # WAS observed -- the opposite of the truth.  Flags fill to 1,
        # values fill to 0, matching PlayerFeatureBuilder.missing_fill.
        fill = {
            c: (1.0 if c.endswith("_is_missing") else 0.0)
            for c in self.columns
        }

        return block.fillna(fill).to_numpy(dtype=np.float64)

    def _compress(self, raw):
        out = raw.copy()
        out[:, self.log_mask] = np.log1p(
            np.clip(out[:, self.log_mask], 0.0, None)
        )
        return out


def fit_scalers(train_df):
    """Fit one BlockScaler per numeric block, on the training frame."""
    return {
        block: BlockScaler().fit(train_df, train_df.attrs[block])
        for block in (
            "player_feature_columns",
            "team_state_columns",
            "auction_state_columns",
        )
    }