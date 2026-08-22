"""
Drop-in replacements for the "Protocol" and "Fit the folds" cells.

Paste CELL_PROTOCOL over the cell that defines `fold()`, and
CELL_FOLDS over the cell that loops over REPORT_YEARS.

NO WEIGHTING CHANGES. The loss weighting stays exactly as shipped
(`balanced`, on both splits). The only change is the epoch budget:
both editions are now fit for the same number of epochs instead of
144 and 25. See src/protocol.py for why.
"""

CELL_PROTOCOL = '''
step("protocol")

import copy
import src.training as T
from src.training import run_training_pipeline_with_holdout
from src.protocol import run_fold, sweep_epoch_budget

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

# One budget, every edition. This is the whole fix: previously 2025
# was fit for ~144 epochs and 2026 for ~25, and no player-level table
# pooled across the two meant anything as a result.
#
# 90 is a starting point, not a measured optimum -- pick it properly
# with sweep_epoch_budget (next cell, commented out) before quoting
# any result. What is known so far: 25 is far too few (Livingstone
# 1300 -> 124), 149 is past the useful point (medAE 75.5, rho 0.822),
# and 74 was better than both (68.9 / 0.842).
EPOCH_BUDGET = 90


def _run(overrides=None, **kw):
    original = copy.deepcopy(T.config)
    try:
        from src.experiments import _deep_update
        merged = _deep_update(original, overrides)
        T.config.clear(); T.config.update(merged)
        return run_training_pipeline_with_holdout(**kw)
    finally:
        T.config.clear(); T.config.update(original)


def fold(report_year, seed=0, overrides=None):
    return run_fold(
        _run,
        report_year,
        seeds=(seed,),
        years=YEARS,
        overrides=copy.deepcopy(overrides) if overrides else None,
        base_kwargs=dict(player_role_df=archetype_df_filtered,
                         archetype_df=archetypes),
        fixed_epochs=EPOCH_BUDGET,
        report_seed=seed,
    )
'''


CELL_CHOOSE_BUDGET = '''
# ---- Run ONCE to choose EPOCH_BUDGET, then set it above and skip ----
#
# Reports both objectives side by side. winner_medAE / winner_spearman
# are the "what will he go for" side; overall_within_interval is the
# per-team valuation side that the left-censored majority carries.
# Pick a budget that does not buy the first by giving up the second.

BUDGETS = (25, 50, 75, 100, 150)

sweep = sweep_epoch_budget(
    _run, REPORT_YEARS, BUDGETS, seeds=SEEDS, years=YEARS,
    overrides=copy.deepcopy(ARCH),
    base_kwargs=dict(player_role_df=archetype_df_filtered,
                     archetype_df=archetypes),
)
save_table(sweep, "epoch_budget_sweep")

cols = ["winner_medAE", "winner_spearman", "overall_within_interval"]
print(sweep.groupby("budget")[cols].median().to_string())
'''


CELL_FOLDS = '''
step("folds")

RESULTS, rows = {}, []
for year in REPORT_YEARS:
    for seed in SEEDS:
        r = fold(year, seed=seed, overrides=copy.deepcopy(ARCH))
        RESULTS[(year, seed)] = r
        row = summarize_predictions(r["val_predictions"], r["history"])
        row.update(year=year, seed=seed, epoch_used=r["epoch_from_stop_season"])
        rows.append(row)
        print(f"  {year} seed {seed}: medAE {row['winner_medAE']:.1f}"
              f" | rho {row['winner_spearman']:.3f}"
              f" | within_interval {row['overall_within_interval']:.3f}"
              f" | epoch {row['epoch_used']}", flush=True)

folds = pd.DataFrame(rows)

# Both editions must show the SAME epoch now. If they do not, the
# fixed budget is not being applied and the old transfer is back.
assert folds["epoch_used"].nunique() == 1, folds[["year", "epoch_used"]]

METRICS = ["winner_medAE", "winner_median_log_ratio", "winner_mad_log_ratio",
           "winner_median_abs_log_ratio", "winner_spearman",
           "winner_within_interval", "overall_within_interval"]
headline = folds.groupby("year")[METRICS].median().T
save_table(headline, "headline_reduced")
headline
'''


CELL_WIRING_ADDITION = '''
# --- added to the wiring cell -------------------------------------
from input_creation_2.money import parse_money
assert parse_money("2.00 Cr") == 200.0
assert parse_money("13.00 Crore") == 1300.0
assert parse_money("--") is None

import input_creation_2.player_features.demographics as _demog
assert _demog.parse_money is parse_money, "demographics kept its own regex"

from input_creation_2.auction_dataset_utils import DEGENERATE_CHECK_COLUMNS
assert "cappedStatus" in DEGENERATE_CHECK_COLUMNS
assert "basePrice" not in DEGENERATE_CHECK_COLUMNS

# The weighting must be the shipped one. An earlier patch changed it
# and selected epoch 1; this guard stops that coming back.
assert T.config["training"].get("valid_loss_weighting", "balanced") == "balanced", \\
    "validation weighting is not the shipped 'balanced' scheme"
assert "epoch_selection" not in T.config["training"], \\
    "winner-only epoch selection is back; it optimises the wrong objective"
'''


if __name__ == "__main__":
    for c in (CELL_PROTOCOL, CELL_CHOOSE_BUDGET, CELL_FOLDS,
              CELL_WIRING_ADDITION):
        print(c)
