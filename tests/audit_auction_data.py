"""
Audit the auction inputs before they reach the model.

Every check here corresponds to a way the replay engine currently
absorbs bad input silently -- producing a plausible-looking training
row rather than raising.  The point is to turn each of those into a
number you can look at, per year.

Usage
-----
    python -m tests.audit_auction_data \\
        --players "path/to/players_{year}.csv" \\
        --bids    "path/to/bids_{year}.csv"

Exit code is non-zero if any BLOCKER fires.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from input_creation_2.auction_replay_engine import AuctionReplayEngine
from src.training import AUCTION_DATES, AUCTION_MAX_PURSES


class Report:
    def __init__(self):
        self.rows = []
        self.blockers = 0

    def add(self, year, severity, check, count, detail=""):
        self.rows.append(
            {
                "year": year,
                "severity": severity,
                "check": check,
                "count": count,
                "detail": detail,
            }
        )
        if severity == "BLOCKER" and count:
            self.blockers += 1

    def frame(self):
        return pd.DataFrame(self.rows)


def audit_year(year, player_path, bid_path, report):
    player_df = pd.read_csv(player_path)
    bid_df = pd.read_csv(bid_path)

    engine = AuctionReplayEngine(
        bid_df=bid_df,
        player_df=player_df,
        auction_max_purse=AUCTION_MAX_PURSES[year],
    )

    # _normalize_inputs has run in the constructor, so engine.player_df
    # is parsed, filtered, role-normalised and (assumed) in forward
    # auction order.  engine.bid_df is parsed and re-numbered.
    players = engine.player_df
    bids = engine.bid_df

    # ------------------------------------------------------------------
    # 1. Auction-order direction
    #
    # The engine assumes the roster CSV arrives in REVERSE order and
    # reverses it.  Nothing validates that.  A forward-ordered file
    # gives every auction_state and team_state feature backwards.
    #
    # The test: replay purses under the assumed order and under the
    # opposite order.  A real auction never lets a team's cumulative
    # spend exceed its purse; a reversed one usually does, because the
    # expensive marquee players end up "bought" after the purse has
    # already been drained.
    # ------------------------------------------------------------------

    def max_overdraft(frame):
        sold = frame[
            frame["auctionStatus"].isin(
                [engine.STATUS_SOLD, engine.STATUS_RTM, engine.STATUS_RETAINED]
            )
        ]
        worst = 0.0
        for team, grp in sold.groupby("playsForTeam"):
            spend = grp["auctionPrice"].fillna(0).cumsum()
            worst = max(worst, float((spend - AUCTION_MAX_PURSES[year]).max()))
        return worst

    forward = max_overdraft(players)
    backward = max_overdraft(players.iloc[::-1])

    report.add(
        year,
        "BLOCKER" if forward > backward else "ok",
        "auction order direction",
        int(forward > backward),
        f"overdraft as-replayed {forward:.0f} vs reversed {backward:.0f} "
        f"(lower is the more plausible order)",
    )

    # ------------------------------------------------------------------
    # 2. Winner is not the top bidder
    #
    # _summary_sold matches the winner by playsForTeam.  When that
    # string never appears in the player's bid ladder, the team that
    # actually paid is scored as "never bid" and the real top bidder
    # falls into the losing-bidder branch with next_bid = NaN.
    # ------------------------------------------------------------------

    sold = players[
        players["auctionStatus"].isin([engine.STATUS_SOLD, engine.STATUS_RTM])
    ]

    winner_absent = []
    winner_not_top = []
    no_bids_at_all = []

    by_player = {pid: g for pid, g in bids.groupby("playerId")}

    for _, p in sold.iterrows():
        ladder = by_player.get(p["playerId"])

        if ladder is None or ladder.empty:
            no_bids_at_all.append(p["playerName"])
            continue

        teams_in_ladder = set(ladder["Team"].dropna())

        if p["playsForTeam"] not in teams_in_ladder:
            winner_absent.append(
                (p["playerName"], p["auctionStatus"], p["playsForTeam"])
            )
            continue

        top = ladder.loc[ladder["BidAmount"].idxmax(), "Team"]
        if top != p["playsForTeam"]:
            winner_not_top.append((p["playerName"], p["playsForTeam"], top))

    report.add(
        year,
        "BLOCKER",
        "winner absent from bid ladder",
        len(winner_absent),
        "; ".join(f"{n} [{s}] -> {t}" for n, s, t in winner_absent[:5]),
    )

    report.add(
        year,
        "BLOCKER",
        "winner is not the top bidder",
        len(winner_not_top),
        "; ".join(f"{n}: paid={w} top={t}" for n, w, t in winner_not_top[:5]),
    )

    report.add(
        year,
        "warn",
        "sold with no bid ladder at all",
        len(no_bids_at_all),
        "; ".join(no_bids_at_all[:5]),
    )

    # ------------------------------------------------------------------
    # 3. Degenerate intervals from a NaN upper bound
    #
    # This is the one the dataset's fillna(0) hides: a losing-bidder
    # row with upper = NaN becomes (last_bid, last_bid + 0.001) by the
    # time the loss sees it.
    # ------------------------------------------------------------------

    training = engine.replay()["training"]

    interval_rows = training[training["observation_type"] == "interval"]

    nan_upper = interval_rows["upper"].isna().sum()

    report.add(
        year,
        "BLOCKER",
        "interval rows with NaN upper (become width-0.001 labels)",
        int(nan_upper),
        "these survive as fake near-exact valuations, not as dropped rows",
    )

    inverted = (
        interval_rows["upper"].notna()
        & (interval_rows["upper"] <= interval_rows["lower"])
    ).sum()

    report.add(
        year,
        "BLOCKER",
        "interval rows with upper <= lower",
        int(inverted),
    )

    # ------------------------------------------------------------------
    # 4. Observations silently discarded
    # ------------------------------------------------------------------

    retained_count = int(
        (player_df["auctionStatus"].astype(str).str.strip().str.upper()
         == engine.STATUS_RETAINED).sum()
    )

    report.add(
        year,
        "warn",
        "retained players producing zero training rows",
        retained_count,
        "_summary_retained is unreachable: _apply_preauction_events "
        "drops these from player_df before replay",
    )

    unsold = players[players["auctionStatus"] == engine.STATUS_UNSOLD]
    unsold_with_bids = sum(
        1
        for _, p in unsold.iterrows()
        if p["playerId"] in by_player and not by_player[p["playerId"]].empty
    )

    report.add(
        year,
        "warn",
        "unsold players who received bids (dropped as 'unknown')",
        unsold_with_bids,
        "_summary_unsold only handles teams that never entered",
    )

    unknown_rows = int((training["observation_type"] == "unknown").sum())
    report.add(
        year,
        "info",
        "rows dropped as observation_type='unknown'",
        unknown_rows,
        f"{unknown_rows / max(len(training), 1):.1%} of emitted rows",
    )

    # ------------------------------------------------------------------
    # 5. Purse arithmetic
    # ------------------------------------------------------------------

    final_state = pd.DataFrame(engine.team_state).T
    negative_purse = int((final_state["remaining_purse"] < 0).sum())

    report.add(
        year,
        "BLOCKER",
        "teams finishing with negative purse",
        negative_purse,
        final_state["remaining_purse"].round(0).to_dict()
        if negative_purse
        else "",
    )

    over_squad = int((final_state["remaining_slots"] < 0).sum())
    report.add(year, "warn", "teams exceeding squad size", over_squad)

    over_overseas = int(
        (final_state["overseas_bought"] > engine.overseas_limit).sum()
    )
    report.add(year, "warn", "teams exceeding overseas limit", over_overseas)

    # ------------------------------------------------------------------
    # 6. Price sanity
    # ------------------------------------------------------------------

    below_base = int(
        (
            sold["auctionPrice"].notna()
            & sold["basePrice"].notna()
            & (sold["auctionPrice"] < sold["basePrice"])
        ).sum()
    )
    report.add(year, "BLOCKER", "sold below base price", below_base)

    missing_price = int(sold["auctionPrice"].isna().sum())
    report.add(
        year,
        "BLOCKER",
        "sold with no parseable auctionPrice",
        missing_price,
        "these silently subtract NaN from the buying team's purse",
    )

    # Repeated exact prices within one team are how a mis-joined price
    # column shows up -- two uncapped players at an identical 1420.0
    # is far more likely to be a merge artefact than an auction.
    dupes = (
        sold[sold["auctionPrice"].notna()]
        .groupby(["playsForTeam", "auctionPrice"])
        .size()
    )
    repeated = dupes[(dupes > 1) & (dupes.index.get_level_values(1) > 500)]

    report.add(
        year,
        "warn",
        "same team, same price >5cr, multiple players",
        int(repeated.sum()),
        "; ".join(f"{t} @ {p:.0f} x{n}" for (t, p), n in repeated.items()),
    )

    # ------------------------------------------------------------------
    # 7. Row shape
    # ------------------------------------------------------------------

    dup_rows = int(training.duplicated(["playerId", "team"]).sum())
    report.add(
        year,
        "BLOCKER",
        "duplicate (playerId, team) training rows",
        dup_rows,
        "one player appearing twice in the roster fans out across every team",
    )

    expected = len(players) * len(engine.teams)
    report.add(
        year,
        "BLOCKER" if len(training) != expected else "ok",
        "row count == players x teams",
        int(len(training) != expected),
        f"{len(training)} emitted, {expected} expected "
        f"({len(players)} players x {len(engine.teams)} teams)",
    )

    report.add(
        year,
        "warn",
        "teams discovered",
        len(engine.teams),
        ", ".join(engine.teams),
    )


def audit_feature_blocks(training_df, report):
    """
    Checks on the attrs contract, run once on a built training frame.
    """
    auction_cols = set(training_df.attrs["auction_state_columns"])
    team_cols = set(training_df.attrs["team_state_columns"])
    player_cols = set(training_df.attrs["player_feature_columns"])

    overlap = auction_cols & team_cols
    report.add(
        "all",
        "BLOCKER",
        "auction_state columns duplicated inside team_state",
        len(overlap),
        "team_state_row is built as {**player_info, **auction_state, "
        f"**team_state}}, so these reach the model twice: {sorted(overlap)}",
    )

    # Anything derived from the outcome must not be in a feature block.
    target_like = {
        "lower", "upper", "winner", "observation_type",
        "last_bid", "next_bid", "previous_bid",
        "auctionPrice", "basePrice", "auctionStatus",
    }

    for name, cols in (
        ("player", player_cols),
        ("team_state", team_cols),
        ("auction_state", auction_cols),
    ):
        leaked = cols & target_like
        report.add(
            "all",
            "BLOCKER",
            f"outcome-derived column inside {name} block",
            len(leaked),
            sorted(leaked),
        )

    ##################################################################
    # basePrice is now IN the player block, deliberately, as
    # ctx_basePrice -- and the rename means the BLOCKER check above
    # does not see it. That is the intended behaviour (a base price is
    # announced before the auction opens, so it is not hindsight), but
    # it should be visible in the audit rather than absent from it,
    # because of what it does to the labels: a 'left' row's interval
    # is literally (0.01, basePrice), so on those rows the upper bound
    # of the target is readable straight off an input column.
    ##################################################################

    promoted = sorted(c for c in player_cols if c.startswith("ctx_"))
    report.add(
        "all",
        "warn" if "ctx_basePrice" in player_cols else "info",
        "pre-auction context promoted into the player block",
        len(promoted),
        ", ".join(promoted)
        + (
            " -- ctx_basePrice bounds the label on every 'left' row"
            if "ctx_basePrice" in player_cols
            else ""
        ),
    )

    # A feature that is constant across the whole training set carries
    # no signal and, before scaling was added, could still dominate a
    # layer through sheer magnitude.
    constant = [
        c
        for c in sorted(player_cols)
        if training_df[c].nunique(dropna=False) <= 1
    ]
    report.add(
        "all",
        "warn",
        "constant player features",
        len(constant),
        ", ".join(constant[:10]),
    )

    # Rows whose entire player-feature block is missing: an unresolved
    # playerId that the left-merge never matched.
    #
    # Career columns only. ctx_* come off the auction row rather than
    # the ball data and are populated for unresolved players too, so
    # including them makes this count structurally zero.
    career_cols = sorted(c for c in player_cols if not c.startswith("ctx_"))
    all_missing = int(training_df[career_cols].isna().all(axis=1).sum())
    report.add(
        "all",
        "warn",
        "rows with a fully-missing player feature block",
        all_missing,
        f"{all_missing / max(len(training_df), 1):.1%} of rows -- "
        "unresolved playerIds; check whether they skew expensive",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", required=True, help="template with {year}")
    parser.add_argument("--bids", required=True, help="template with {year}")
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=sorted(AUCTION_DATES),
    )
    args = parser.parse_args()

    report = Report()

    for year in args.years:
        try:
            audit_year(
                year,
                args.players.format(year=year),
                args.bids.format(year=year),
                report,
            )
        except Exception as exc:  # noqa: BLE001
            report.add(year, "BLOCKER", "audit raised", 1, f"{type(exc).__name__}: {exc}")

    frame = report.frame()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)

    print("\n=== findings (count > 0 means the check fired) ===\n")
    fired = frame[frame["count"] > 0]
    print(fired.to_string(index=False) if len(fired) else "nothing fired")

    print(f"\n{report.blockers} blocker(s)")
    return 1 if report.blockers else 0


if __name__ == "__main__":
    sys.exit(main())