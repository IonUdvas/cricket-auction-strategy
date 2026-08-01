"""
One-call data checks, meant to be run from a Kaggle notebook after
cloning the repo.

    !git clone -q https://github.com/<you>/cricket-auction-strategy.git
    %cd cricket-auction-strategy

    from src.checks import run_checks

    chk = run_checks(player_template, bid_template)

Everything prints as it goes. The returned object holds the frames, and
carries the built feature context so follow-up questions are cheap:

    chk.player("Cameron Green")      # career the pipeline attached
    chk.trajectory("Rishabh Pant")   # totals at every auction date
    chk.ladder("Rishabh Pant", 2025) # the interval every team got
    chk.triage                       # unresolved playerIds, by cost

The ball data and the resolution cache come from the cloned repo by
default, so neither needs a path.
"""

import os

import numpy as np
import pandas as pd

from input_creation_2.auction_dataset_utils import PlayerFeatureContext
from input_creation_2.auction_replay_engine import AuctionReplayEngine
from input_creation_2.player_features.identity_triage import (
    classify_no_t20_record,
    triage_identity,
)
from src.training import (
    AUCTION_DATES,
    AUCTION_MAX_PURSES,
    DEFAULT_BBB_DIR,
    DEFAULT_RESOLUTION,
)


# Names worth checking by default: one unmistakable career, the
# name-variant cases the resolver exists for, the namesake traps, and
# whoever the model most recently got badly wrong.
DEFAULT_SPOT_CHECKS = [
    "Virat Kohli",
    "Rohit Sharma",
    "Lokesh Rahul",
    "Wanindu Hasaranga",
    "Varun Chakravarthy",
    "Hardik Pandya",
    "Krunal Pandya",
    "Rinku Singh",
    "Cameron Green",
    "Rishabh Pant",
]


def _fmt(x, nd=1):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    return f"{x:,.{nd}f}"


class CheckResult:
    """Holds the built context and every frame the checks produced."""

    def __init__(self, context, rosters, bid_template):
        self.context = context
        self.rosters = rosters
        self.bid_template = bid_template
        self.triage = None
        self.replay = None
        self.spot = None

    # ------------------------------------------------------------------
    # Interactive follow-ups
    # ------------------------------------------------------------------

    def _find(self, name):
        """(playerId, every roster appearance) for a name fragment."""
        hits = []
        for year, df in self.rosters.items():
            m = df[df["playerName"].str.contains(name, case=False, na=False)]
            for r in m.itertuples():
                hits.append(
                    {
                        "year": year,
                        "playerId": r.playerId,
                        "playerName": r.playerName,
                        "team": r.playsForTeam,
                        "basePrice": getattr(r, "basePrice", None),
                        "auctionPrice": getattr(r, "auctionPrice", None),
                        "status": getattr(r, "auctionStatus", None),
                    }
                )
        return pd.DataFrame(hits)

    def player(self, name, year=None):
        """
        The career the pipeline actually attached to this cricketer.

        This is the check that matters. `identity: 715/808 resolved`
        says nothing about whether the 715 resolved to the RIGHT
        people, and the only cheap test is looking at a player you know
        and asking whether the career is his.
        """
        hits = self._find(name)

        if hits.empty:
            print(f"no roster row matching {name!r}")
            return None

        print(hits.to_string(index=False))

        as_of = AUCTION_DATES[year or max(hits["year"])]
        rows = []

        for pid_ in sorted(hits["playerId"].unique()):
            person = self.context.person_by_player.get(pid_)

            if person is None:
                print(f"\n  playerId {pid_}: UNRESOLVED -> empty career")
                continue

            s = self.context.aggregator.get_player_stats(person, as_of)
            bat, bowl = s["batting"], s["bowling"]

            rows.append(
                {
                    "playerId": pid_,
                    "person_id": person,
                    "as_of": as_of,
                    "matches": s["experience"]["matches"],
                    "runs": bat["raw"]["runs"],
                    "balls": bat["raw"]["balls"],
                    "sixes": bat["raw"]["sixes"],
                    "bat_avg": bat["metrics"]["average"],
                    "bat_sr": bat["metrics"]["strike_rate"],
                    "bowl_balls": bowl["raw"]["balls"],
                    "wickets": bowl["raw"]["wickets"],
                    "econ": bowl["metrics"]["economy"],
                }
            )

        if len(hits["playerId"].unique()) > 1:
            print(
                f"  NOTE: {hits['playerId'].nunique()} distinct playerIds -- "
                f"the auction source itself treats these as different people"
            )

        frame = pd.DataFrame(rows)
        if len(frame):
            print()
            print(frame.round(2).to_string(index=False))
        return frame

    def trajectory(self, name):
        """
        Career totals at every auction date.

        Cumulative totals can only rise. A fall means the as-of filter
        is wrong for at least one of those dates, which is the shape an
        as-of leak takes here.
        """
        hits = self._find(name)
        if hits.empty:
            print(f"no roster row matching {name!r}")
            return None

        pid_ = hits["playerId"].iloc[0]
        person = self.context.person_by_player.get(pid_)

        if person is None:
            print(f"{name}: unresolved, no career to trace")
            return None

        rows = []
        for year, date in sorted(AUCTION_DATES.items()):
            s = self.context.aggregator.get_player_stats(person, date)
            rows.append(
                {
                    "auction": year,
                    "as_of": date,
                    "matches": s["experience"]["matches"],
                    "runs": s["batting"]["raw"]["runs"],
                    "balls": s["batting"]["raw"]["balls"],
                    "wickets": s["bowling"]["raw"]["wickets"],
                    "bat_sr": s["batting"]["metrics"]["strike_rate"],
                }
            )

        frame = pd.DataFrame(rows)

        for col in ("matches", "runs", "balls", "wickets"):
            if (frame[col].diff().dropna() < 0).any():
                print(f"  AS-OF BUG: {col} decreases between auctions")

        print(frame.round(1).to_string(index=False))
        return frame

    def ladder(self, name, year):
        """
        The (lower, upper) interval every team ended up with for this
        player, straight out of the replay -- i.e. the labels the model
        is actually trained on.
        """
        engine = AuctionReplayEngine(
            bid_df=pd.read_csv(self.bid_template.format(year=year)),
            player_df=self.rosters[year],
            auction_max_purse=AUCTION_MAX_PURSES[year],
        )
        training = engine.replay()["training"]

        rows = training[
            training["playerName"].str.contains(name, case=False, na=False)
        ]

        if rows.empty:
            print(f"{name!r} produced no training rows in {year} "
                  f"(retained, traded or drafted players produce none)")
            return None

        cols = ["playerName", "team", "lower", "upper", "winner",
                "observation_type", "basePrice", "auctionPrice"]
        print(rows[[c for c in cols if c in rows.columns]].to_string(index=False))
        return rows


def run_checks(
    player_template,
    bid_template,
    bbb_dir=None,
    resolution=None,
    years=None,
    spot_checks=None,
    archetype_df=None,
):
    """
    Every data check in one call.

    player_template / bid_template : paths containing "{year}"
    bbb_dir     : defaults to the cloned repo's own data/bbb
    resolution  : defaults to the repo's data/identity/cricinfo_resolution.csv
    archetype_df: optional; only used to report role coverage
    """

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    years = list(years or AUCTION_DATES)

    if bbb_dir is None:
        bbb_dir = DEFAULT_BBB_DIR
    if resolution is None and os.path.exists(DEFAULT_RESOLUTION):
        resolution = DEFAULT_RESOLUTION

    # A path to a single .parquet is the old interface and the most
    # common thing to get wrong when moving this to Kaggle; say so
    # rather than letting PlayerFeatureContext raise from three frames
    # down.
    if str(bbb_dir).endswith(".parquet") or os.path.isfile(bbb_dir):
        raise NotADirectoryError(
            f"bbb_dir={bbb_dir!r} is a single parquet file. This pipeline "
            f"needs the DIRECTORY written by data.build_bbb, holding "
            f"deliveries/fielding/people parquet. The cloned repo ships "
            f"one at data/bbb -- leave bbb_dir=None to use it."
        )

    print("=" * 72)
    print("1. ROSTERS")
    print("=" * 72)

    rosters = {}
    for year in years:
        path = player_template.format(year=year)
        if not os.path.exists(path):
            print(f"  {year}: MISSING {path}")
            continue
        rosters[year] = pd.read_csv(path)

    if not rosters:
        raise FileNotFoundError(
            "no roster files found -- check player_template"
        )

    shape = pd.DataFrame(
        [
            {
                "year": y,
                "players": len(d),
                "ids": d["playerId"].nunique(),
                **d["auctionStatus"].str.upper().value_counts().to_dict(),
            }
            for y, d in rosters.items()
        ]
    ).fillna(0)

    for c in shape.columns:
        if c != "year":
            shape[c] = shape[c].astype(int)

    print(shape.to_string(index=False))

    dupes = {
        y: int(d["playerId"].duplicated().sum())
        for y, d in rosters.items()
        if d["playerId"].duplicated().any()
    }
    if dupes:
        print(f"\n  DUPLICATE playerIds within a roster: {dupes}")
        print("  each duplicate fans out across every team in training")

    print("\n" + "=" * 72)
    print("2. IDENTITY")
    print("=" * 72)

    context = PlayerFeatureContext(bbb_dir, resolution=resolution)

    result = CheckResult(context, rosters, bid_template)

    context.register_rosters(rosters)

    triage = classify_no_t20_record(
        triage_identity(context, rosters), context
    )
    result.triage = triage

    if len(triage):
        # The number that decides whether any of this is worth fixing.
        print(
            f"\n  most expensive unresolved player ever sold for "
            f"{_fmt(triage['max_price'].max(), 0)} lakh; "
            f"{int((triage['max_price'] > 100).sum())} of {len(triage)} "
            f"ever cleared 1 crore"
        )

    print("\n" + "=" * 72)
    print("3. REPLAY, PER YEAR")
    print("=" * 72)

    replay_rows = []

    for year in sorted(rosters):
        engine = AuctionReplayEngine(
            bid_df=pd.read_csv(bid_template.format(year=year)),
            player_df=rosters[year],
            auction_max_purse=AUCTION_MAX_PURSES[year],
        )
        training = engine.replay()["training"]
        report = engine.quality_report()

        mix = training["observation_type"].value_counts()
        purses = [s["remaining_purse"] for s in engine.team_state.values()]

        replay_rows.append(
            {
                "year": year,
                "rows": len(training),
                "teams": training["team"].nunique(),
                "left": int(mix.get("left", 0)),
                "interval": int(mix.get("interval", 0)),
                "right": int(mix.get("right", 0)),
                "unknown": int(mix.get("unknown", 0)),
                "order_from": report["auction_order_column"]
                or report["auction_order_method"],
                "bid_order": report["bid_order_source"],
                "rtm_recovered": report.get("matched_at_top", 0),
                "dropped": report["dropped_bad_interval"],
                "backfilled": report["next_bid_backfilled"],
                "min_purse": round(min(purses)),
                "violations": report["final_state_violations"],
            }
        )

    replay = pd.DataFrame(replay_rows)
    result.replay = replay
    print(replay.to_string(index=False))

    problems = []
    if (replay["dropped"] > 0).any():
        problems.append("dropped>0: intervals the replay could not build")
    if (replay["unknown"] > 0).any():
        problems.append("unknown>0: rows excluded from training")
    if (replay["min_purse"] < 0).any():
        problems.append(
            "min_purse<0: a team overspent -- suspect missing "
            "traded/drafted handling, a wrong purse constant, or "
            "auction order"
        )
    if (replay["violations"] > 0).any():
        problems.append("violations>0: squad size or overseas limit exceeded")
    if (replay["bid_order"] != "recorded").any():
        problems.append("bid_order != recorded: BidNumber missing, ladder inferred")

    print()
    if problems:
        for p in problems:
            print(f"  PROBLEM  {p}")
    else:
        print("  replay clean across every year")

    print("\n" + "=" * 72)
    print("4. FEATURE SPOT CHECKS")
    print("=" * 72)

    spot_rows = []

    for name in (spot_checks or DEFAULT_SPOT_CHECKS):
        hits = result._find(name)
        if hits.empty:
            spot_rows.append({"name": name, "status": "not in any roster"})
            continue

        year = int(max(hits["year"]))
        pid_ = hits["playerId"].iloc[0]
        person = context.person_by_player.get(pid_)

        if person is None:
            spot_rows.append(
                {"name": name, "playerId": pid_, "status": "UNRESOLVED"}
            )
            continue

        s = context.aggregator.get_player_stats(person, AUCTION_DATES[year])

        spot_rows.append(
            {
                "name": name,
                "playerId": pid_,
                "status": "ok",
                "as_of": year,
                "matches": s["experience"]["matches"],
                "runs": s["batting"]["raw"]["runs"],
                "bat_sr": round(s["batting"]["metrics"]["strike_rate"] or 0, 1),
                "wickets": s["bowling"]["raw"]["wickets"],
                "econ": round(s["bowling"]["metrics"]["economy"] or 0, 2),
            }
        )

    spot = pd.DataFrame(spot_rows).fillna("")
    result.spot = spot
    print(spot.to_string(index=False))
    print(
        "\n  read this as a human: do these careers belong to these "
        "cricketers? A zero career on a famous name is an identity "
        "failure, not a debutant."
    )

    print("\n" + "=" * 72)
    print("5. AS-OF MONOTONICITY")
    print("=" * 72)

    # Cumulative career totals can only rise between auctions. Checked
    # over everyone, not a sample: this is the only test that catches
    # as-of leakage without a second source of truth.
    cum = ["bat_runs", "bat_balls", "bowl_balls", "bowl_wickets", "exp_matches"]
    per_year = {}

    for year in sorted(rosters):
        feats = context.features_for(rosters[year], AUCTION_DATES[year])
        cols = [c for c in cum if c in feats.columns]
        per_year[year] = feats.set_index("playerId")[cols]

    years_sorted = sorted(per_year)
    violations = []

    for a, b in zip(years_sorted, years_sorted[1:]):
        fa, fb = per_year[a], per_year[b]
        shared = fa.index.intersection(fb.index)
        if not len(shared):
            continue
        back = fb.loc[shared] < fa.loc[shared] - 1e-6
        for col in back.columns:
            n = int(back[col].sum())
            if n:
                violations.append(
                    {"from": a, "to": b, "column": col, "players": n}
                )

    if violations:
        print("  AS-OF LEAK -- cumulative totals fell between auctions:")
        print(pd.DataFrame(violations).to_string(index=False))
    else:
        print(f"  clean across {len(years_sorted)} auctions")

    if archetype_df is not None:
        print("\n" + "=" * 72)
        print("6. ROLE / ARCHETYPE COVERAGE")
        print("=" * 72)

        key = "player_id" if "player_id" in archetype_df.columns else "playerId"
        tagged = set(pd.to_numeric(archetype_df[key], errors="coerce").dropna())

        rows = []
        for year in sorted(rosters):
            ids = set(rosters[year]["playerId"])
            hit = ids & tagged
            rows.append(
                {
                    "year": year,
                    "players": len(ids),
                    "tagged": len(hit),
                    "coverage": f"{len(hit) / max(len(ids), 1):.1%}",
                }
            )

        print(pd.DataFrame(rows).to_string(index=False))
        print(
            "  untagged players get an all-zero role vector, which reads "
            "as 'no role' rather than 'role unknown'"
        )

    print("\n" + "=" * 72)
    print("chk.player(name) / chk.trajectory(name) / chk.ladder(name, year) "
          "/ chk.triage")
    print("=" * 72)

    return result