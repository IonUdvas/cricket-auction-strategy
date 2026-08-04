"""
The engine emits archetype-resolved state only when it is given the tag
table -- and for the whole history of this pipeline nothing gave it one.

archetypes.py was already well tested in isolation (test_archetypes.py).
What was never tested was the join: that build_training_samples actually
hands the table down, and that the state blocks widen when it does. A unit
test of a module nothing calls passes forever.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_creation_2 import archetypes as A
from input_creation_2.auction_replay_engine import AuctionReplayEngine

TEAMS = ["MI", "CSK", "RCB"]


def _archetype_table():
    cols = list(A._DIRECT.values()) + ["pace", "RA", "LA"]
    rows = [
        (1, {"opener": True, "top_order": True}),
        (2, {"pace": True, "RA": True}),
        (3, {"pace": True, "LA": True}),
        (4, {}),                       # untagged, and goes unsold
        (5, {"opener": True, "top_order": True}),   # two tags, one body
    ]
    out = []
    for pid, tags in rows:
        r = {c: False for c in cols}
        r.update(tags)
        r["player_id"] = pid
        out.append(r)
    return pd.DataFrame(out)


def _auction():
    player_df = pd.DataFrame([
        {"playerId": pid, "playerName": f"P{pid}", "basePrice": 20.0,
         "auctionPrice": price, "auctionStatus": status,
         "playsForTeam": team, "role": role, "country": "India",
         "countryId": 1, "isPlayerOverseas": False, "cappedStatus": True}
        for pid, price, status, team, role in [
            (1, 100.0, "SOLD", "MI", "BATTER"),
            (2, 80.0, "SOLD", "CSK", "BOWLER"),
            (3, 60.0, "SOLD", "MI", "BOWLER"),
            (4, None, "UNSOLD", None, "BOWLER"),
            (5, 20.0, "SOLD", "RCB", "BATTER"),
        ]
    ])
    bids = []
    n = 0
    for pid, ladder in [(1, [("CSK", 40.0), ("MI", 100.0)]),
                        (2, [("MI", 50.0), ("CSK", 80.0)]),
                        (3, [("RCB", 30.0), ("MI", 60.0)]),
                        (5, [("RCB", 20.0)])]:
        for team, amt in ladder:
            n += 1
            bids.append({"BidNumber": n, "playerId": pid, "Team": team,
                         "BidAmount": amt,
                         "basePrice": 20.0,
                         "auctionPrice": player_df.loc[
                             player_df.playerId == pid, "auctionPrice"].iloc[0],
                         "auctionStatus": "SOLD",
                         "playsForTeam": player_df.loc[
                             player_df.playerId == pid, "playsForTeam"].iloc[0]})
    return player_df, pd.DataFrame(bids)


def _replay(archetype_df):
    player_df, bid_df = _auction()
    engine = AuctionReplayEngine(
        bid_df=bid_df, player_df=player_df,
        auction_max_purse=8000, archetype_df=archetype_df,
    )
    return engine.replay()


@pytest.fixture(scope="module")
def without():
    return _replay(None)


@pytest.fixture(scope="module")
def with_tags():
    return _replay(_archetype_table())


def test_legacy_path_emits_no_archetype_state(without):
    """The silent fallback: no error, no archetype columns, just coarse roles."""
    team_cols = set(without["team_state"].columns)
    assert "batters_bought" in team_cols
    assert not [c for c in team_cols if c.startswith("opener")]
    assert not [c for c in team_cols if "scarcity" in c]


def test_archetype_state_appears_only_when_the_table_is_passed(
        without, with_tags):
    gained = set(with_tags["team_state"].columns) - set(
        without["team_state"].columns)
    assert gained, "passing archetype_df changed nothing -- the wiring is dead"
    # Every squad-construction archetype must get a bought counter.
    for a in A.ARCHETYPES:
        assert f"{a}_bought" in gained


def test_legacy_counters_survive_alongside(with_tags):
    """Kept deliberately, so an ablation is a column selection not a rebuild."""
    cols = set(with_tags["team_state"].columns)
    assert {"batters_bought", "bowlers_bought", "allrounders_bought"} <= cols


def test_auction_state_gains_supply_side(without, with_tags):
    gained = set(with_tags["auction_state"].columns) - set(
        without["auction_state"].columns)
    assert gained, "auction state did not widen; A^(r) has no archetype supply"


def test_counts_are_coverage_not_bodies(with_tags):
    """
    Player 1 is opener AND top_order, so MI's archetype counters sum to more
    than the players it bought. A partition assumption would break here.
    """
    team_state = with_tags["team_state"]

    # Snapshots are taken BEFORE each lot and the auction runs in reversed
    # file order (5,4,3,2,1), so RCB's purchase of player 5 -- the only
    # multi-tag body bought early enough to show up -- is visible from the
    # playerId 1 snapshot onward. Picking a row by position rather than by
    # this reasoning is how this test got it wrong the first time.
    rcb = team_state[(team_state["team"] == "RCB")
                     & (team_state["playerId"] == 1)].iloc[0]
    bought = [c for c in team_state.columns
              if c.endswith("_bought") and c[:-len("_bought")] in A.ARCHETYPES]
    total_tags = sum(int(rcb[c]) for c in bought)

    assert int(rcb["players_bought"]) == 1
    assert total_tags == 2, (
        f"one body carrying two tags counted as {total_tags}; archetype "
        f"counts have collapsed to a partition")


def test_training_rows_are_unchanged_by_archetypes(without, with_tags):
    """
    The intervals are a property of the bid ladder, not of the tag table.
    If wiring archetypes moved a single label, something is wrong.
    """
    a = without["training"][["playerId", "team", "lower", "upper",
                             "observation_type", "winner"]]
    b = with_tags["training"][["playerId", "team", "lower", "upper",
                               "observation_type", "winner"]]
    pd.testing.assert_frame_equal(
        a.sort_values(["playerId", "team"]).reset_index(drop=True),
        b.sort_values(["playerId", "team"]).reset_index(drop=True),
    )
