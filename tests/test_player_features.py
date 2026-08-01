"""
Hand-verified tests for the ball-by-ball build and the statistics aggregator.

Two synthetic Cricsheet matches, written to the real JSON schema, exercise every
rule that the old pipeline got wrong:

  * a wide is not a ball faced and not a legal ball, but is charged to the bowler
  * a no-ball IS faced, is NOT legal, and IS charged to the bowler
  * byes/leg-byes are never charged to the bowler
  * a run out is a dismissal for the batter but no wicket for the bowler
  * a batter run out at the NON-striker's end is still dismissed
  * `retired hurt` is not an out
  * a 4 flagged `non_boundary` is four runs but not a boundary
  * two people who share a name stay separate (different person ids)
  * one person with two name variants stays one player (same person id)
  * matches on or after the as-of date are excluded
"""

import json
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.build_bbb import build  # noqa: E402
from input_creation_2.player_features.player_features import (  # noqa: E402
    PlayerFeatureBuilder,
    PlayerStatsAggregator,
)

AB = "p0000001"   # appears as "AB Player" then "A Player" -- one person
CD1 = "p0000002"  # "CD Player" in match 1
CD2 = "p0000099"  # a DIFFERENT "CD Player" in match 2
EF = "p0000003"
GH = "p0000004"
IJ = "p0000005"
KL = "p0000006"


def _d(batter, non_striker, bowler, batter_runs, extras=None, wickets=None):
    extras = extras or {}
    total = batter_runs + sum(extras.values())
    out = {
        "actual_delivery": "0.1",
        "batter": batter,
        "non_striker": non_striker,
        "bowler": bowler,
        "runs": {"batter": batter_runs, "extras": sum(extras.values()), "total": total},
    }
    if extras:
        out["extras"] = extras
    if wickets:
        out["wickets"] = wickets
    return out


def match_one():
    reg = {
        "AB Player": AB, "CD Player": CD1, "EF Bowler": EF,
        "GH Field": GH, "IJ Player": IJ,
    }
    return {
        "meta": {"data_version": "1.2.0", "created": "2020-01-01", "revision": 1},
        "info": {
            "balls_per_over": 6, "overs": 20, "match_type": "T20",
            "gender": "male", "team_type": "club", "season": "2018",
            "dates": ["2018-01-01"], "teams": ["Alpha", "Beta"],
            "venue": "Ground A", "city": "Cityville",
            "event": {"name": "Test League"},
            "toss": {"winner": "Alpha", "decision": "bat"},
            "outcome": {"winner": "Alpha"},
            "players": {"Alpha": ["AB Player", "CD Player", "IJ Player"],
                        "Beta": ["EF Bowler", "GH Field"]},
            "registry": {"people": reg},
        },
        "innings": [{
            "team": "Alpha",
            "overs": [{
                "over": 0,
                "deliveries": [
                    # 1. a genuine four
                    _d("AB Player", "CD Player", "EF Bowler", 4),
                    # 2. a wide: not faced, not legal, charged to the bowler
                    _d("AB Player", "CD Player", "EF Bowler", 0, {"wides": 1}),
                    # 3. a no-ball: faced, not legal, charged to the bowler
                    _d("AB Player", "CD Player", "EF Bowler", 2, {"noballs": 1}),
                    # 4. leg-byes: faced, legal, NOT charged to the bowler
                    _d("AB Player", "CD Player", "EF Bowler", 0, {"legbyes": 2}),
                    # 5. non-striker run out -- CD is dismissed, EF gets nothing
                    _d("AB Player", "CD Player", "EF Bowler", 0, wickets=[{
                        "kind": "run out", "player_out": "CD Player",
                        "fielders": [{"name": "GH Field"}],
                    }]),
                    # 6. striker caught -- AB dismissed, EF credited
                    _d("AB Player", "IJ Player", "EF Bowler", 0, wickets=[{
                        "kind": "caught", "player_out": "AB Player",
                        "fielders": [{"name": "GH Field"}],
                    }]),
                ],
            }],
        }],
    }


def match_two():
    # "A Player" is the SAME person as match one's "AB Player" (same id).
    # "CD Player" here is a DIFFERENT person from match one's "CD Player".
    reg = {
        "A Player": AB, "CD Player": CD2, "EF Bowler": EF,
        "IJ Player": IJ, "KL Bowler": KL,
    }
    return {
        "meta": {"data_version": "1.2.0", "created": "2020-01-01", "revision": 1},
        "info": {
            "balls_per_over": 6, "overs": 20, "match_type": "T20",
            "gender": "male", "team_type": "club", "season": "2018",
            "dates": ["2018-02-01"], "teams": ["Alpha", "Beta"],
            "venue": "Ground B",
            "event": {"name": "Test League"},
            "toss": {"winner": "Beta", "decision": "field"},
            "outcome": {"winner": "Beta"},
            "players": {"Alpha": ["A Player", "CD Player", "IJ Player"],
                        "Beta": ["EF Bowler", "KL Bowler"]},
            "registry": {"people": reg},
        },
        "innings": [{
            "team": "Alpha",
            "overs": [
                {"over": 0, "deliveries": [
                    _d("A Player", "CD Player", "KL Bowler", 6),
                    # four runs, but all run -- not a boundary
                    dict(_d("A Player", "CD Player", "KL Bowler", 4),
                         runs={"batter": 4, "extras": 0, "total": 4,
                               "non_boundary": True}),
                    # retired hurt: a wicket entry, but NOT an out
                    _d("A Player", "CD Player", "KL Bowler", 0, wickets=[{
                        "kind": "retired hurt", "player_out": "A Player"}]),
                ]},
                {"over": 15, "deliveries": [
                    _d("CD Player", "IJ Player", "EF Bowler", 1),
                ]},
            ],
        }],
    }


@pytest.fixture(scope="module")
def built():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "json")
    os.makedirs(src)
    for name, doc in (("match1", match_one()), ("match2", match_two())):
        with open(os.path.join(src, f"{name}.json"), "w") as fh:
            json.dump(doc, fh)
    return build([src], os.path.join(tmp, "out"), verbose=False)


@pytest.fixture(scope="module")
def agg(built):
    return PlayerStatsAggregator(built["deliveries"], fielding=built["fielding"])


# --- build-level facts -----------------------------------------------------

def test_delivery_count(built):
    assert len(built["deliveries"]) == 10


def test_wide_is_not_faced_and_not_legal(built):
    d = built["deliveries"]
    w = d[d["is_wide"]]
    assert len(w) == 1
    assert not w["ball_faced"].any()
    assert not w["legal_ball"].any()
    assert int(w["runs_conceded_bowler"].iloc[0]) == 1


def test_noball_is_faced_but_not_legal(built):
    d = built["deliveries"]
    nb = d[d["is_noball"]]
    assert len(nb) == 1
    assert nb["ball_faced"].all()
    assert not nb["legal_ball"].any()
    # 2 off the bat plus the 1-run no-ball, all charged to the bowler
    assert int(nb["runs_conceded_bowler"].iloc[0]) == 3


def test_legbyes_not_charged_to_bowler(built):
    d = built["deliveries"]
    lb = d[d["legbyes"] > 0]
    assert len(lb) == 1
    assert int(lb["runs_conceded_bowler"].iloc[0]) == 0
    assert lb["legal_ball"].all() and lb["ball_faced"].all()


def test_non_boundary_four_is_not_a_boundary(built):
    d = built["deliveries"]
    nb = d[d["non_boundary"]]
    assert len(nb) == 1
    assert int(nb["runs_batter"].iloc[0]) == 4
    assert not nb["is_four"].any()
    assert not nb["is_boundary"].any()


def test_bowler_credit(built):
    d = built["deliveries"]
    assert int(d["bowler_credited"].sum()) == 1          # only the catch
    kinds = set(d.loc[d["is_wicket"], "wicket_kind"])
    assert kinds == {"run out", "caught", "retired hurt"}


def test_retired_hurt_is_not_an_out(built):
    d = built["deliveries"]
    rh = d[d["wicket_kind"] == "retired hurt"]
    assert len(rh) == 1
    assert not rh["player_out_counts"].any()


def test_one_person_two_names(built):
    people = built["people"].set_index("person_id")
    assert people.loc[AB, "n_variants"] == 2
    assert set(people.loc[AB, "name_variants"].split("|")) == {"AB Player", "A Player"}


def test_two_people_one_name(built):
    people = built["people"].set_index("person_id")
    assert people.loc[CD1, "name_variants"] == "CD Player"
    assert people.loc[CD2, "name_variants"] == "CD Player"
    assert CD1 != CD2


# --- batting ---------------------------------------------------------------

def test_batting_after_match_one(agg):
    s = agg.get_player_stats(AB, "2018-01-15")
    raw, met = s["batting"]["raw"], s["batting"]["metrics"]
    assert raw["balls"] == 5          # 6 deliveries minus the wide
    assert raw["runs"] == 6           # 4 + 2
    assert raw["fours"] == 1
    assert raw["sixes"] == 0
    assert raw["dots"] == 3           # the leg-bye ball, the run out, the catch
    assert raw["outs"] == 1
    assert met["average"] == pytest.approx(6.0)
    assert met["strike_rate"] == pytest.approx(120.0)
    assert met["boundary_percentage"] == pytest.approx(0.2)
    assert met["dot_ball_percentage"] == pytest.approx(0.6)
    assert s["experience"]["matches"] == 1


def test_batting_after_both_matches(agg):
    s = agg.get_player_stats(AB, "2018-03-01")
    raw, met = s["batting"]["raw"], s["batting"]["metrics"]
    assert raw["balls"] == 8
    assert raw["runs"] == 16          # match 1: 6, match 2: 6 + 4 + 0
    assert raw["fours"] == 1          # the all-run 4 does not count
    assert raw["sixes"] == 1
    assert raw["outs"] == 1           # retired hurt is not an out
    assert met["average"] == pytest.approx(16.0)
    assert met["strike_rate"] == pytest.approx(200.0)
    assert met["boundary_percentage"] == pytest.approx(0.25)  # one 4, one 6
    assert s["experience"]["matches"] == 2
    assert s["experience"]["batting_innings"] == 2


def test_non_striker_run_out_counts_as_a_dismissal(agg):
    s = agg.get_player_stats(CD1, "2018-03-01")
    assert s["batting"]["raw"]["balls"] == 0     # never faced a ball
    assert s["batting"]["raw"]["outs"] == 1      # still dismissed
    assert s["batting"]["metrics"]["average"] == pytest.approx(0.0)
    # undefined, not zero
    assert s["batting"]["metrics"]["strike_rate"] is None


def test_namesakes_do_not_merge(agg):
    one = agg.get_player_stats(CD1, "2018-03-01")["batting"]["raw"]
    two = agg.get_player_stats(CD2, "2018-03-01")["batting"]["raw"]
    assert (one["runs"], one["balls"], one["outs"]) == (0, 0, 1)
    assert (two["runs"], two["balls"], two["outs"]) == (1, 1, 0)


def test_phase_split(agg):
    s = agg.get_player_stats(CD2, "2018-03-01")["batting"]["raw"]
    assert s["balls_death"] == 1 and s["runs_death"] == 1
    assert s["balls_powerplay"] == 0


# --- bowling ---------------------------------------------------------------

def test_bowling_after_match_one(agg):
    s = agg.get_player_stats(EF, "2018-01-15")
    raw, met = s["bowling"]["raw"], s["bowling"]["metrics"]
    assert raw["balls"] == 4          # 6 minus the wide and the no-ball
    assert raw["runs"] == 8           # 4 + 1(wide) + 3(no-ball incl. 2 off bat)
    assert raw["wickets"] == 1        # the run out is not the bowler's
    assert raw["wides"] == 1 and raw["noballs"] == 1
    assert raw["dots"] == 2           # the leg-bye ball is not a bowler's dot
    assert raw["maidens"] == 0
    assert met["economy"] == pytest.approx(12.0)
    assert met["average"] == pytest.approx(8.0)
    assert met["strike_rate"] == pytest.approx(4.0)


def test_bowling_undefined_metrics_are_none(agg):
    s = agg.get_player_stats(KL, "2018-03-01")
    raw, met = s["bowling"]["raw"], s["bowling"]["metrics"]
    assert raw["balls"] == 3 and raw["runs"] == 10 and raw["wickets"] == 0
    assert met["economy"] == pytest.approx(20.0)
    assert met["average"] is None      # no wickets -- undefined, not 0.0
    assert met["strike_rate"] is None


def test_bowling_phase_split(agg):
    s = agg.get_player_stats(EF, "2018-03-01")["bowling"]["raw"]
    assert s["balls_powerplay"] == 4 and s["runs_powerplay"] == 8
    assert s["balls_death"] == 1 and s["runs_death"] == 1
    assert s["balls"] == 5


# --- fielding --------------------------------------------------------------

def test_fielding(agg):
    f = agg.get_player_stats(GH, "2018-03-01")["fielding"]
    assert f["catches"] == 1
    assert f["run_out_involvements"] == 1
    assert f["stumpings"] == 0


# --- as-of semantics -------------------------------------------------------

def test_as_of_date_is_strict(agg):
    """A match played ON the as-of date must be excluded."""
    on_the_day = agg.get_player_stats(AB, "2018-01-01")["batting"]["raw"]
    day_after = agg.get_player_stats(AB, "2018-01-02")["batting"]["raw"]
    assert on_the_day["balls"] == 0
    assert day_after["balls"] == 5


def test_unknown_player_is_empty_not_an_error(agg):
    s = agg.get_player_stats("zzzzzzzz", "2018-03-01")
    assert s["has_history"] is False
    assert s["batting"]["raw"]["runs"] == 0
    assert s["batting"]["metrics"]["average"] is None


# --- feature table ---------------------------------------------------------

def test_feature_table_row_cardinality(agg):
    builder = PlayerFeatureBuilder(agg)
    players = pd.DataFrame({
        "player_id": [AB, CD1, CD2, EF, KL, "zzzzzzzz"],
        "playerName": ["A Player", "CD Player", "CD Player",
                       "EF Bowler", "KL Bowler", "Nobody"],
    })
    table = builder.build_feature_table(players, "2018-03-01")
    assert len(table) == 6
    assert table["player_id"].is_unique
    # the two namesakes are separate rows with different numbers
    ab = table.set_index("player_id").loc[AB]
    cd1 = table.set_index("player_id").loc[CD1]
    cd2 = table.set_index("player_id").loc[CD2]
    assert ab["bat_runs"] == 16
    assert cd1["bat_runs"] == 0 and cd2["bat_runs"] == 1


def test_missing_indicators(agg):
    builder = PlayerFeatureBuilder(agg)
    table = builder.build_feature_table(
        pd.DataFrame({"player_id": [KL, EF]}), "2018-03-01"
    ).set_index("player_id")
    # KL has no wickets: average is filled but flagged
    assert table.loc[KL, "bowl_average"] == 0.0
    assert table.loc[KL, "bowl_average_is_missing"] == 1.0
    # EF has wickets: real value, not flagged
    assert table.loc[EF, "bowl_average"] > 0
    assert table.loc[EF, "bowl_average_is_missing"] == 0.0


def test_no_nans_in_feature_table(agg):
    builder = PlayerFeatureBuilder(agg)
    table = builder.build_feature_table(
        pd.DataFrame({"player_id": [AB, CD1, CD2, EF, KL, "zzzzzzzz"]}),
        "2018-03-01",
    )
    numeric = table.drop(columns=["player_id"])
    assert not numeric.isna().to_numpy().any()


# --- identity bridge -------------------------------------------------------

def test_identity_resolver(built):
    from input_creation_2.player_features.identity import PlayerIdentityResolver

    people = built["people"]
    r = PlayerIdentityResolver(people)

    roster = pd.DataFrame({
        "playerId": [1, 2, 3, 4, 5],
        "playerName": ["AB Player",      # exact, via a non-canonical variant
                       "A Player",       # the other variant -> same person
                       "EF  Bowler",     # spacing noise -> normalised match
                       "CD Player",      # namesake -> ambiguous, not guessed
                       "Nobody At All"], # absent -> unresolved
    })
    out = r.resolve(roster)
    assert out.loc[0, "person_id"] == AB
    assert out.loc[1, "person_id"] == AB
    assert out.loc[2, "person_id"] == EF
    assert out.loc[2, "match_method"] == "normalised"
    assert pd.isna(out.loc[3, "person_id"])
    assert out.loc[3, "match_method"] == "ambiguous"
    assert out.loc[4, "match_method"] == "unresolved"
    assert len(out) == len(roster)          # nothing dropped
    assert "ambiguous" in r.report(out)


def test_identity_override_disambiguates(built):
    from input_creation_2.player_features.identity import PlayerIdentityResolver

    overrides = pd.DataFrame({
        "playerId": [4], "person_id": [CD2], "action": ["map"],
    })
    r = PlayerIdentityResolver(built["people"], overrides=overrides)
    out = r.resolve(pd.DataFrame({"playerId": [4], "playerName": ["CD Player"]}))
    assert out.loc[0, "person_id"] == CD2
    assert out.loc[0, "match_method"] == "override"
