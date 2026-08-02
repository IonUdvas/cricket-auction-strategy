"""
Tests for input_creation_2/archetypes.py.

The properties worth pinning down are the three that a natural implementation
gets wrong: archetype counts are multi-label and must NOT sum to the squad
size, untagged players must survive into a bucket rather than vanish, and
demand must exclude teams that cannot actually bid.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_creation_2 import archetypes as A


def _table(rows):
    """Build an archetype frame from partial dicts; unset tags are False."""
    cols = (list(A._DIRECT.values()) + ["pace", "RA", "LA"])
    out = []
    for pid, tags in rows:
        r = {c: False for c in cols}
        r.update(tags)
        r["player_id"] = pid
        out.append(r)
    return pd.DataFrame(out)


@pytest.fixture
def tags():
    return A.build_archetype_tags(_table([
        (1, {"opener": True, "top_order": True}),          # two batting tags
        (2, {"pace": True, "RA": True}),                   # right-arm quick
        (3, {"pace": True, "LA": True}),                   # left-arm quick
        (4, {"finger_spin": True, "middle_order": True,
             "bowling_allrounder": True}),
        (5, {}),                                           # untagged
        (6, {"wrist_spin": True}),
        (7, {"finisher": True, "wicketkeeper": True}),
    ]))


# -- vocabulary -------------------------------------------------------------

def test_pace_is_split_by_arm(tags):
    assert A.tags_for(tags, 2) == ("right_arm_pace",)
    assert A.tags_for(tags, 3) == ("left_arm_pace",)


def test_untagged_player_gets_a_bucket(tags):
    assert A.tags_for(tags, 5) == ("untagged",)


def test_unknown_player_is_untagged_not_an_error(tags):
    assert A.tags_for(tags, 9999) == ("untagged",)


def test_multi_label_players_keep_every_tag(tags):
    assert set(A.tags_for(tags, 4)) == {
        "finger_spin", "middle_order", "bowling_allrounder"}


def test_paced_player_without_an_arm_is_rejected():
    with pytest.raises(ValueError, match="arm tag"):
        A.build_archetype_tags(_table([(1, {"pace": True})]))


def test_duplicate_player_id_is_rejected():
    df = _table([(1, {"opener": True}), (1, {"finisher": True})])
    with pytest.raises(ValueError, match="duplicated"):
        A.build_archetype_tags(df)


# -- team state -------------------------------------------------------------

def test_counts_are_coverage_not_bodies(tags):
    """One purchase can increment several archetypes. This is the point."""
    counts = A.empty_team_archetype_counts()
    A.apply_purchase(counts, A.tags_for(tags, 4))
    total = sum(v for k, v in counts.items() if k.endswith("_bought"))
    assert total == 3, "a three-archetype player must increment three counters"
    assert counts["finger_spin_bought"] == 1
    assert counts["middle_order_bought"] == 1


def test_needed_is_floored_at_zero(tags):
    counts = A.empty_team_archetype_counts()
    for _ in range(A.ARCHETYPE_TARGETS["opener"] + 2):
        A.apply_purchase(counts, ("opener",))
    feats = A.team_archetype_features(counts)
    assert feats["opener_needed"] == 0
    # bought keeps counting past the target, so an over-full cupboard is
    # distinguishable from a just-filled one
    assert feats["opener_bought"] == A.ARCHETYPE_TARGETS["opener"] + 2


# -- pool and demand --------------------------------------------------------

def test_pool_counts_every_tag_of_every_remaining_player(tags):
    pool = A.pool_archetype_counts([1, 4, 5], tags)
    assert pool["opener"] == 1 and pool["top_order"] == 1
    assert pool["finger_spin"] == 1
    assert pool["untagged"] == 1


def test_demand_excludes_teams_with_no_slots():
    full = {"remaining_slots": 0, "remaining_purse": 500.0}
    open_ = {"remaining_slots": 4, "remaining_purse": 500.0}
    d = A.archetype_demand({"A": full, "B": open_})
    assert d["opener"] == 1, "a team with no slots left is not demand"


def test_demand_excludes_teams_that_cannot_afford_the_floor():
    broke = {"remaining_slots": 5, "remaining_purse": 0.1}
    rich = {"remaining_slots": 5, "remaining_purse": 500.0}
    d = A.archetype_demand({"A": broke, "B": rich}, min_purse=20.0)
    assert d["opener"] == 1


def test_demand_falls_as_a_team_fills_the_archetype():
    state = {"remaining_slots": 5, "remaining_purse": 500.0,
             **A.empty_team_archetype_counts()}
    before = A.archetype_demand({"A": state})["wrist_spin"]
    for _ in range(A.ARCHETYPE_TARGETS["wrist_spin"]):
        A.apply_purchase(state, ("wrist_spin",))
    after = A.archetype_demand({"A": state})["wrist_spin"]
    assert before == 1 and after == 0


def test_scarcity_is_finite_when_supply_is_exhausted():
    s = A.scarcity({"opener": 0}, {"opener": 6})
    assert np.isfinite(s["opener"]) and s["opener"] == 6.0


def test_scarcity_rises_as_supply_shrinks():
    demand = {a: 5 for a in A.ARCHETYPES}
    loose = A.scarcity({a: 40 for a in A.ARCHETYPES}, demand)
    tight = A.scarcity({a: 2 for a in A.ARCHETYPES}, demand)
    assert tight["opener"] > loose["opener"]


# -- focus block ------------------------------------------------------------

def test_focus_is_fixed_width_regardless_of_tag_count(tags):
    pool = A.pool_archetype_counts([1, 2, 3, 4, 5, 6, 7], tags)
    demand = {a: 3 for a in A.ARCHETYPES}
    counts = A.empty_team_archetype_counts()
    one = A.focus_features(A.tags_for(tags, 2), counts, pool, demand,
                           [1, 2, 3], tags)
    three = A.focus_features(A.tags_for(tags, 4), counts, pool, demand,
                             [1, 2, 3], tags)
    assert set(one) == set(three)
    assert one["focus_n_archetypes"] == 1
    assert three["focus_n_archetypes"] == 3


def test_substitutes_counts_players_sharing_any_archetype(tags):
    pool = A.pool_archetype_counts([1, 2, 3, 4, 6], tags)
    demand = {a: 0 for a in A.ARCHETYPES}
    # player 4 is finger_spin + middle_order + bowling_allrounder; among
    # {1,2,3,6} nobody shares a tag with him, so only he substitutes.
    f = A.focus_features(A.tags_for(tags, 4), A.empty_team_archetype_counts(),
                         pool, demand, [1, 2, 3, 4, 6], tags)
    assert f["focus_substitutes_remaining"] == 1

    # player 1 is opener + top_order; nobody else in the pool is either.
    f1 = A.focus_features(A.tags_for(tags, 1), A.empty_team_archetype_counts(),
                          pool, demand, [1, 2, 3, 4, 6], tags)
    assert f1["focus_substitutes_remaining"] == 1


def test_team_wants_any_goes_false_once_every_tag_is_filled(tags):
    counts = A.empty_team_archetype_counts()
    pool, demand = {}, {}
    own = A.tags_for(tags, 2)  # right_arm_pace only
    f = A.focus_features(own, counts, pool, demand, [], tags)
    assert f["focus_team_wants_any"] == 1
    for _ in range(A.ARCHETYPE_TARGETS["right_arm_pace"]):
        A.apply_purchase(counts, own)
    f = A.focus_features(own, counts, pool, demand, [], tags)
    assert f["focus_team_wants_any"] == 0


def test_untagged_player_never_reads_as_wanted(tags):
    f = A.focus_features(("untagged",), A.empty_team_archetype_counts(),
                         {}, {}, [], tags)
    assert f["focus_team_wants_any"] == 0


# -- real table -------------------------------------------------------------

def test_real_archetype_table_builds():
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "auction", "player_archetypes.csv")
    if not os.path.exists(path):
        pytest.skip("player_archetypes.csv not present")
    tags = A.build_archetype_tags(pd.read_csv(path))
    assert len(tags) > 700
    # Every player lands in at least one bucket.
    assert tags.any(axis=1).all()
    # right_arm_pace + left_arm_pace must exactly partition `pace`.
    raw = pd.read_csv(path)
    assert (tags["right_arm_pace"].sum() + tags["left_arm_pace"].sum()
            == int(raw["pace"].fillna(False).sum()))
