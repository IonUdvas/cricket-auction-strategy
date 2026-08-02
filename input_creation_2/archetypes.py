"""
Archetype-resolved team and auction state.

What changed and why
--------------------
The replay engine used to carry three role counters per team --
batters_bought, bowlers_bought, allrounders_bought -- and three matching
remaining-pool counters.  Those are the wrong granularity for an IPL auction.
A team that has bought three batters has not thereby solved its batting: if
all three open, it still has nobody to finish, and it will bid on a finisher
like a team with no batters at all.  The same collapse hides the difference
between a left-arm quick and a fourth right-arm seamer, which is most of what
an IPL squad is actually assembled around.

So roles are replaced by the archetype tags already curated in
player_archetypes.csv, split the way squads are actually built:

    batting order   opener, top_order, middle_order, finisher
    bowling type    right_arm_pace, left_arm_pace, finger_spin, wrist_spin
    all-rounders    batting_allrounder, bowling_allrounder
    keeper          wicketkeeper

Three facts about that table drive the whole design of this module, and each
one breaks a natural-looking implementation:

1. **The tags are multi-label, not a partition.**  183 of 808 players carry
   two batting-order tags; a top-order batter who also finishes is genuinely
   both.  So a bought player increments *every* archetype he carries, and the
   archetype counts do not sum to the squad size.  That is correct: these
   count role *coverage*, not bodies.  Anything that treats them as a
   partition -- a softmax, a "primary archetype", a sum check against
   players_bought -- is wrong about the cricket.

2. **178 of 808 players (22%) carry no role tag at all.**  Uncapped domestic
   players mostly arrive untagged.  They are given an explicit `untagged`
   archetype rather than being dropped, because dropping them makes the
   remaining-pool counts disagree with players_remaining, and a silent
   disagreement between two counters that should reconcile is the kind of bug
   that is found six weeks later in a model that was quietly mispriced.

3. **`right_arm_pace` and `left_arm_pace` are derived, not stored.**  The
   table has `pace`, `RA` and `LA` separately.  Every paced player has
   exactly one arm set (205 RA, 56 LA, 0 neither), so the conjunction is
   safe -- but it is asserted rather than assumed.

Supply and demand
-----------------
`ARCHETYPE_TARGETS` says how many players of each archetype a squad wants.
It is a modelling assumption, not a fact, and it is the one number in here
worth sweeping: it sets `demand` and therefore every scarcity feature.

    supply(A)    players left in the pool carrying A
    demand(A)    teams that still want A (bought < target) and can still
                 act on that want (a slot free, and purse above the base
                 price of the cheapest remaining A)
    scarcity(A)  demand / supply, high when a tag is running out

The archetype-focused block narrows all of that to the player under the
hammer.  He carries a variable number of tags, so his own counts cannot be
emitted directly without a ragged feature vector; they are reduced to fixed
width by taking min/mean/max across his own tag set, plus counts of the
players who could substitute for him and the teams who might want him.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

BATTING_ARCHETYPES = ("opener", "top_order", "middle_order", "finisher")
BOWLING_ARCHETYPES = ("right_arm_pace", "left_arm_pace", "finger_spin",
                      "wrist_spin")
ALLROUNDER_ARCHETYPES = ("batting_allrounder", "bowling_allrounder")
OTHER_ARCHETYPES = ("wicketkeeper", "untagged")

ARCHETYPES = (BATTING_ARCHETYPES + BOWLING_ARCHETYPES
              + ALLROUNDER_ARCHETYPES + OTHER_ARCHETYPES)

# Columns read straight off player_archetypes.csv.
_DIRECT = {
    "opener": "opener",
    "top_order": "top_order",
    "middle_order": "middle_order",
    "finisher": "finisher",
    "finger_spin": "finger_spin",
    "wrist_spin": "wrist_spin",
    "batting_allrounder": "batting_allrounder",
    "bowling_allrounder": "bowling_allrounder",
    "wicketkeeper": "wicketkeeper",
}

# How many of each archetype a squad wants before it stops needing more.
# These are assumptions about squad construction, deliberately gathered in one
# place so they can be swept rather than rediscovered inline. An XI needs two
# openers and a top three, four or five bowling options, and the bench roughly
# doubles the specialist slots.
ARCHETYPE_TARGETS = {
    "opener": 3,
    "top_order": 3,
    "middle_order": 4,
    "finisher": 3,
    "right_arm_pace": 4,
    "left_arm_pace": 2,
    "finger_spin": 3,
    "wrist_spin": 2,
    "batting_allrounder": 2,
    "bowling_allrounder": 3,
    "wicketkeeper": 2,
    # A team never "needs" an untagged player; the bucket exists so the pool
    # counts reconcile, not so it can be targeted.
    "untagged": 0,
}


# ---------------------------------------------------------------------------
# Tag table
# ---------------------------------------------------------------------------

def build_archetype_tags(archetype_df, id_column="player_id"):
    """
    player_id -> boolean row over ARCHETYPES.

    Returns a DataFrame indexed by player id with one bool column per
    archetype.  A player absent from this table is not an error; the lookup
    helpers below treat him as `untagged`, which is what an uncapped debutant
    genuinely is.
    """
    a = archetype_df.copy()
    missing = {c for c in _DIRECT.values()} | {"pace", "RA", "LA"}
    absent = missing - set(a.columns)
    if absent:
        raise ValueError(f"archetype table is missing columns: {sorted(absent)}")

    if a[id_column].duplicated().any():
        n = int(a[id_column].duplicated().sum())
        raise ValueError(
            f"{n} duplicated {id_column} values in the archetype table; one "
            f"player must have one tag row or the bought-counters double-count"
        )

    out = pd.DataFrame(index=a[id_column].to_numpy())
    for name, col in _DIRECT.items():
        out[name] = a[col].fillna(False).to_numpy().astype(bool)

    pace = a["pace"].fillna(False).to_numpy().astype(bool)
    ra = a["RA"].fillna(False).to_numpy().astype(bool)
    la = a["LA"].fillna(False).to_numpy().astype(bool)

    # Every paced bowler must have exactly one arm, or the two pace archetypes
    # do not partition the pace group and one of them silently loses players.
    armless = int((pace & ~(ra | la)).sum())
    both = int((pace & ra & la).sum())
    if armless or both:
        raise ValueError(
            f"{armless} paced players have no arm tag and {both} have both; "
            f"right_arm_pace/left_arm_pace cannot be derived cleanly"
        )
    out["right_arm_pace"] = pace & ra
    out["left_arm_pace"] = pace & la

    role_cols = [c for c in ARCHETYPES if c != "untagged"]
    out["untagged"] = ~out[role_cols].any(axis=1)
    out.index.name = id_column
    return out[list(ARCHETYPES)]


def tags_for(tag_table, player_id):
    """The archetypes a player carries, as a tuple.  Unknown -> ('untagged',)."""
    if player_id not in tag_table.index:
        return ("untagged",)
    row = tag_table.loc[player_id]
    got = tuple(a for a in ARCHETYPES if bool(row[a]))
    return got or ("untagged",)


# ---------------------------------------------------------------------------
# Team state
# ---------------------------------------------------------------------------

def empty_team_archetype_counts():
    return {f"{a}_bought": 0 for a in ARCHETYPES}


def apply_purchase(team_counts, archetypes):
    """Increment every archetype the bought player carries (see note 1)."""
    for a in archetypes:
        team_counts[f"{a}_bought"] += 1


def team_archetype_features(team_counts, targets=None):
    """
    Per-archetype bought / still-wanted counts for one team.

    `*_needed` is the gap to the target, floored at zero.  It is emitted
    alongside `*_bought` rather than instead of it because the two are not
    redundant once a team is over target: four openers and three openers both
    give needed = 0, and the model should be able to tell a full cupboard from
    a just-filled one.
    """
    targets = targets or ARCHETYPE_TARGETS
    out = {}
    for a in ARCHETYPES:
        bought = int(team_counts.get(f"{a}_bought", 0))
        out[f"{a}_bought"] = bought
        out[f"{a}_needed"] = max(0, int(targets.get(a, 0)) - bought)
    return out


# ---------------------------------------------------------------------------
# Auction state
# ---------------------------------------------------------------------------

def pool_archetype_counts(remaining_ids, tag_table):
    """How many players carrying each archetype are still to come."""
    counts = {a: 0 for a in ARCHETYPES}
    for pid in remaining_ids:
        for a in tags_for(tag_table, pid):
            counts[a] += 1
    return counts


def auction_archetype_features(pool_counts):
    return {f"{a}_remaining": int(pool_counts.get(a, 0)) for a in ARCHETYPES}


# ---------------------------------------------------------------------------
# Supply and demand
# ---------------------------------------------------------------------------

def archetype_demand(team_states, targets=None, purse_key="remaining_purse",
                     slots_key="remaining_slots", min_purse=0.0):
    """
    Teams that still want each archetype *and* could still act on it.

    A team with no slots left, or without the purse to make even a minimum
    bid, is not demand however badly it needs a finisher -- counting it
    inflates scarcity exactly at the end of the auction, where scarcity is
    supposed to be falling.
    """
    targets = targets or ARCHETYPE_TARGETS
    demand = {a: 0 for a in ARCHETYPES}
    for state in team_states.values():
        if state.get(slots_key, 0) <= 0 or state.get(purse_key, 0.0) < min_purse:
            continue
        for a in ARCHETYPES:
            if state.get(f"{a}_bought", 0) < int(targets.get(a, 0)):
                demand[a] += 1
    return demand


def scarcity(pool_counts, demand_counts):
    """
    demand / supply per archetype.

    Supply is floored at 1 rather than guarded with a None: an archetype with
    live demand and nobody left to fill it is the most scarce state there is,
    and it should read as a large finite number, not as missing data.
    """
    return {
        a: demand_counts.get(a, 0) / max(pool_counts.get(a, 0), 1)
        for a in ARCHETYPES
    }


# ---------------------------------------------------------------------------
# The archetype-focused block
# ---------------------------------------------------------------------------

def focus_features(player_archetypes, team_counts, pool_counts, demand_counts,
                   remaining_ids, tag_table, targets=None, prefix="focus_"):
    """
    Everything above, narrowed to the player currently under the hammer.

    A player carries between one and four archetypes, so his own counters are
    ragged and are reduced to fixed width by min/mean/max over his own tag
    set.  Two extra counts are not reductions of anything and carry most of
    the signal:

      `substitutes_remaining` -- players still to come who share at least one
      archetype with him.  This is the number the bidding actually responds
      to: a team that misses him has this many other ways to fill the same
      hole, and when it is zero the bidding is a last chance.

      `interested_teams` -- teams that need at least one of his archetypes and
      can still bid.  Demand for *him*, rather than for a tag.
    """
    targets = targets or ARCHETYPE_TARGETS
    own = [a for a in player_archetypes if a in set(ARCHETYPES)] or ["untagged"]

    bought = [team_counts.get(f"{a}_bought", 0) for a in own]
    needed = [max(0, int(targets.get(a, 0)) - b) for a, b in zip(own, bought)]
    supply = [pool_counts.get(a, 0) for a in own]
    demand = [demand_counts.get(a, 0) for a in own]
    scarce = [d / max(s, 1) for d, s in zip(demand, supply)]

    own_set = set(own)
    subs = sum(
        1 for pid in remaining_ids
        if own_set & set(tags_for(tag_table, pid))
    )

    out = {
        f"{prefix}n_archetypes": len(own),
        f"{prefix}substitutes_remaining": int(subs),
        f"{prefix}team_bought_min": int(min(bought)),
        f"{prefix}team_bought_max": int(max(bought)),
        f"{prefix}team_bought_mean": float(np.mean(bought)),
        f"{prefix}team_needed_min": int(min(needed)),
        f"{prefix}team_needed_max": int(max(needed)),
        f"{prefix}team_needed_mean": float(np.mean(needed)),
        f"{prefix}supply_min": int(min(supply)),
        f"{prefix}supply_mean": float(np.mean(supply)),
        f"{prefix}demand_max": int(max(demand)),
        f"{prefix}demand_mean": float(np.mean(demand)),
        f"{prefix}scarcity_max": float(max(scarce)),
        f"{prefix}scarcity_mean": float(np.mean(scarce)),
        # Does this team specifically still want him? The single most direct
        # statement of fit, and the one a bid most immediately depends on.
        f"{prefix}team_wants_any": int(any(n > 0 for n in needed)),
    }
    return out


def interested_teams(player_archetypes, team_states, targets=None,
                     slots_key="remaining_slots"):
    """Teams with a free slot that are still short of one of his archetypes."""
    targets = targets or ARCHETYPE_TARGETS
    own = [a for a in player_archetypes if a in set(ARCHETYPES)] or ["untagged"]
    n = 0
    for state in team_states.values():
        if state.get(slots_key, 0) <= 0:
            continue
        if any(state.get(f"{a}_bought", 0) < int(targets.get(a, 0)) for a in own):
            n += 1
    return n


def feature_names(targets=None):
    """The full column vocabulary this module can emit, for attrs bookkeeping."""
    team = [f"{a}_{k}" for a in ARCHETYPES for k in ("bought", "needed")]
    auction = [f"{a}_remaining" for a in ARCHETYPES]
    focus = sorted(focus_features(
        ["untagged"], {}, {}, {}, [], pd.DataFrame(index=[]), targets).keys())
    return {"team": team, "auction": auction, "focus": focus}
