"""
Player demographics and auction history, computed as of each row's own date.

These are the three requested player features that are not derivable from ball
data: age, country, and what the player was last paid.

All three are trivial to compute and easy to leak, and the leak is the reason
this is a module rather than three lines inline.  `player_archetypes.csv`
already ships `age_at_last_auction`, and using it would put a 2026 age on a
2019 row -- the config file already calls this out and drops the column.  The
fix is not to drop the information but to recompute it against the auction
date of the row being built, which is what `age_at` does.

`last_salary` has the same shape and is worse, because the natural
implementation is a groupby-max over the player's prices and that reads the
future directly.  It must be the most recent price *strictly before* the
current auction, and a player at his first auction must get `None` rather
than 0 -- a debutant is not someone who was paid nothing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from input_creation_2.money import parse_money


def age_at(date_of_birth, as_of_date):
    """Age in years at `as_of_date`.  None when the DOB is unknown."""
    dob = pd.to_datetime(date_of_birth, errors="coerce")
    ref = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(dob) or pd.isna(ref):
        return None
    return float((ref - dob).days) / 365.25


def build_demographics(archetype_df, id_column="player_id"):
    """player_id -> {date_of_birth, country}, the inputs to `age_at`."""
    cols = [id_column, "date_of_birth", "country"]
    missing = set(cols) - set(archetype_df.columns)
    if missing:
        raise ValueError(f"archetype table is missing {sorted(missing)}")
    d = archetype_df[cols].drop_duplicates(id_column).set_index(id_column)
    d["date_of_birth"] = pd.to_datetime(d["date_of_birth"], errors="coerce")
    return d


def add_age_feature(frame, demographics, id_column="playerId",
                    date_column="auction_date", out_column="age"):
    """
    Attach age computed against each row's own auction date.

    Emitted as a value plus `<out>_is_missing`, matching the convention
    PlayerFeatureBuilder uses, so a missing DOB is a learnable state rather
    than a 0-year-old.
    """
    dob = frame[id_column].map(demographics["date_of_birth"])
    ref = pd.to_datetime(frame[date_column], errors="coerce")
    age = (ref - dob).dt.days / 365.25
    out = frame.copy()
    out[out_column] = age.astype(float)
    out[f"{out_column}_is_missing"] = age.isna().astype(float)
    out[out_column] = out[out_column].fillna(0.0)
    return out


def build_salary_history(auction_rows, id_column="playerId",
                         year_column="auction_year", price_column="auctionPrice",
                         sold_statuses=("SOLD", "RETAINED", "TRADED")):
    """
    Long table of every price a player has actually been paid.

    Only rows where money changed hands count.  An unsold player has no
    salary that year -- not a salary of zero -- and including him would drag
    every subsequent `last_salary` toward the floor.
    """
    a = auction_rows
    if "auctionStatus" in a.columns:
        a = a[a["auctionStatus"].str.upper().isin(
            {s.upper() for s in sold_statuses})]
    a = a[a[price_column].notna()]
    out = a[[id_column, year_column, price_column]].copy()
    out.columns = ["player_id", "year", "price"]
    return out.sort_values(["player_id", "year"]).reset_index(drop=True)


def build_salary_history_from_earnings(earnings_frames, id_column="playerId",
                                       season_column="Season",
                                       amount_column="Amount",
                                       price_parser=None):
    """
    Salary history from the earnings tables, which include RETENTIONS.

    `build_salary_history` can only see what the auction trail sees,
    and the replay engine removes retained/traded/drafted players from
    the pool before emitting rows -- correctly, they never went under
    the hammer. The consequence is that a player retained for three
    seasons and then re-auctioned arrives at that auction with no
    prior price at all, which is why established internationals show
    up with last_salary_is_missing = 1: Rishabh Pant enters the 2025
    auction as a debutant despite two seasons at 16 crore.

    The earnings files carry exactly the missing rows -- one
    (player, season, amount) per season played, retention included --
    and nothing in the repo read them. Across the nine shipped files
    they hold 2,943 distinct (player, season) salaries spanning
    2008-2026, against the 963 the auction trail alone recovers, and
    the files agree with each other on every overlapping pair.

    earnings_frames : iterable of DataFrames, or one DataFrame.
        Read from data_sources.earnings_template().

    Returns the same (player_id, year, price) contract as
    build_salary_history, so it drops into add_last_salary_feature
    unchanged. `year` is the SEASON the money was earned.

    On leakage: a season-S salary is fixed at the auction that
    precedes season S, so it is known before any later auction. The
    strictly-before rule in add_last_salary_feature therefore remains
    exactly right -- a row at the auction for season T may see S < T
    and must not see S == T, which is its own label.
    """
    if isinstance(earnings_frames, pd.DataFrame):
        earnings_frames = [earnings_frames]

    parse = price_parser or _parse_money

    parts = []
    for frame in earnings_frames:
        missing = {id_column, season_column, amount_column} - set(frame.columns)
        if missing:
            raise ValueError(
                f"earnings frame is missing {sorted(missing)}; expected the "
                f"columns written by the earnings scrape"
            )
        part = frame[[id_column, season_column, amount_column]].copy()
        part.columns = ["player_id", "year", "price"]
        part["price"] = part["price"].apply(parse)
        part["year"] = pd.to_numeric(part["year"], errors="coerce")
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["player_id", "year", "price"])
    out["year"] = out["year"].astype(int)

    # One row per (player, season). The files overlap heavily -- each
    # is a snapshot of a player's whole history as of that edition --
    # and they agree, so any of the duplicates will do.
    out = out.drop_duplicates(["player_id", "year"])

    return out.sort_values(["player_id", "year"]).reset_index(drop=True)


def _parse_money(value):
    """
    '27.00 Cr' / '30.00 L' / '--' -> lakh, or None.

    Delegates to input_creation_2.money.parse_money so that the
    salary history and the replay engine cannot disagree about what a
    price string means. The previous local regex required the unit
    with a word boundary, which made "27.00 Crore" fall through to
    float() and return None -- a silently dropped salary rather than
    an error. See money.py.
    """
    return parse_money(value, require_unit=False)


def last_salary(salary_history, player_id, before_year):
    """
    The most recent price strictly before `before_year`, or None.

    Strictly before is the leakage boundary and it is the whole point of the
    function: a 2025 row must not be able to see the 2025 hammer price, which
    is the label.
    """
    rows = salary_history[
        (salary_history["player_id"] == player_id)
        & (salary_history["year"] < before_year)
    ]
    if len(rows) == 0:
        return None
    return float(rows.iloc[-1]["price"])


def add_last_salary_feature(frame, salary_history, id_column="playerId",
                            year_column="auction_year", out_column="last_salary"):
    """
    Vectorised `last_salary` over a whole frame.

    Implemented as a merge_asof on year rather than a per-row lookup: at
    ~15k rows x ~800 players the loop is the slowest thing in the build, and
    merge_asof with `allow_exact_matches=False` encodes the strictly-before
    rule in the join itself rather than in a filter that can be edited out.
    """
    left = frame[[id_column, year_column]].copy()
    left.columns = ["player_id", "year"]
    left["_row"] = np.arange(len(left))
    left = left.sort_values("year")

    right = salary_history.sort_values("year")
    merged = pd.merge_asof(
        left, right, on="year", by="player_id",
        direction="backward", allow_exact_matches=False,
    ).sort_values("_row")

    out = frame.copy()
    price = merged["price"].to_numpy()
    out[out_column] = np.nan_to_num(price, nan=0.0)
    out[f"{out_column}_is_missing"] = np.isnan(price).astype(float)
    return out


####################################################################
# Capped status, as of each row's own auction date.
#
# WHY THIS EXISTS
#
# `cappedStatus` on the auction roster is one of the strongest
# signals in the dataset -- capped players clear at 6-12x the
# uncapped median in every edition where the column is populated:
#
#     2022  240L vs 20L     2025  320L vs 30L
#     2024  200L vs 20L     2026  200L vs 30L
#
# and it is UNPOPULATED for 2018, 2019, 2020 and 2021: every player
# in those four rosters reads UNCAPPED (2019 has exactly one CAPPED,
# itself an artifact). That is 690 in-pool players, roughly half of
# them genuinely capped internationals, all given one value.
# add_player_context_features now demotes those editions to missing
# rather than asserting the falsehood, but that only stops the lie --
# it does not recover the signal.
#
# This recovers it, from Cricbuzz profile debut dates scraped by
# pipelines/scrape_cricbuzz_profiles.py. Validated against the five
# editions that DO carry a real cappedStatus: 98.3% agreement,
# against a 44.7% "call everyone uncapped" baseline.
#
# ON LEAKAGE: a debut is a dated event, so
#     capped(t) = any(debut_date < t)
# is the same strictly-before rule last_salary uses. Nothing here can
# see past the auction it is describing.
#
# KNOWN RESIDUAL (~1.7%, measured, not estimated). Two causes, both
# understood, neither fixable from debut dates alone:
#
#   1. The five-year reversion rule. BCCI treats a player who has not
#      appeared internationally in the preceding five years as
#      uncapped again. Debut dates give the FIRST cap, never the last,
#      so this rule cannot be evaluated here. Piyush Chawla (last cap
#      2012), Karn Sharma (2014) and Mayank Markande (2019) all read
#      CAPPED here and UNCAPPED on the 2025 roster, correctly by that
#      rule. This is most of the residual and it is one-directional:
#      the derived column over-reports capped for long-retired
#      internationals.
#
#   2. The roster column is itself sometimes a POST-HOC snapshot
#      rather than an as-of fact. Ravi Bishnoi is recorded CAPPED on
#      the 2022 roster, but his international debut was 2022-02-16 --
#      four days AFTER the 2022-02-12 auction. In that case the
#      derived value is the correct one and the "disagreement" is the
#      roster being wrong. So the 98.3% is a floor on agreement with
#      truth, not a ceiling on correctness.
#
# Because of (2), this is emitted for ALL NINE editions rather than
# only the four broken ones: one consistently-defined column across
# the panel beats a patch that behaves differently before and after
# 2022. The roster's own cappedStatus still reaches the model
# separately via ctx_cappedStatus, so nothing is lost by having both.
####################################################################

# International formats only. An IPL debut is not a cap.
INTERNATIONAL_DEBUT_COLUMNS = ("test_debut", "odi_debut", "t20i_debut")
INTERNATIONAL_LAST_COLUMNS = ("last_test", "last_odi", "last_t20i")

# BCCI's reversion window. Validated by
# pipelines/scrape_cricbuzz_profiles.sweep_reversion rather than assumed.
DEFAULT_REVERSION_YEARS = 5


def build_debut_table(debut_df, id_column="playerId"):
    """
    playerId -> (earliest international debut, latest international
    appearance), both as Timestamps, as a two-column DataFrame.

    `debut_df` is what scrape_cricbuzz_profiles.scrape() returns. The
    LAST-appearance half is what makes the five-year reversion rule
    decidable; a debut date alone can only say whether a player was
    EVER capped, never whether he still is.

    Players with no international debut on record are dropped, so they
    fall through to the missing-flag downstream rather than being
    asserted uncapped.
    """
    debuts = [c for c in INTERNATIONAL_DEBUT_COLUMNS if c in debut_df.columns]
    if not debuts:
        raise ValueError(
            f"debut table has none of {list(INTERNATIONAL_DEBUT_COLUMNS)}; got "
            f"{list(debut_df.columns)}. Expected the frame written by "
            f"pipelines/scrape_cricbuzz_profiles.py."
        )
    lasts = [c for c in INTERNATIONAL_LAST_COLUMNS if c in debut_df.columns]

    d = debut_df[[id_column] + debuts + lasts].copy()
    d[id_column] = pd.to_numeric(d[id_column], errors="coerce")
    for c in debuts + lasts:
        d[c] = pd.to_datetime(d[c], errors="coerce")

    out = pd.DataFrame({
        "debut": d[debuts].min(axis=1, skipna=True),
        # No last-match column parsed -> the debut stands in for it. A
        # player with one cap twenty years ago and no last-match record
        # should revert, and his debut is the last appearance we can
        # actually evidence.
        "last": (d[lasts].max(axis=1, skipna=True) if lasts
                 else d[debuts].min(axis=1, skipna=True)),
    })
    out.index = d[id_column].to_numpy()
    out["last"] = out["last"].fillna(out["debut"])
    out = out[~out.index.duplicated(keep="first")]
    return out.dropna(subset=["debut"])


def add_capped_feature(frame, debut_table, id_column="playerId",
                       date_column="auction_date", out_column="capped",
                       reversion_years=DEFAULT_REVERSION_YEARS):
    """
    Capped status as of each row's own auction date, with reversion.

    Emitted as 0/1 plus `<out>_is_missing`, so "no debut on record" is
    a learnable state rather than a confident UNCAPPED -- most of the
    pool is genuinely uncapped domestic players, and a failed scrape
    looks identical to a real debutant unless the flag separates them.

    reversion_years=None turns the rule off (capped forever once
    capped). Any number N means "capped only if the last international
    appearance is not more than N years before this auction".

    On the post-auction branch: when a player's last international
    post-dates the auction being scored, this data cannot say whether
    he appeared in the N years immediately before it, only that he was
    active at some point after. He is left CAPPED. The alternative --
    ignoring appearances after the auction -- would revert every
    still-active veteran whose debut predates the window, which is
    badly wrong. See capped_as_of in the scraper for the same note.
    """
    out = frame.copy()

    if debut_table is None or len(debut_table) == 0:
        out[out_column] = 0.0
        out[f"{out_column}_is_missing"] = 1.0
        return out

    debut = frame[id_column].map(debut_table["debut"])
    last = frame[id_column].map(debut_table["last"])
    ref = pd.to_datetime(frame[date_column], errors="coerce")

    known = debut.notna()
    capped = (debut < ref) & known

    if reversion_years is not None:
        cutoff = ref - pd.DateOffset(years=int(reversion_years))
        # Revert only when the last appearance is BOTH before this
        # auction and older than the window.
        reverted = capped & last.notna() & (last < ref) & (last < cutoff)
        capped = capped & ~reverted

    out[out_column] = capped.astype(float)
    out[f"{out_column}_is_missing"] = (~known).astype(float)
    return out
