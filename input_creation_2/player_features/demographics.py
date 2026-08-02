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

import numpy as np
import pandas as pd


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
