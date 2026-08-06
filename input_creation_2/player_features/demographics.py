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
    """'27.00 Cr' / '30.00 L' / '--' -> lakh, or None."""
    if value is None or isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if text in ("", "--", "nan", "None"):
        return None
    match = re.match(r"([\d.]+)\s*(cr|l)\b", text, flags=re.IGNORECASE)
    if match:
        amount = float(match.group(1))
        return amount * 100.0 if match.group(2).lower() == "cr" else amount
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


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
