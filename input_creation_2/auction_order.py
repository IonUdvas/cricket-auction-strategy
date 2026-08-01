"""
Establish the order players went under the hammer.

The engine used to do this:

    self.player_df = self.player_df.iloc[::-1].reset_index(drop=True)

i.e. assume the roster CSV arrives newest-first and reverse it.  That
assumption is load-bearing -- `auction_order`, `players_remaining` and
every team's purse trajectory are derived from it, and the auction
adjustment head reads all of them -- and nothing checked it.  A file
that arrived in forward order produced a perfectly well-formed frame
describing an auction that ran backwards.

This module replaces the assumption with a search.  Candidate orderings
are scored against a fact that only holds for the real one: replayed in
true auction order, no team's cumulative spend ever exceeds its purse.
Replayed backwards, the expensive marquee lots settle last, after the
purse has already been drained by the cheap ones, and teams overdraw.

Order of preference:

  1. an explicit ordering column, when the file has one and it passes
  2. reverse-of-file, if it passes and forward does not
  3. file order, if it passes and reverse does not

If both directions pass, the file's own ordering column decides, and
without one the reverse is kept -- with a warning, because at that
point it is still just the old assumption.
"""

import numpy as np
import pandas as pd


# Columns that plausibly carry auction sequence, most explicit first.
# Matched case-insensitively on substrings, so "auctionTimestamp",
# "auction_ts" and "AUCTION TIME" all land on the same rule.
_ORDER_HINTS = [
    "lotnumber", "lot_no", "lotno", "lot",
    "auctionorder", "auction_order", "auctionindex",
    "serial", "srno", "sr_no", "slno",
    "auctiontimestamp", "auctiontime", "auctiondatetime",
    "timestamp", "unixtime", "epoch", "time", "date",
    "setnumber", "setno", "set",
]


def _candidate_order_columns(frame):
    """Columns that could plausibly order the auction, best guess first."""

    found = []

    for hint in _ORDER_HINTS:
        for col in frame.columns:
            key = col.lower().replace(" ", "").replace("-", "")
            if hint in key and col not in found:
                found.append(col)

    # Only keep ones that are actually usable as a sort key: numeric, or
    # parseable as a datetime, and with enough distinct values to order
    # most of the field rather than bucket it.
    usable = []

    for col in found:
        series = frame[col]

        as_num = pd.to_numeric(series, errors="coerce")
        if as_num.notna().mean() > 0.9:
            usable.append((col, as_num))
            continue

        as_dt = pd.to_datetime(series, errors="coerce", format="mixed")
        if as_dt.notna().mean() > 0.9:
            usable.append((col, as_dt.astype("int64")))

    return usable


def _overdraft(frame, max_purse, price_col="auctionPrice",
               team_col="playsForTeam", status_col="auctionStatus"):
    """
    Worst amount by which any team's running spend exceeds its purse,
    replaying `frame` top to bottom.  Zero for a plausible ordering.
    """

    bought = frame[
        frame[status_col].astype(str).str.upper().isin(
            ["SOLD", "RTM", "RETAINED"]
        )
    ]

    if bought.empty:
        return 0.0

    worst = 0.0

    for _, grp in bought.groupby(team_col):
        spend = pd.to_numeric(grp[price_col], errors="coerce").fillna(0).cumsum()
        worst = max(worst, float((spend - max_purse).max()))

    return max(worst, 0.0)


def _monotone_fraction(values):
    """How much of a candidate key is already sorted ascending."""
    arr = np.asarray(values, dtype="float64")
    if len(arr) < 2:
        return 1.0
    return float(np.mean(np.diff(arr) >= 0))


def resolve_auction_order(player_df, max_purse, verbose=True):
    """
    Return (ordered_frame, decision_dict).

    `decision_dict` records what was chosen and why, so the choice
    shows up in the build log rather than being a silent assumption.
    """

    forward = player_df.reset_index(drop=True)
    reverse = player_df.iloc[::-1].reset_index(drop=True)

    over_forward = _overdraft(forward, max_purse)
    over_reverse = _overdraft(reverse, max_purse)

    decision = {
        "overdraft_file_order": over_forward,
        "overdraft_reversed": over_reverse,
        "candidate_columns": [],
        "method": None,
        "column": None,
        "warning": None,
    }

    # ------------------------------------------------------------------
    # 1. An explicit ordering column, if the file has one that works.
    # ------------------------------------------------------------------

    for col, key in _candidate_order_columns(player_df):

        ascending_frame = (
            player_df.assign(_k=key.to_numpy())
            .sort_values("_k", kind="mergesort")
            .drop(columns="_k")
            .reset_index(drop=True)
        )

        descending_frame = ascending_frame.iloc[::-1].reset_index(drop=True)

        over_asc = _overdraft(ascending_frame, max_purse)
        over_desc = _overdraft(descending_frame, max_purse)

        decision["candidate_columns"].append(
            {
                "column": col,
                "distinct": int(pd.Series(key).nunique()),
                "already_sorted": round(_monotone_fraction(key), 3),
                "overdraft_ascending": over_asc,
                "overdraft_descending": over_desc,
            }
        )

        # A usable key must order nearly every player distinctly --
        # a "set number" with 12 values groups lots, it does not
        # sequence them, and sorting by it shuffles within each set.
        if pd.Series(key).nunique() < 0.9 * len(player_df):
            continue

        if over_asc == 0:
            decision.update(method="column_ascending", column=col)
            return ascending_frame, decision

        if over_desc == 0:
            decision.update(method="column_descending", column=col)
            return descending_frame, decision

    # ------------------------------------------------------------------
    # 2/3. Fall back to direction, decided by the purse test.
    # ------------------------------------------------------------------

    if over_reverse == 0 and over_forward > 0:
        decision["method"] = "reversed_file_order"
        chosen = reverse

    elif over_forward == 0 and over_reverse > 0:
        decision["method"] = "file_order"
        decision["warning"] = (
            "file order passes the purse test and the reverse does not, "
            "so this file is NOT newest-first -- the engine's historical "
            "iloc[::-1] would have replayed it backwards"
        )
        chosen = forward

    elif over_forward == 0 and over_reverse == 0:
        decision["method"] = "reversed_file_order"
        decision["warning"] = (
            "both directions satisfy the purse constraint, so it cannot "
            "distinguish them; keeping the historical reversal. Add an "
            "explicit lot-number or timestamp column to settle this."
        )
        chosen = reverse

    else:
        decision["method"] = "reversed_file_order"
        decision["warning"] = (
            f"NEITHER direction satisfies the purse constraint "
            f"(file {over_forward:.0f}, reversed {over_reverse:.0f} over "
            f"a purse of {max_purse}). The prices, the purse or the "
            f"roster is wrong -- team_state features from this year are "
            f"not trustworthy."
        )
        chosen = reverse if over_reverse <= over_forward else forward

    if verbose:
        print(f"  auction order: {decision['method']}", end="")
        if decision["column"]:
            print(f" (column {decision['column']!r})", end="")
        print()
        if decision["warning"]:
            print(f"    WARNING: {decision['warning']}")

    return chosen, decision


def describe_order_candidates(player_df, max_purse):
    """
    Inspect what ordering evidence a roster file actually contains.

    Run this once per year's CSV when adding a new season, before
    trusting the automatic choice.
    """

    _, decision = resolve_auction_order(player_df, max_purse, verbose=False)

    print(f"file order overdraft : {decision['overdraft_file_order']:.0f}")
    print(f"reversed   overdraft : {decision['overdraft_reversed']:.0f}")
    print(f"chosen               : {decision['method']} {decision['column'] or ''}")

    if decision["candidate_columns"]:
        print("\ncandidate ordering columns:")
        print(pd.DataFrame(decision["candidate_columns"]).to_string(index=False))
    else:
        print(
            "\nno candidate ordering column found. Columns available:\n  "
            + ", ".join(map(str, player_df.columns))
        )

    return decision