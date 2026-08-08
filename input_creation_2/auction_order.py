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


def resolve_auction_order(player_df, max_purse, verbose=True, bid_df=None):
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

            ####################################################
            # `_overdraft` returning 0 does NOT mean this order
            # is right -- it returns 0 for every permutation
            # (see _bid_infeasibility). Check the order-sensitive
            # constraint, and repair rows whose timestamp puts
            # them outside the auction window entirely.
            ####################################################
            if bid_df is not None:
                sorted_key = (
                    pd.Series(key.to_numpy())
                    .sort_values(kind="mergesort")
                    .to_numpy()
                )
                before = _bid_infeasibility(
                    ascending_frame, bid_df, max_purse)
                decision["bid_infeasibility"] = before

                if before > 0:
                    repaired, info = repair_stale_order(
                        ascending_frame, bid_df, max_purse,
                        sorted_key, verbose=verbose,
                    )
                    after = _bid_infeasibility(
                        repaired, bid_df, max_purse)
                    decision["bid_infeasibility_after"] = after
                    decision["stale_repair"] = info
                    if after < before:
                        decision.update(
                            method="column_ascending_repaired")
                        return repaired, decision

            return ascending_frame, decision

        if over_desc == 0:
            decision.update(method="column_descending", column=col)
            return descending_frame, decision

        # Neither direction of this column clears the purse test.
        #
        # That is NOT a reason to discard the column. The purse test is
        # a statement about prices; an explicit lot number is a
        # statement about order, and the two fail independently. If the
        # price column is corrupt -- duplicated sale prices, a mis-join,
        # a wrong purse constant -- every ordering will overdraw, and
        # falling through to "reverse the file and hope" throws away the
        # one piece of hard evidence the file actually contains.
        #
        # So keep the column, take whichever direction overdraws less,
        # and say plainly that the prices are the thing to fix.
        best_frame, best_over, direction = (
            (ascending_frame, over_asc, "ascending")
            if over_asc <= over_desc
            else (descending_frame, over_desc, "descending")
        )

        decision.update(
            method=f"column_{direction}_unverified",
            column=col,
            warning=(
                f"ordering taken from {col!r} ({direction}), but no "
                f"ordering of this auction satisfies the purse "
                f"constraint (best overdraft {best_over:.0f} against a "
                f"purse of {max_purse}). The order is probably right and "
                f"the PRICES are probably wrong -- check auctionPrice "
                f"and the purse constant before trusting team_state."
            ),
        )

        if verbose:
            print(f"  auction order: {decision['method']} ({col!r})")
            print(f"    WARNING: {decision['warning']}")

        return best_frame, decision

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


# ---------------------------------------------------------------------------
# Order-sensitive validity: can every recorded bid actually be paid for?
# ---------------------------------------------------------------------------

def _bid_infeasibility(frame, bid_df, max_purse, team_col="playsForTeam",
                       price_col="auctionPrice", status_col="auctionStatus"):
    """
    Total shortfall, in lakh, over every bid a team could not have afforded
    when `frame` is replayed top to bottom.

    This is the test `_overdraft` was meant to be and is not. `_overdraft`
    compares each team's CUMULATIVE spend against its purse; because no
    team's total spend ever exceeds its purse, that quantity is zero for
    every permutation of the roster, so it can never reject an ordering.

    A bid is different. When a team bids X on a lot, it must hold at least
    X at that instant. Move an expensive lot from the start of the auction
    to the end and its underbidders are suddenly bidding crores they have
    already spent -- which is exactly what a stale `updatedTime` does. That
    is order-sensitive, so it discriminates.

    Returns 0.0 when nothing in the ordering is impossible.
    """
    if bid_df is None or not len(bid_df):
        return 0.0

    need = _bid_requirements(bid_df)

    purse = {}
    for t in set(frame[team_col].dropna()) | {
            t for v in need.values() for t, _ in v}:
        purse[t] = float(max_purse)

    paid = frame[status_col].astype(str).str.upper().isin(
        ["SOLD", "RTM", "RETAINED", "TRADED"])

    shortfall = 0.0
    import pandas as _pd

    for row, is_paid in zip(frame.itertuples(index=False), paid):
        pid_ = getattr(row, "playerId", None)
        for team, amount in need.get(pid_, ()):
            if team in purse and purse[team] < amount:
                shortfall += amount - purse[team]
        if is_paid:
            price = _pd.to_numeric(
                _pd.Series([getattr(row, price_col, None)]),
                errors="coerce").iloc[0]
            team = getattr(row, team_col, None)
            if team in purse and price == price:
                purse[team] -= float(price)

    return float(shortfall)


def _stale_timestamp_mask(key, window_days=4.0):
    """
    Which timestamp values sit outside the auction's own window.

    An IPL auction runs over one to three days, so the honest timestamps
    form a tight cluster. `updatedTime` is a record-modified time, though,
    so any row touched afterwards -- a profile edit, a later re-scrape --
    carries a timestamp days or years away and sorts to the end of the
    auction. Rishabh Pant's 2025 record reads 2025-11-27 against an auction
    held on 2024-11-24, which placed the most expensive lot in IPL history
    dead last, after LSG had spent its purse.

    Measured against the MEDIAN rather than the mean or the range: the
    stale rows are exactly the outliers, so any statistic they can drag is
    the wrong one to compare them against.
    """
    import numpy as _np
    import pandas as _pd

    v = _pd.to_numeric(_pd.Series(_np.asarray(key, dtype="float64")),
                       errors="coerce")
    if v.notna().sum() < 3:
        return _np.zeros(len(v), dtype=bool)

    # Heuristic unit detection: seconds since epoch vs nanoseconds.
    span = float(v.max() - v.min())
    scale = 1e9 if v.median() > 1e17 else 1.0
    window = window_days * 86400.0 * scale

    med = float(v.median())
    return (v - med).abs().to_numpy() > window


def repair_stale_order(frame, bid_df, max_purse, key, window_days=4.0,
                       verbose=True, max_repairs=6):
    """
    Re-place rows whose ordering timestamp is outside the auction window.

    Each stale row is moved to the LATEST position at which every bid
    recorded on that player is affordable. Latest-feasible rather than
    first: it is the weakest claim the data supports -- "no later than
    this" -- rather than an assumption that edited records were marquee
    lots.

    Stale rows are placed most expensive first, because an expensive lot
    constrains the trajectory of every cheaper one.
    """
    import numpy as _np
    import pandas as _pd

    stale = _stale_timestamp_mask(key, window_days)

    ####################################################################
    # A timestamp inside the window can still be wrong. The window test
    # catches a record edited a year later; it does not catch one edited
    # the morning after. So the mask is widened by the constraint itself:
    # any player some team is modelled as outbidding with money it no
    # longer has is mis-placed, whatever its timestamp says.
    ####################################################################
    stale = stale | _infeasible_player_mask(frame, bid_df, max_purse)

    if not stale.any():
        return frame, {"stale_rows": 0, "moved": []}

    # Bound the search: it is O(n^2) per repaired row, and beyond the
    # worst offenders the residual is noise rather than a misplaced lot.
    if stale.sum() > max_repairs:
        price = _pd.to_numeric(frame["auctionPrice"], errors="coerce")
        keep = price.where(stale).nlargest(max_repairs).index
        stale = frame.index.isin(keep)

    clean = frame.loc[~stale].reset_index(drop=True)
    dirty = frame.loc[stale].copy()
    dirty["_price"] = _pd.to_numeric(dirty["auctionPrice"], errors="coerce")
    dirty = dirty.sort_values("_price", ascending=False, na_position="last")
    dirty = dirty.drop(columns="_price")

    moved = []
    out = clean

    for _, row in dirty.iterrows():
        # Coarse-to-fine scan of the insertion point. Evaluating every
        # position is O(n) calls of an O(n) test per repaired row, which
        # is minutes per auction; a stride pass followed by a local
        # refinement around the winner is the same answer for a fraction
        # of the work, because infeasibility is monotone in position
        # (moving a lot later can only ever cost a bidder money it has
        # already spent).
        def _score(pos):
            cand = _pd.concat(
                [out.iloc[:pos], row.to_frame().T, out.iloc[pos:]],
                ignore_index=True,
            )
            return _bid_infeasibility(cand, bid_df, max_purse)

        n = len(out)
        stride = max(1, n // 24)
        best_pos, best_bad = 0, None

        coarse = list(range(0, n + 1, stride))
        if coarse[-1] != n:
            coarse.append(n)

        for pos in coarse:
            bad = _score(pos)
            if best_bad is None or bad < best_bad - 1e-9:
                best_bad, best_pos = bad, pos
            elif bad <= best_bad + 1e-9 and bad == 0.0:
                best_pos = pos          # latest feasible wins

        for pos in range(max(0, best_pos - stride),
                         min(n, best_pos + stride) + 1):
            bad = _score(pos)
            if bad < best_bad - 1e-9:
                best_bad, best_pos = bad, pos
            elif bad <= best_bad + 1e-9 and bad == 0.0:
                best_pos = pos
        out = _pd.concat(
            [out.iloc[:best_pos], row.to_frame().T, out.iloc[best_pos:]],
            ignore_index=True,
        )
        moved.append((row.get("playerName"), best_pos, best_bad))

    if verbose:
        print(f"    stale-timestamp repair: {int(stale.sum())} row(s) "
              f"outside the auction window re-placed")
        for name, pos, bad in moved:
            print(f"      {name!r} -> position {pos} "
                  f"(residual infeasibility {bad:.0f})")

    return out.reset_index(drop=True), {
        "stale_rows": int(stale.sum()),
        "moved": moved,
    }


def _infeasible_player_mask(frame, bid_df, max_purse,
                            team_col="playsForTeam",
                            price_col="auctionPrice",
                            status_col="auctionStatus"):
    """
    Which rows carry a bid the bidding team could not have afforded.

    Same walk as `_bid_infeasibility`, but it attributes the shortfall to
    the lot rather than summing it, so the repair knows which rows to move.
    """
    import numpy as _np
    import pandas as _pd

    out = _np.zeros(len(frame), dtype=bool)
    if bid_df is None or not len(bid_df):
        return out

    need = _bid_requirements(bid_df)

    purse = {}
    for t in set(frame[team_col].dropna()) | {
            t for v in need.values() for t, _ in v}:
        purse[t] = float(max_purse)

    paid = frame[status_col].astype(str).str.upper().isin(
        ["SOLD", "RTM", "RETAINED", "TRADED"]).to_numpy()
    prices = _pd.to_numeric(frame[price_col], errors="coerce").to_numpy()
    teams = frame[team_col].to_numpy()
    pids = (frame["playerId"].to_numpy() if "playerId" in frame.columns
            else _np.full(len(frame), None))

    for i in range(len(frame)):
        for team, amount in need.get(pids[i], ()):
            if team in purse and purse[team] < amount - 1e-9:
                out[i] = True
        if paid[i] and teams[i] in purse and prices[i] == prices[i]:
            purse[teams[i]] -= float(prices[i])

    return out


_BID_REQ_CACHE = {}


def _bid_requirements(bid_df):
    """
    playerId -> [(team, highest bid that team placed on him)].

    Cached on the identity of the bid frame. Both feasibility walks call
    this once per candidate position, and rebuilding it is a groupby over
    every bid in the auction -- which made the repair search quadratic in
    the wrong variable.
    """
    key = id(bid_df)
    hit = _BID_REQ_CACHE.get(key)
    if hit is not None and hit[0] is bid_df:
        return hit[1]

    need = {}
    for pid_, grp in bid_df.groupby("playerId"):
        by_team = grp.groupby("Team")["BidAmount"].max()
        need[pid_] = [(t, float(v)) for t, v in by_team.items() if v == v]

    _BID_REQ_CACHE.clear()
    _BID_REQ_CACHE[key] = (bid_df, need)
    return need
