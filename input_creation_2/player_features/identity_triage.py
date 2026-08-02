"""
Triage for unresolved auction playerIds.

`identity: 715/808 (88.5%)` is one number covering four different
situations, only two of which are fixable:

  no_t20_record   the cricketer exists but has never played a T20 in
                  any competition in the ball data.  Uncapped domestic
                  teenagers bought as futures, mostly.  There is no
                  Cricsheet person to resolve to.  NOT a failure; it
                  needs labelling, not fixing.
  ambiguous       several people are name-plausible and squad evidence
                  did not separate them.  Fixable by hand.
  conflict        two spellings of one playerId answered differently.
                  Fixable by hand.
  unresolved      nothing matched at all.  Usually because the two
                  sources use genuinely different names (Cricsheet
                  files Wanindu Hasaranga under "PWH de Silva"), not
                  because of a spelling variant.  Fixable by hand, via
                  squad residue -- see pipelines/identity/propose_matches.py.

Triage sorts the fixable ones by how much they cost.  A playerId that
appears in one auction at base price is worth less of your afternoon
than one who went for 14 crore, and fixing in price order gets the
model most of the benefit from the first dozen rows you add to
the cricinfo_resolution.csv carried by the inputs dataset.
"""

import re

import pandas as pd


def _parse_price(value):
    """
    Lakh, from the auction file's own money format.

    The rosters store "2.00 Cr" and "30.00 L" as strings, so a plain
    pd.to_numeric returns NaN for every row and the cost ranking below
    silently collapses to zero for all 93 players -- which is exactly
    the ordering that makes this report worth reading. Mirrors
    AuctionReplayEngine._parse_price.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return float("nan")
    match = re.match(r"([\d.]+)\s*(CR|CRORE|L|LAKH)?", text.upper())
    if match is None:
        return float("nan")
    amount = float(match.group(1))
    return amount * 100 if match.group(2) in ("CR", "CRORE") else amount


def triage_identity(feature_context, rosters, verbose=True):
    """
    feature_context : a PlayerFeatureContext with register_rosters()
                      already called.
    rosters         : {year: player_df}, the same dict passed to
                      register_rosters.

    Returns a DataFrame, one row per unresolved playerId, sorted by
    the money at stake.
    """

    resolver = feature_context.resolver

    every = pd.concat(
        [df.assign(season_year=year) for year, df in rosters.items()],
        ignore_index=True,
    )

    resolved_ids = set(feature_context.person_by_player)

    # Why each one failed, from the resolver's own bookkeeping.
    reason = {}
    detail = {}

    for pid_, name, hits in resolver.conflicts:
        reason[pid_] = "conflict"
        detail[pid_] = f"spellings disagreed between {hits}"

    for pid_, name, how, hits in resolver.ambiguous:
        reason[pid_] = "ambiguous"
        detail[pid_] = f"{how} matched {len(hits)} people: {hits[:4]}"

    for pid_, name in resolver.unresolved:
        reason[pid_] = "unresolved"
        detail[pid_] = "no name-plausible candidate"

    for pid_, name, hits in feature_context.identity_conflicts:
        reason[pid_] = "cross_year_conflict"
        detail[pid_] = f"resolved differently across years: {hits}"

    rows = []

    for pid_, grp in every.groupby("playerId"):

        if pid_ in resolved_ids:
            continue

        prices = (
            grp["auctionPrice"].map(_parse_price)
            if "auctionPrice" in grp.columns
            else pd.Series(dtype="float64")
        )

        rows.append(
            {
                "playerId": pid_,
                "playerName": grp["playerName"].iloc[0],
                "spellings": sorted(set(grp["playerName"].dropna())),
                "years": sorted(set(grp["season_year"])),
                "teams": (
                    sorted(set(grp["playsForTeam"].dropna()))
                    if "playsForTeam" in grp.columns
                    else []
                ),
                "appearances": len(grp),
                "max_price": (
                    float(prices.max()) if prices.notna().any() else 0.0
                ),
                "reason": reason.get(pid_, "unknown"),
                "detail": detail.get(pid_, ""),
            }
        )

    frame = pd.DataFrame(rows)

    if frame.empty:
        if verbose:
            print("identity triage: nothing unresolved")
        return frame

    # Cost of leaving it broken: an expensive player appearing in
    # several auctions poisons more rows than a one-off at base price.
    frame["cost"] = frame["max_price"].fillna(0) * frame["appearances"]

    frame = frame.sort_values("cost", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\nidentity triage: {len(frame)} unresolved playerIds")
        print(frame["reason"].value_counts().to_string())

        worth_it = frame[frame["max_price"] > 0]
        print(
            f"\n  {len(worth_it)} were bought at least once "
            f"(these are the ones worth fixing); "
            f"{len(frame) - len(worth_it)} never sold at any auction"
        )

        print("\n  top 15 by cost:")
        for r in frame.head(15).itertuples():
            print(
                f"    {r.playerName:28s} id={r.playerId:<8} "
                f"max={r.max_price:7.0f}  x{r.appearances:<3} "
                f"[{r.reason}] {r.detail[:70]}"
            )

    return frame


def classify_no_t20_record(triage_frame, feature_context, verbose=True):
    """
    Split "fixable" from "there is nothing to fix", and where possible
    name the candidate.

    The test that matters is squad RESIDUE, not squad coverage. Knowing
    that the 2025 CSK season exists in the ball data says nothing about
    whether a given unresolved playerId is in it -- we cannot look him
    up, which is the whole problem. What we can do is subtract:

        residue = everyone who played for that franchise that season
                  MINUS everyone already claimed by another playerId

    An empty residue means every cricketer who took the field for that
    team is already spoken for, so this playerId did not play -- there
    is genuinely nothing to resolve him to. A non-empty residue is a
    shortlist, often of one, and it is the answer.

    Verdicts, in `status`:

      fixable        a name tier found candidates and could not choose
                     between them. Disambiguation problem; a row in
                     cricinfo_resolution.csv settles it.
      residue_hit    no name matched, but subtraction leaves a short
                     candidate list. See `candidates`. This is the
                     Wanindu Hasaranga case: Cricsheet files him under
                     "PWH de Silva", which no string metric reaches,
                     but he is the only unclaimed man in that squad.
      no_t20_record  every franchise-season he was listed with is
                     covered by the ball data and has no unclaimed
                     player left. Label, do not fix.
      inconclusive   none of his franchise-seasons are in the ball data
                     (the 2026 auction has no 2026 deliveries), so
                     absence is not evidence. Needs a human.
    """

    squad_index = feature_context.resolver.squad_index

    claimed = set(feature_context.person_by_player.values())

    statuses, candidate_lists = [], []

    for r in triage_frame.itertuples():

        if r.reason in ("ambiguous", "conflict", "cross_year_conflict"):
            statuses.append("fixable")
            candidate_lists.append([])
            continue

        if squad_index is None:
            statuses.append("inconclusive")
            candidate_lists.append([])
            continue

        residue = set()
        covered_any = False

        for year in r.years:
            for team in r.teams:
                members = squad_index.members(year, team)
                if not members:
                    continue
                covered_any = True
                residue |= (members - claimed)

        if not covered_any:
            statuses.append("inconclusive")
            candidate_lists.append([])
        elif residue:
            statuses.append("residue_hit")
            candidate_lists.append(sorted(residue))
        else:
            statuses.append("no_t20_record")
            candidate_lists.append([])

    out = triage_frame.copy()
    out["status"] = statuses
    out["candidates"] = candidate_lists
    out["n_candidates"] = [len(c) for c in candidate_lists]

    if verbose:
        print()
        print(out["status"].value_counts().to_string())

        # Two unresolved playerIds proposing the SAME person is not a
        # match, it is a collision: the residue is computed per
        # playerId, so a squad with two unclaimed slots and two
        # unresolved names offers each of them both. At most one can be
        # right, and which one needs a human.
        from collections import Counter

        proposed = Counter(
            c
            for cands in out["candidates"]
            for c in cands
        )
        contested = {c for c, k in proposed.items() if k > 1}

        if contested:
            clash = out[out["candidates"].apply(
                lambda cs: any(c in contested for c in cs)
            )]
            print(
                f"\n  {len(clash)} playerIds propose a candidate that "
                f"another unresolved playerId also proposes -- at most "
                f"one of each pair can be right:"
            )
            for r in clash.itertuples():
                print(f"    {r.playerName:28s} id={r.playerId:<10} -> {r.candidates}")

        singles = out[
            (out["status"] == "residue_hit")
            & (out["n_candidates"] == 1)
            & (~out["candidates"].apply(lambda cs: any(c in contested for c in cs)))
        ]
        if len(singles):
            print(
                f"\n  {len(singles)} unresolved playerIds have exactly ONE "
                f"unclaimed candidate in a squad they were listed with. "
                f"These are near-certain matches -- verify and add:"
            )
            for r in singles.head(20).itertuples():
                print(
                    f"    {r.playerName:28s} id={r.playerId:<8} "
                    f"-> {r.candidates[0]}"
                )

        print(
            "\n  work order: residue_hit (cheapest, often already the "
            "answer), then fixable, then inconclusive. no_t20_record "
            "should be labelled, not resolved."
        )

    return out


def write_resolution_stubs(triage_frame, path, top_n=None):
    """
    Emit skeleton rows for the cricinfo_resolution.csv carried by the inputs dataset.

    Deliberately leaves cricinfo_id blank.  Filling it is the one step
    that has to be a human looking at a Cricinfo profile -- everything
    that could be automated is already in the resolver, and everything
    left needs somebody to decide which of two brothers this is.
    """

    frame = triage_frame if top_n is None else triage_frame.head(top_n)

    stub = pd.DataFrame(
        {
            "playerId": frame["playerId"],
            "playerName": frame["playerName"],
            "cricinfo_id": "",
            "dob": "",
            "method": "",
            "note": (
                frame["reason"].astype(str)
                + ": "
                + frame["detail"].astype(str).str.slice(0, 80)
            ),
        }
    )

    stub.to_csv(path, index=False)

    print(
        f"wrote {len(stub)} stub rows to {path}\n"
        f"  fill in cricinfo_id from the player's Cricinfo URL "
        f"(espncricinfo.com/cricketers/<slug>-<ID>), then append the "
        f"completed rows into cricinfo_resolution.csv and re-upload the "
        f"inputs dataset"
    )

    return stub