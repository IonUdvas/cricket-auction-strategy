"""
In-pipeline verification of the built training frames.

`build_training_df` calls this after every year is built.  It is
deliberately cheap and deliberately noisy: the failures it looks for
are the ones that produce a frame of the right shape full of wrong
numbers, which is the only kind of failure this pipeline has actually
had.

Nothing here raises by default.  Pass strict=True to turn the
structural checks into exceptions once a year is known good.
"""

import numpy as np
import pandas as pd


def verify_year(year, training_df, engine_report=None, verbose=True):
    """
    Structural checks on one year's training frame.

    engine_report : AuctionReplayEngine.quality_report(), if available.
    """

    findings = []

    def note(severity, check, value, detail=""):
        findings.append(
            {
                "year": year,
                "severity": severity,
                "check": check,
                "value": value,
                "detail": detail,
            }
        )

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    n_players = training_df["playerId"].nunique()
    n_teams = training_df["team"].nunique()
    expected = n_players * n_teams

    note(
        "error" if len(training_df) > expected else "ok",
        "rows vs players x teams",
        len(training_df),
        f"{n_players} players x {n_teams} teams = {expected}"
        + (
            f"; {expected - len(training_df)} fewer, expected only if "
            "intervals were dropped"
            if len(training_df) < expected
            else ""
        ),
    )

    dupes = int(training_df.duplicated(["playerId", "team"]).sum())
    note("error" if dupes else "ok", "duplicate (playerId, team)", dupes)

    # ------------------------------------------------------------------
    # Teams
    #
    # The team embedding is sized from the vocabulary pooled over every
    # year, so a 2018 frame legitimately sits inside a 10-team
    # embedding table even though only 8 franchises played -- the 2
    # unused rows are simply never looked up that year.  That is
    # correct and is NOT what to worry about.  What matters is the
    # reverse: a team that appears in validation but has no training
    # rows gets an embedding that was randomly initialised and never
    # updated.
    # ------------------------------------------------------------------

    teams = sorted(training_df["team"].dropna().unique())
    note("info", "teams this year", n_teams, ", ".join(teams))

    per_team = training_df["team"].value_counts()
    lopsided = per_team[per_team < 0.5 * per_team.median()]
    note(
        "warn" if len(lopsided) else "ok",
        "teams with unusually few rows",
        len(lopsided),
        lopsided.to_dict() if len(lopsided) else "",
    )

    # ------------------------------------------------------------------
    # Observation mix
    # ------------------------------------------------------------------

    mix = training_df["observation_type"].value_counts()
    note(
        "info",
        "observation_type mix",
        len(mix),
        {k: int(v) for k, v in mix.items()},
    )

    winners = int(training_df["winner"].fillna(False).sum())
    sold = int(
        training_df.loc[
            training_df["auctionStatus"].isin(["SOLD", "RTM"]), "playerId"
        ].nunique()
    )
    note(
        "error" if winners != sold else "ok",
        "winner rows vs sold players",
        winners,
        f"{sold} players sold/RTM; every one needs exactly one winner row",
    )

    # ------------------------------------------------------------------
    # Intervals
    # ------------------------------------------------------------------

    known = training_df[training_df["observation_type"] != "unknown"]

    bad_bounds = int(
        (
            known["lower"].isna()
            | known["upper"].isna()
            | (known["upper"] <= known["lower"])
        ).sum()
    )
    note("error" if bad_bounds else "ok", "unusable intervals", bad_bounds)

    if len(known):
        # A genuine bid increment at the top of the market is narrow in
        # log terms: 25 lakh on top of 2500 is a log-width of 0.010. So
        # 0.01 flags real rows, and only widths an order of magnitude
        # below that indicate a fabricated interval.
        widths = np.log(known["upper"] / known["lower"])

        razor = int((widths < 1e-3).sum())
        note(
            "error" if razor else "ok",
            "fabricated-width intervals",
            razor,
            "log-width < 0.001; no real bid increment is this small",
        )

        tight = int(((widths >= 1e-3) & (widths < 2e-2)).sum())
        note(
            "info",
            "tight but plausible intervals",
            tight,
            "log-width < 0.02; expected at the top of the market",
        )

    # ------------------------------------------------------------------
    # Team state
    # ------------------------------------------------------------------

    if "remaining_purse" in training_df.columns:
        purse = training_df["remaining_purse"]
        note(
            "error" if (purse < 0).any() else "ok",
            "negative remaining_purse rows",
            int((purse < 0).sum()),
            f"min {purse.min():.0f}, max {purse.max():.0f}",
        )
        note(
            "error" if purse.isna().any() else "ok",
            "NaN remaining_purse rows",
            int(purse.isna().sum()),
            "a single NaN sale price poisons every later row",
        )

    # ------------------------------------------------------------------
    # Player features
    # ------------------------------------------------------------------

    player_cols = training_df.attrs.get("player_feature_columns", [])

    ##################################################################
    # "No feature row at all" means the ball-by-ball career came back
    # empty, i.e. identity resolution missed the player. It has to be
    # measured on the career columns ONLY.
    #
    # add_player_context_features now puts ctx_basePrice /
    # ctx_cappedStatus / ctx_isPlayerOverseas in this same block, and
    # those come off the auction row, so they are populated for every
    # player including the unresolved ones. Left as-is,
    # `block.isna().all(axis=1)` can no longer be True for ANY row and
    # this check silently reports 0 unresolved players forever --
    # which is the same number it reports when identity resolution is
    # perfect.
    ##################################################################

    career_cols = [c for c in player_cols if not c.startswith("ctx_")]

    if career_cols:
        block = training_df[career_cols]
        all_missing = block.isna().all(axis=1)

        unresolved = training_df.loc[all_missing, "playerId"].nunique()
        note(
            "warn" if unresolved else "ok",
            "players with no feature row at all",
            unresolved,
            f"{all_missing.mean():.1%} of rows; unresolved identity",
        )

        # Do the unresolved players skew expensive?  A few uncapped
        # teenagers with no career is expected and harmless.  A
        # marquee overseas signing with no career means the identity
        # map missed someone who matters.
        if unresolved:
            costly = (
                training_df.loc[
                    all_missing & training_df["winner"].fillna(False),
                    ["playerName", "auctionPrice"],
                ]
                .drop_duplicates()
                .nlargest(5, "auctionPrice")
            )
            if len(costly) and costly["auctionPrice"].max() > 200:
                note(
                    "error",
                    "EXPENSIVE players with no features",
                    len(costly),
                    "; ".join(
                        f"{r.playerName} @ {r.auctionPrice:.0f}"
                        for r in costly.itertuples()
                    ),
                )

        if "has_history" in training_df.columns:
            no_history = training_df.loc[
                training_df["has_history"] == 0, "playerId"
            ].nunique()
            note(
                "info",
                "players with zero career history",
                no_history,
                "uncapped debutants; expected to be non-zero",
            )

    if engine_report is not None:

        # Counts should be zero; the order fields are descriptive
        # strings whose severity depends on which value they took.
        clean_values = {
            0,
            "recorded",
            "column_ascending",
            "column_descending",
            "column_ascending_unverified",
            "column_descending_unverified",
            "reversed_file_order",
            None,
        }

        for key, value in engine_report.items():

            if key == "auction_order_warning":
                note("warn" if value else "ok", "engine: order warning",
                     1 if value else 0, value or "")
                continue

            if key == "auction_order_method":
                note(
                    "ok"
                    if value
                    and value.startswith("column")
                    and not value.endswith("_unverified")
                    else "warn",
                    "engine: auction order",
                    value,
                    "inferred from the purse test rather than a column"
                    if value and not value.startswith("column")
                    else (
                        "column used, but no ordering clears the purse "
                        "test -- suspect the prices, not the order"
                        if value and value.endswith("_unverified")
                        else ""
                    ),
                )
                continue

            if key == "matched_at_top":
                note("info", "engine: RTM top bidders right-censored",
                     value, "matched rather than outbid; recovered")
                continue

            severity = "ok" if value in clean_values else (
                "warn" if isinstance(value, str) else "error"
            )
            note(severity, f"engine: {key}", value)

    frame = pd.DataFrame(findings)

    if verbose:
        interesting = frame[frame["severity"].isin(["error", "warn"])]
        if len(interesting):
            print(f"  [verify {year}] {len(interesting)} finding(s):")
            for row in interesting.itertuples():
                print(
                    f"    {row.severity.upper():5s} {row.check}: "
                    f"{row.value} {row.detail}"
                )
        else:
            print(f"  [verify {year}] clean")

    return frame


def verify_feature_monotonicity(training_dfs, verbose=True):
    """
    Cross-year check on as-of correctness.

    Career totals are cumulative and strictly non-decreasing in time.
    So for one player, `bat_runs` as of the 2020 auction can never be
    smaller than `bat_runs` as of the 2019 auction.  If it is, the
    aggregator's as-of filter is wrong for at least one of those dates
    -- which is the shape a leak takes here: a later auction's features
    quietly computed over a window that excludes matches it should
    contain, or an earlier one including matches it should not.

    This is the one check that can catch as-of leakage without a second
    source of truth, because it only needs the pipeline's own output
    from two different dates.

    training_dfs : {year: training_df}
    """

    years = sorted(training_dfs)

    cumulative = [
        "bat_runs", "bat_balls", "bat_fours", "bat_sixes", "bat_outs",
        "bowl_balls", "bowl_runs", "bowl_wickets",
        "exp_matches", "exp_batting_innings", "exp_bowling_innings",
    ]

    first = training_dfs[years[0]]
    cols = [c for c in cumulative if c in first.columns]

    if not cols:
        if verbose:
            print("  [verify] no cumulative columns found; skipping")
        return pd.DataFrame()

    per_year = {}
    for year in years:
        df = training_dfs[year]
        per_year[year] = (
            df.drop_duplicates("playerId")
            .set_index("playerId")[cols]
        )

    violations = []

    for earlier, later in zip(years, years[1:]):
        a, b = per_year[earlier], per_year[later]
        shared = a.index.intersection(b.index)

        if not len(shared):
            continue

        # A tiny tolerance absorbs float noise, not real regressions.
        went_backwards = (b.loc[shared] < a.loc[shared] - 1e-6)

        for col in cols:
            n = int(went_backwards[col].sum())
            if n:
                worst = (
                    (b.loc[shared, col] - a.loc[shared, col])
                    .nsmallest(3)
                    .to_dict()
                )
                violations.append(
                    {
                        "from_year": earlier,
                        "to_year": later,
                        "column": col,
                        "players": n,
                        "of_shared": len(shared),
                        "worst_deltas": worst,
                    }
                )

    frame = pd.DataFrame(violations)

    if verbose:
        if len(frame):
            print(
                f"  [verify] AS-OF LEAK: {len(frame)} (year-pair, column) "
                f"combinations where a cumulative career total DECREASED "
                f"between auctions:"
            )
            print(frame.to_string(index=False))
        else:
            print(
                "  [verify] as-of monotonicity clean across "
                f"{len(years)} years"
            )

    return frame


def describe_player(training_df, name):
    """
    Spot-check one player's features as the pipeline built them.

    The cheapest way to catch an identity mismatch is to look at a
    cricketer you know and ask whether the career attached to him is
    his.  `identity: 715/808 resolved` says nothing about whether the
    715 are resolved to the RIGHT people.
    """

    rows = training_df[
        training_df["playerName"].str.contains(name, case=False, na=False)
    ]

    if rows.empty:
        return f"no player matching {name!r}"

    row = rows.iloc[0]

    interesting = [
        "playerId", "playerName", "role", "country", "basePrice",
        "auctionPrice", "auctionStatus", "playsForTeam",
        "has_history", "exp_matches",
        "bat_runs", "bat_balls", "bat_strike_rate", "bat_average",
        "bowl_balls", "bowl_wickets", "bowl_economy",
    ]

    return row[[c for c in interesting if c in row.index]]