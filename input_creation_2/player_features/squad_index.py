"""
Disambiguate auction identities by IPL squad membership.

Name matching alone cannot separate the five Rashid Khans, and no amount of
string cleverness will: they are genuinely different people with identical
names.  But the auction row carries a fact the ball data can check --
`playsForTeam` -- and only one of those five actually turned out for Sunrisers
Hyderabad in 2018.

So this is a *disambiguator*, never a matcher.  It is only ever handed a set of
candidates that already passed a name test, and it narrows them.  Used the
other way round -- surname match plus squad membership, with no name-plausibility
constraint -- it maps "Mehdi Hasan" onto "Shakib Al Hasan", because they share a
surname token and Shakib really was in that squad.  That failure is the reason
`PlayerIdentityResolver` only calls this with pre-filtered candidates.

Squad membership is taken from who actually appears in the deliveries, not from
a roster: a player who was bought but never took the field will not be found.
That is a deliberate false-negative -- it costs a resolution, whereas guessing
would cost correctness.
"""

from __future__ import annotations

import pandas as pd

# Auction abbreviations -> every Cricsheet name that franchise has used.
# Franchises rename (Kings XI Punjab -> Punjab Kings, RCB's Bangalore ->
# Bengaluru) and the ball data keeps the name that was current at the time, so
# each abbreviation has to map to a list rather than a string.
FRANCHISE_NAMES = {
    "MI":   ["Mumbai Indians"],
    "KKR":  ["Kolkata Knight Riders"],
    "KXIP": ["Kings XI Punjab", "Punjab Kings"],
    "PBKS": ["Punjab Kings", "Kings XI Punjab"],
    "RR":   ["Rajasthan Royals"],
    "SRH":  ["Sunrisers Hyderabad"],
    "CSK":  ["Chennai Super Kings"],
    "DD":   ["Delhi Daredevils", "Delhi Capitals"],
    "DC":   ["Delhi Capitals", "Delhi Daredevils"],
    "RCB":  ["Royal Challengers Bangalore", "Royal Challengers Bengaluru"],
    "GT":   ["Gujarat Titans"],
    "LSG":  ["Lucknow Super Giants"],
    "RPS":  ["Rising Pune Supergiant", "Rising Pune Supergiants"],
    "GL":   ["Gujarat Lions"],
    "KTK":  ["Kochi Tuskers Kerala"],
    "PWI":  ["Pune Warriors"],
    "DECCAN": ["Deccan Chargers"],
}


class SquadIndex:
    """(season, franchise, person_id) membership, built from the ball data."""

    def __init__(self, deliveries, competition="Indian Premier League"):
        need = {"competition", "season", "batting_team", "bowling_team",
                "striker_id", "non_striker_id", "bowler_id"}
        missing = need - set(deliveries.columns)
        if missing:
            raise ValueError(f"deliveries missing columns: {sorted(missing)}")

        d = deliveries[deliveries["competition"] == competition]
        if len(d) == 0:
            raise ValueError(f"no deliveries for competition={competition!r}")

        # A player counts as present if they batted, were at the non-striker's
        # end, or bowled -- the three roles a delivery records.  A pure
        # substitute fielder is invisible here, which is fine: they are also
        # invisible to anyone reading a scorecard.
        frames = [
            d[["season", "batting_team", "striker_id"]]
            .rename(columns={"batting_team": "team", "striker_id": "pid"}),
            d[["season", "batting_team", "non_striker_id"]]
            .rename(columns={"batting_team": "team", "non_striker_id": "pid"}),
            d[["season", "bowling_team", "bowler_id"]]
            .rename(columns={"bowling_team": "team", "bowler_id": "pid"}),
        ]
        squads = pd.concat(frames).dropna().drop_duplicates()

        # Cricsheet writes some seasons as "2007/08"; the auction knows only
        # the calendar year it was held for, so key on the leading four digits.
        self._members = set(
            zip(squads["season"].str[:4], squads["team"], squads["pid"])
        )

    def _keys(self, season, franchise):
        names = FRANCHISE_NAMES.get(str(franchise).strip().upper())
        if not names:
            return []
        return [(str(season)[:4], name) for name in names]

    def members(self, season, team):
        """
        Everyone who took the field for `team` in `season`.

        Used by identity triage rather than by resolution: if a
        playerId was listed with a franchise-season the ball data
        covers, and no candidate of his is in that squad, he almost
        certainly has no T20 record rather than a mis-spelled one.
        """
        out = set()
        for key in self._keys(season, team):
            out |= {
                pid for (yr, tm, pid) in self._members
                if (yr, tm) == key
            }
        return out

    def covers(self, season, team):
        """
        Whether the ball data has ANY record of this franchise-season.

        Distinguishes "he was not in that squad" from "that season has
        not been played yet".  The 2026 auction has no 2026 ball data,
        so an empty squad there means nothing at all -- treating it as
        evidence would mark every 2026 signing as having no T20 record.
        """
        return len(self.members(season, team)) > 0

    def narrow(self, candidates, season, franchise):
        """
        Restrict `candidates` to those who actually played for `franchise` in
        `season`.  Returns a set; empty means the squad evidence did not help,
        which is not the same as the candidates being wrong.
        """
        if not candidates or franchise is None or pd.isna(franchise):
            return set()
        names = FRANCHISE_NAMES.get(str(franchise).strip().upper())
        if not names:
            return set()
        season = str(season)[:4]
        return {
            c for c in candidates
            for name in names
            if (season, name, c) in self._members
        }