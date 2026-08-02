"""
Explain identity conflicts and propose resolution-cache rows for them.

A conflict is one auction `playerId` that resolved to two different Cricsheet
people across years -- almost always because the roster spells the name
differently from one season to the next ("Shivam Dubey" in 2018, "Shivam Dube"
from 2019), and a name tier answers confidently but wrongly for one spelling.

`PlayerFeatureContext.register_rosters` deliberately refuses to majority-vote
these: disagreement is evidence that at least one match is wrong, and a wrong
match silently welds one cricketer's career onto another.

The adjudication itself is usually trivial once you see the evidence, because
the auction roster records which franchise the player was at each year and the
ball data records who actually turned out for that franchise. In every conflict
observed so far the losing candidate had *zero* IPL appearances -- it was a
pure name collision. This script lays that side by side and writes the rows you
can append to the cache.

    python -m data.identity.explain_conflicts \
        --rosters "<completed_players dir>/completed_players_*.csv" \
        --out /kaggle/working/conflict_proposals.csv

Nothing is applied automatically. Read the evidence, then append the rows you
agree with into cricinfo_resolution.csv, then re-upload the
inputs Kaggle dataset.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

def _ds():
    """data_sources, with the repo root on sys.path first."""
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import data_sources
    return data_sources



from input_creation_2.auction_dataset_utils import PlayerFeatureContext


def _year_of(path):
    m = re.search(r"(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None


def _ipl_history(ipl, person_id):
    sub = ipl[
        (ipl["striker_id"] == person_id)
        | (ipl["non_striker_id"] == person_id)
        | (ipl["bowler_id"] == person_id)
    ]
    if len(sub) == 0:
        return 0, {}, (None, None)
    teams = pd.concat([
        sub.loc[sub["bowler_id"] == person_id, "bowling_team"],
        sub.loc[sub["striker_id"] == person_id, "batting_team"],
    ]).value_counts().to_dict()
    return len(sub), teams, (sub["season"].min(), sub["season"].max())


def explain(roster_glob=None, bbb_dir=None, resolution=None, out_path=None):
    ds = _ds()
    if roster_glob is None:
        roster_glob = ds.player_template().format(year="*")
    if bbb_dir is None:
        bbb_dir = ds.bbb_dir()
    if resolution is None:
        resolution = ds.resolution_path()
    if out_path is None:
        out_path = os.path.join(ds.output_dir(), "conflict_proposals.csv")

    paths = sorted(glob.glob(roster_glob))
    if not paths:
        raise SystemExit(f"no rosters matched {roster_glob!r}")
    rosters = {_year_of(p): pd.read_csv(p) for p in paths}

    ctx = PlayerFeatureContext(bbb_dir, resolution=resolution, verbose=False)
    ctx.register_rosters(rosters)

    if not ctx.identity_conflicts:
        print("no identity conflicts")
        return pd.DataFrame()

    people = pd.read_parquet(os.path.join(bbb_dir, "people.parquet")).set_index("person_id")
    deliveries = pd.read_parquet(
        os.path.join(bbb_dir, "deliveries.parquet"),
        columns=["competition", "season", "batting_team", "bowling_team",
                 "striker_id", "non_striker_id", "bowler_id"],
    )
    ipl = deliveries[deliveries["competition"] == "Indian Premier League"]

    every = pd.concat(
        [df.assign(_year=y) for y, df in rosters.items()], ignore_index=True
    )

    proposals = []
    for player_id, name, candidates in ctx.identity_conflicts:
        rows = every[every["playerId"] == player_id]
        franchises = (
            rows[["_year", "playerName", "playsForTeam"]]
            .drop_duplicates().sort_values("_year")
        )
        print(f"\n=== playerId {player_id}  {name!r}")
        print(franchises.to_string(index=False))

        scored = []
        for cand in candidates:
            balls, teams, span = _ipl_history(ipl, cand)
            label = people["canonical_name"].get(cand, "?")
            cricinfo = people["key_cricinfo"].get(cand)
            print(f"  {cand}  {label:<24} cricinfo={cricinfo}")
            if balls:
                print(f"      IPL {span[0]}-{span[1]}  {teams}")
            else:
                print("      NO IPL appearances")
            scored.append((balls, cand, label, cricinfo))

        # Propose only when exactly one candidate has any IPL record at all.
        # That is the pattern every observed conflict has taken; anything else
        # needs a human, so say so rather than inventing a tie-break.
        with_ipl = [s for s in scored if s[0] > 0]
        if len(with_ipl) == 1:
            _, cand, label, cricinfo = with_ipl[0]
            rivals = [s[2] for s in scored if s[1] != cand]
            proposals.append({
                "playerId": player_id,
                "playerName": name,
                "cricinfo_id": int(cricinfo) if pd.notna(cricinfo) else "",
                "dob": "",
                "method": "squad_corroborated",
                "note": (f"{label}; rivals with 0 IPL appearances: "
                         f"{', '.join(rivals)}"),
            })
            print(f"  -> PROPOSE {label} (cricinfo {cricinfo})")
        else:
            print("  -> NEEDS A HUMAN: more than one candidate has IPL history")

    out = pd.DataFrame(proposals)
    if out_path and len(out):
        out.to_csv(out_path, index=False)
        print(f"\n{len(out)} proposals written to {out_path}")
        print("Review, then append the ones you agree with to "
              "cricinfo_resolution.csv, then re-upload the inputs dataset")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rosters", default=None,
                   help="glob for completed_players_*.csv; defaults to "
                        "the auction Kaggle dataset")
    p.add_argument("--bbb-dir", default=None,
                   help="defaults to the inputs Kaggle dataset")
    p.add_argument("--resolution", default=None,
                   help="defaults to the inputs Kaggle dataset")
    p.add_argument("--out", default=None,
                   help="defaults to <output_dir>/conflict_proposals.csv")
    a = p.parse_args(argv)
    explain(a.rosters, a.bbb_dir, a.resolution, a.out)


if __name__ == "__main__":
    main()