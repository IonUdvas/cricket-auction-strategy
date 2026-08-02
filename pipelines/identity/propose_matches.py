"""
Propose resolution-cache rows for auction playerIds the resolver could not
settle, using squad residue rather than name cleverness.

The resolver is deliberately unwilling to guess between namesakes, so what it
leaves behind is of two kinds:

  ambiguous  -- several Cricsheet people are name-plausible and squad evidence
                did not separate them (both Pandya brothers were at Mumbai;
                both Rinku Singh and Ramandeep Singh were at Kolkata)
  unresolved -- no Cricsheet person is name-plausible at all, because the two
                sources spell the man completely differently.  Cricsheet files
                Varun Chakravarthy under "CV Varun", Wanindu Hasaranga under
                his actual surname "PWH de Silva", and Mujeeb Zadran as
                "Mujeeb Ur Rahman".  No string metric reaches those; they are
                not spelling variants, they are different names.

For both kinds the same fact settles it, and it is not a name.  The auction
roster says which franchise the player was at in which year.  The ball data
says who actually turned out for that franchise that year.  Subtract everyone
already claimed by another playerId and what remains is a short list -- often a
list of one.

    python -m data.identity.propose_matches \
        --rosters "<completed_players dir>/completed_players_*.csv" \
        --out /kaggle/working/match_proposals.csv

Nothing is applied automatically.  Read the evidence, then append the rows you
believe into cricinfo_resolution.csv, then re-upload the
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



from input_creation_2.player_features.identity import (
    PlayerIdentityResolver, name_signature, normalize_name, _is_subsequence,
)
from input_creation_2.player_features.squad_index import SquadIndex, FRANCHISE_NAMES


def _year_of(path):
    m = re.search(r"(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_rosters(roster_glob):
    paths = sorted(glob.glob(roster_glob))
    if not paths:
        raise SystemExit(f"no rosters matched {roster_glob!r}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["season_year"] = _year_of(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _squad_members(deliveries, season, franchise):
    """person_ids who appeared for `franchise` in `season`, from the ball data."""
    names = FRANCHISE_NAMES.get(str(franchise).strip().upper())
    if not names:
        return set()
    d = deliveries[
        (deliveries["competition"] == "Indian Premier League")
        & (deliveries["season"].str[:4] == str(season)[:4])
    ]
    bat = d[d["batting_team"].isin(names)]
    bowl = d[d["bowling_team"].isin(names)]
    return (set(bat["striker_id"].dropna())
            | set(bat["non_striker_id"].dropna())
            | set(bowl["bowler_id"].dropna()))


def _name_plausibility(a, b):
    """
    2 = shares a token that is not the surname, 1 = same surname with
    compatible initials, 0 = not plausible.

    Only ever used to *rank* an already-short squad-residue list, never to
    admit a candidate that squad evidence did not already put forward.

    The initials test on level 1 is load-bearing, not decoration.  Without it
    a shared surname alone was enough, and because an unresolved star leaves
    his own Cricsheet person unclaimed, the residue happily offered RG Sharma
    to "Shivalik Sharma" -- a different man, same surname, and the only
    Sharma left in the pile.  Requiring "s" and "rg" to be compatible kills
    that, while "d" and "djm" (D Arcy Short / DJM Short) still pass.
    """
    sa, sb = name_signature(a), name_signature(b)
    ta = set(normalize_name(a).split()) - {sa[0]}
    tb = set(normalize_name(b).split()) - {sb[0]}
    if ta & tb:
        return 2
    if sa[0] and sa[0] == sb[0]:
        x, y = sa[1], sb[1]
        if x and y and (x.startswith(y) or y.startswith(x)
                        or _is_subsequence(x, y) or _is_subsequence(y, x)):
            return 1
    return 0


def propose(roster_glob=None, bbb_dir=None, resolution=None, out_path=None,
            max_candidates=6):
    ds = _ds()
    if roster_glob is None:
        roster_glob = ds.player_template().format(year="*")
    if bbb_dir is None:
        bbb_dir = ds.bbb_dir()
    if resolution is None:
        resolution = ds.resolution_path()
    if out_path is None:
        out_path = os.path.join(ds.output_dir(), "match_proposals.csv")

    every = load_rosters(roster_glob)
    people = pd.read_parquet(os.path.join(bbb_dir, "people.parquet"))
    deliveries = pd.read_parquet(
        os.path.join(bbb_dir, "deliveries.parquet"),
        columns=["competition", "season", "batting_team", "bowling_team",
                 "striker_id", "non_striker_id", "bowler_id"],
    )

    resolver = PlayerIdentityResolver(
        people, resolution=resolution, squad_index=SquadIndex(deliveries)
    )
    resolved = resolver.resolve(every)

    reg = people.set_index("person_id")
    claimed = set(resolved["person_id"].dropna())

    stuck = resolved[resolved["person_id"].isna()]
    if stuck.empty:
        print("nothing unresolved")
        return pd.DataFrame()

    # Cache squad lookups: many playerIds share a (season, franchise) pair.
    squad_cache = {}

    def squad(season, team):
        key = (str(season)[:4], str(team).strip().upper())
        if key not in squad_cache:
            squad_cache[key] = _squad_members(deliveries, season, team)
        return squad_cache[key]

    proposals = []
    for player_id, grp in stuck.groupby("playerId", sort=False):
        names = list(dict.fromkeys(grp["playerName"]))
        label = names[0]
        pairs = sorted({(int(y), t) for y, t in
                        zip(grp["season_year"], grp["playsForTeam"])
                        if pd.notna(t)})

        # Everyone who was in one of this player's franchises in one of this
        # player's years, minus everyone another playerId already accounts for.
        residue = set()
        for year, team in pairs:
            residue |= squad(year, team)
        residue -= claimed

        ranked = sorted(
            residue,
            key=lambda c: (-max((_name_plausibility(n, reg["canonical_name"].get(c, ""))
                                 for n in names), default=0),
                           reg["canonical_name"].get(c, "")),
        )
        scored = [(c, max((_name_plausibility(n, reg["canonical_name"].get(c, ""))
                           for n in names), default=0)) for c in ranked]

        print(f"\n=== playerId {player_id}  {label!r}  [{grp['match_method'].iloc[0]}]")
        print(f"    rosters: {', '.join(f'{y} {t}' for y, t in pairs) or '(no team)'}")
        if not scored:
            print("    no unclaimed squad member -- this player may never have "
                  "played an IPL match, in which case has_history=0 is correct")
            continue

        for cand, score in scored[:max_candidates]:
            mark = {2: "  <-- shares a name token", 1: "  <- shares a surname"}.get(score, "")
            print(f"    {cand}  {reg['canonical_name'].get(cand, '?'):<26}"
                  f" cricinfo={reg['key_cricinfo'].get(cand)}{mark}")

        best = [c for c, sc in scored if sc == scored[0][1]]
        # Propose only when the squad residue leaves exactly one name-plausible
        # person.  Anything else is a judgement call and is printed, not written.
        if len(best) == 1 and scored[0][1] > 0:
            cand = best[0]
            cricinfo = reg["key_cricinfo"].get(cand)
            proposals.append({
                "playerId": player_id,
                "playerName": label,
                "cricinfo_id": int(cricinfo) if pd.notna(cricinfo) else "",
                "dob": "",
                "method": "squad_residue",
                "note": (f"{reg['canonical_name'].get(cand)}; only unclaimed "
                         f"{'/'.join(t for _, t in pairs)} squad member matching "
                         f"the roster name"),
            })
            print(f"    -> PROPOSE {reg['canonical_name'].get(cand)}")
        else:
            print("    -> NEEDS A HUMAN")

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
                   help="defaults to <output_dir>/match_proposals.csv")
    a = p.parse_args(argv)
    propose(a.rosters, a.bbb_dir, a.resolution, a.out)


if __name__ == "__main__":
    main()