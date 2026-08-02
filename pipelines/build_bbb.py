"""
Build a ball-by-ball dataset for men's T20 cricket from Cricsheet JSON.

Cricsheet JSON format reference: https://cricsheet.org/format/json/

Why this exists
---------------
The previous parquet keyed everything on player *name strings*, which are both
fragmented (one player appears as "KL Rahul" and "Lokesh Rahul") and collided
(two different players share "Amit Mishra").  Cricsheet ships a per-match
`info.registry.people` block mapping every name used in that file to a stable
8-character person id, so every row produced here carries an id and the name is
only ever a display label.

Outputs (parquet, written to --out-dir)
--------------------------------------
deliveries.parquet   one row per delivery
matches.parquet      one row per match
people.parquet       person_id -> canonical name + all name variants seen
wickets.parquet      one row per wicket (a delivery can carry more than one)
fielding.parquet     one row per (wicket, fielder) pair

Usage
-----
Sources and destination are resolved through `data_sources` by default, so on
Kaggle with the cricsheet dataset attached this is the whole command:

    python -m pipelines.build_bbb

which reads the 20 Cricsheet zips out of udvasbasak2/ipl-auction-model-inputs
and writes the five parquet files to /kaggle/working/bbb.  `--sources`
overrides the inputs (zips, directories of json, or individual json files) and
`--out-dir` the destination.

This is expected to run once per session and takes a few minutes.  The output
is deliberately not stored as a dataset: it is fully derived, and a stored
copy is one more thing that can drift from the zips it came from.

`--download` still fetches from cricsheet.org directly, which is how the
dataset is refreshed -- but it needs internet enabled on the notebook, and the
result should be uploaded as a dataset version rather than left in the
session.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cricsheet download URLs.  Men's T20 only.
# "The Hundred" (hnd) is deliberately excluded from the default set: it uses
# 5-ball overs, so its balls are not comparable and its over-based phase
# definitions do not apply.  Add it explicitly if you want it.
# ---------------------------------------------------------------------------

CRICSHEET_BASE = "https://cricsheet.org/downloads"

DEFAULT_SOURCES = {
    # internationals
    "t20s_male": "T20 internationals",
    "it20s_male": "Non-official T20 internationals",
    # franchise / domestic T20
    "ipl_male": "Indian Premier League",
    "bbl_male": "Big Bash League",
    "ntb_male": "T20 Blast",
    "sma_male": "Syed Mushtaq Ali Trophy",
    "bpl_male": "Bangladesh Premier League",
    "cpl_male": "Caribbean Premier League",
    "psl_male": "Pakistan Super League",
    "ctc_male": "CSA T20 Challenge",
    "ssm_male": "Super Smash",
    "lpl_male": "Lanka Premier League",
    "ilt_male": "International League T20",
    "sat_male": "SA20",
    "mlc_male": "Major League Cricket",
    "mlt_male": "Major League Tournament",
    "msl_male": "Mzansi Super League",
    "npl_male": "Nepal Premier League",
    "ipt_male": "Cricket Ireland Inter-Provincial Twenty20 Trophy",
    "mct_male": "Major Clubs T20 Tournament",
}

OPTIONAL_SOURCES = {
    "hnd_male": "The Hundred (5-ball overs -- excluded by default)",
}

PEOPLE_CSV_URL = "https://cricsheet.org/register/people.csv"


# ---------------------------------------------------------------------------
# Dismissal semantics.  These are the only places cricket rules enter the code,
# so they are constants rather than inline literals.
#
# Cricsheet `wickets[].kind` vocabulary (per the format docs):
#   bowled, caught, caught and bowled, lbw, stumped, run out, retired hurt,
#   hit wicket, obstructing the field, hit the ball twice, handled the ball,
#   timed out
# "retired out" and "retired not out" are not in that list but are accepted
# here defensively in case the vocabulary is extended.
# ---------------------------------------------------------------------------

# Dismissals credited to the bowler.
BOWLER_CREDITED_KINDS = frozenset({
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
})

# Dismissals that do NOT count as an "out" for the batter's average.
# A retirement that is not a "retired out" leaves the batter not out.
NOT_AN_OUT_KINDS = frozenset({
    "retired hurt",
    "retired not out",
})

# T20 phase boundaries, on the 0-based over index.
PHASE_POWERPLAY = range(0, 6)    # overs 1-6
PHASE_MIDDLE = range(6, 15)      # overs 7-15
# everything from over index 15 onwards is death


def _phase(over_idx):
    if over_idx in PHASE_POWERPLAY:
        return "powerplay"
    if over_idx in PHASE_MIDDLE:
        return "middle"
    return "death"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def download_sources(raw_dir, sources=None, include_optional=False):
    """Fetch the Cricsheet zips.  Requires `requests`."""
    import requests

    sources = dict(sources or DEFAULT_SOURCES)
    if include_optional:
        sources.update(OPTIONAL_SOURCES)

    os.makedirs(raw_dir, exist_ok=True)
    paths = []
    for code, label in tqdm(sources.items()):
        url = f"{CRICSHEET_BASE}/{code}_json.zip"
        dest = os.path.join(raw_dir, f"{code}_json.zip")
        if os.path.exists(dest):
            print(f"  cached  {code:14s} {label}")
            paths.append(dest)
            continue
        print(f"  fetch   {code:14s} {label}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(r.content)
        paths.append(dest)

    # The register is optional -- the per-match registry is enough for identity.
    # people.csv only adds cross-site ids (Cricinfo, CricketArchive, ...).
    try:
        r = requests.get(PEOPLE_CSV_URL, timeout=120)
        r.raise_for_status()
        with open(os.path.join(raw_dir, "people.csv"), "wb") as fh:
            fh.write(r.content)
        print("  fetch   people.csv (Cricsheet register)")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not fetch people.csv ({exc}); continuing without it")

    return paths


def iter_match_documents(paths):
    """Yield (match_id, parsed_json) from a mixture of zips, dirs and files."""
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith(".json") and name != "README.json":
                    with open(os.path.join(path, name)) as fh:
                        yield os.path.splitext(name)[0], json.load(fh)
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                for name in sorted(zf.namelist()):
                    if not name.endswith(".json") or name.endswith("README.json"):
                        continue
                    with zf.open(name) as fh:
                        doc = json.load(io.TextIOWrapper(fh, "utf-8"))
                    yield os.path.splitext(os.path.basename(name))[0], doc
        elif path.endswith(".json"):
            with open(path) as fh:
                yield os.path.splitext(os.path.basename(path))[0], json.load(fh)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def match_passes_filter(info, *, genders, match_types, balls_per_over, overs):
    if genders and info.get("gender") not in genders:
        return False
    if match_types and info.get("match_type") not in match_types:
        return False
    if balls_per_over and info.get("balls_per_over") not in balls_per_over:
        return False
    if overs and info.get("overs") not in overs:
        return False
    if not info.get("dates"):
        return False
    return True


def _competition(info):
    """A single stable competition label per match."""
    event = info.get("event") or {}
    name = event.get("name")
    if name:
        return name
    mt = info.get("match_type")
    if mt in ("T20", "IT20"):
        return "T20 International" if mt == "T20" else "Other T20 International"
    return mt or "unknown"


def parse_match(match_id, doc):
    """
    Return (match_row, delivery_rows, wicket_rows, fielding_rows, registry_pairs).

    Raises ValueError on structurally broken input rather than guessing.
    """
    info = doc["info"]
    registry = (info.get("registry") or {}).get("people") or {}
    if not registry:
        raise ValueError(f"{match_id}: no registry.people block")

    def pid(name):
        """Name -> person id.  Missing names are a data error, not a fallback."""
        if name is None:
            return None
        try:
            return registry[name]
        except KeyError as exc:
            raise ValueError(
                f"{match_id}: '{name}' appears in the deliveries but not in "
                f"registry.people"
            ) from exc

    teams = info["teams"]
    match_date = pd.Timestamp(sorted(info["dates"])[0])

    match_row = {
        "match_id": match_id,
        "match_date": match_date,
        "season": str(info.get("season", "")),
        "competition": _competition(info),
        "event_name": (info.get("event") or {}).get("name"),
        "match_type": info["match_type"],
        "team_type": info.get("team_type"),
        "gender": info["gender"],
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team_a": teams[0],
        "team_b": teams[1] if len(teams) > 1 else None,
        "toss_winner": (info.get("toss") or {}).get("winner"),
        "toss_decision": (info.get("toss") or {}).get("decision"),
        "winner": (info.get("outcome") or {}).get("winner"),
        "outcome_result": (info.get("outcome") or {}).get("result"),
        "balls_per_over": info.get("balls_per_over"),
        "overs": info.get("overs"),
        "data_version": (doc.get("meta") or {}).get("data_version"),
    }

    deliveries, wickets, fielding = [], [], []
    registry_pairs = [(v, k) for k, v in registry.items()]

    for inns_idx, inns in enumerate(doc.get("innings") or [], start=1):
        batting_team = inns["team"]
        bowling_team = next((t for t in teams if t != batting_team), None)
        is_super_over = bool(inns.get("super_over", False))
        target = (inns.get("target") or {}).get("runs")

        # Running innings state, captured *before* each delivery is applied.
        runs_before = 0
        wkts_before = 0
        legal_before = 0
        ball_seq = 0

        for over in inns.get("overs") or []:
            over_idx = over["over"]
            for ball_in_over, d in enumerate(over.get("deliveries") or [], start=1):
                extras = d.get("extras") or {}
                runs = d["runs"]

                wides = int(extras.get("wides", 0))
                noballs = int(extras.get("noballs", 0))
                byes = int(extras.get("byes", 0))
                legbyes = int(extras.get("legbyes", 0))
                penalty = int(extras.get("penalty", 0))

                is_wide = wides > 0
                is_noball = noballs > 0

                runs_batter = int(runs["batter"])
                runs_total = int(runs["total"])
                non_boundary = bool(runs.get("non_boundary", False))

                # A wide is not a ball faced by the batter and does not count
                # towards the bowler's over.  A no-ball IS faced by the batter
                # but does NOT count towards the bowler's over.
                ball_faced = not is_wide
                legal_ball = not (is_wide or is_noball)

                # Runs charged to the bowler: off the bat, plus wides and
                # no-balls.  Byes and leg-byes are the keeper's/fielders'
                # concern and are never charged to the bowler.
                runs_conceded = runs_batter + wides + noballs

                # Wickets on this delivery.
                kinds = []
                batter_dismissed = False
                bowler_credited = False
                primary_kind = None
                primary_out = None

                for w in d.get("wickets") or []:
                    kind = str(w["kind"]).strip().lower()
                    player_out = w["player_out"]
                    kinds.append(kind)

                    credited = kind in BOWLER_CREDITED_KINDS
                    is_out = kind not in NOT_AN_OUT_KINDS

                    wickets.append({
                        "match_id": match_id,
                        "innings": inns_idx,
                        "ball_seq": ball_seq,
                        "over": over_idx,
                        "kind": kind,
                        "player_out": player_out,
                        "player_out_id": pid(player_out),
                        "bowler": d["bowler"],
                        "bowler_id": pid(d["bowler"]),
                        "bowler_credited": credited,
                        "counts_as_out": is_out,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                    })

                    for f in (w.get("fielders") or []):
                        fname = f.get("name")
                        fielding.append({
                            "match_id": match_id,
                            "innings": inns_idx,
                            "ball_seq": ball_seq,
                            "kind": kind,
                            "fielder": fname,
                            # substitutes have no registry entry sometimes
                            "fielder_id": registry.get(fname),
                            "is_substitute": bool(f.get("substitute", False)),
                            "player_out_id": pid(player_out),
                        })

                    if credited:
                        bowler_credited = True
                    if player_out == d["batter"] and is_out:
                        batter_dismissed = True
                    # Prefer a real dismissal over a co-occurring retirement
                    # when picking the single kind shown on the delivery row.
                    if primary_kind is None or (
                        primary_kind in NOT_AN_OUT_KINDS and kind not in NOT_AN_OUT_KINDS
                    ):
                        primary_kind = kind
                        primary_out = player_out

                deliveries.append({
                    "match_id": match_id,
                    "match_date": match_date,
                    "competition": match_row["competition"],
                    "match_type": match_row["match_type"],
                    "season": match_row["season"],
                    "venue": match_row["venue"],

                    "innings": inns_idx,
                    "is_super_over": is_super_over,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "target": target,

                    "over": over_idx,
                    "ball_in_over": ball_in_over,
                    "ball_seq": ball_seq,
                    "phase": _phase(over_idx),

                    "striker": d["batter"],
                    "striker_id": pid(d["batter"]),
                    "non_striker": d["non_striker"],
                    "non_striker_id": pid(d["non_striker"]),
                    "bowler": d["bowler"],
                    "bowler_id": pid(d["bowler"]),

                    "runs_batter": runs_batter,
                    "runs_extras": int(runs.get("extras", 0)),
                    "runs_total": runs_total,
                    "runs_conceded_bowler": runs_conceded,
                    "non_boundary": non_boundary,

                    "wides": wides,
                    "noballs": noballs,
                    "byes": byes,
                    "legbyes": legbyes,
                    "penalty": penalty,
                    "is_wide": is_wide,
                    "is_noball": is_noball,

                    "ball_faced": ball_faced,
                    "legal_ball": legal_ball,

                    # A four/six only counts as a boundary when it actually was
                    # one -- `non_boundary` marks all-run and overthrow 4s/6s.
                    "is_four": runs_batter == 4 and not non_boundary,
                    "is_six": runs_batter == 6 and not non_boundary,
                    "is_boundary": runs_batter in (4, 6) and not non_boundary,
                    # A dot for the batter: a ball faced off which no run was
                    # scored off the bat.  Byes/leg-byes off the ball do not
                    # make it a scoring shot.
                    "is_dot_batter": ball_faced and runs_batter == 0,

                    "is_wicket": bool(kinds),
                    "wicket_kind": primary_kind,
                    "player_out": primary_out,
                    "player_out_id": pid(primary_out) if primary_out else None,
                    # Retirements that leave the batter not out must not count
                    # towards a batting average.  Carried on the delivery row so
                    # downstream aggregation needs this frame alone.
                    "player_out_counts": bool(
                        primary_kind is not None and primary_kind not in NOT_AN_OUT_KINDS
                    ),
                    "bowler_credited": bowler_credited,
                    "batter_dismissed": batter_dismissed,

                    "runs_before": runs_before,
                    "wickets_before": wkts_before,
                    "legal_balls_before": legal_before,
                })

                runs_before += runs_total
                wkts_before += sum(
                    1 for k in kinds if k not in NOT_AN_OUT_KINDS
                )
                legal_before += int(legal_ball)
                ball_seq += 1

    if not deliveries:
        raise ValueError(f"{match_id}: no deliveries")

    return match_row, deliveries, wickets, fielding, registry_pairs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(
    paths,
    out_dir,
    *,
    genders=("male",),
    match_types=("T20", "IT20"),
    balls_per_over=(6,),
    overs=(20,),
    drop_super_overs=True,
    register_path=None,
    verbose=True,
):
    os.makedirs(out_dir, exist_ok=True)

    matches, deliveries, wickets, fielding = [], [], [], []
    name_variants = defaultdict(set)
    skipped_filter = 0
    errors = []

    for i, (match_id, doc) in enumerate(iter_match_documents(paths)):
        info = doc.get("info") or {}
        if not match_passes_filter(
            info,
            genders=genders,
            match_types=match_types,
            balls_per_over=balls_per_over,
            overs=overs,
        ):
            skipped_filter += 1
            continue
        try:
            m, d, w, f, reg = parse_match(match_id, doc)
        except (ValueError, KeyError) as exc:
            errors.append(str(exc))
            continue

        matches.append(m)
        deliveries.extend(d)
        wickets.extend(w)
        fielding.extend(f)
        for person_id, name in reg:
            name_variants[person_id].add(name)

        if verbose and (i + 1) % 2000 == 0:
            print(f"  ... {i + 1} files scanned, {len(matches)} matches kept")

    if not matches:
        raise SystemExit("no matches survived the filters -- check your inputs")

    matches_df = pd.DataFrame(matches)
    deliveries_df = pd.DataFrame(deliveries)
    wickets_df = pd.DataFrame(wickets)
    fielding_df = pd.DataFrame(fielding) if fielding else pd.DataFrame(
        columns=["match_id", "innings", "ball_seq", "kind", "fielder",
                 "fielder_id", "is_substitute", "player_out_id"]
    )

    if drop_super_overs:
        n = int(deliveries_df["is_super_over"].sum())
        if n:
            # wickets and fielding are keyed on (match_id, innings) and must be
            # filtered by the same rule. Otherwise they keep rows pointing at
            # innings that no longer exist in deliveries, and the downstream
            # join on (match_id, innings) produces nulls that
            # `astype(np.int32)` turns into INT_MIN rather than raising.
            super_keys = set(
                map(
                    tuple,
                    deliveries_df.loc[
                        deliveries_df["is_super_over"], ["match_id", "innings"]
                    ].drop_duplicates().to_numpy(),
                )
            )
            keep = ~deliveries_df["is_super_over"]
            deliveries_df = deliveries_df.loc[keep]

            def _not_super(df):
                if len(df) == 0:
                    return df
                keys = map(tuple, df[["match_id", "innings"]].to_numpy())
                return df.loc[[k not in super_keys for k in keys]]

            n_w, n_f = len(wickets_df), len(fielding_df)
            wickets_df = _not_super(wickets_df).reset_index(drop=True)
            fielding_df = _not_super(fielding_df).reset_index(drop=True)
            if verbose:
                print(f"  dropped {n} super-over deliveries, "
                      f"{n_w - len(wickets_df)} wickets, "
                      f"{n_f - len(fielding_df)} fielding rows")

    # Canonical name = the longest variant seen, which is reliably the
    # full-name form rather than an initials form ("Lokesh Rahul" over
    # "KL Rahul" is arbitrary but consistent; the id is what matters).
    people_df = pd.DataFrame(
        [
            {
                "person_id": pid_,
                "canonical_name": max(sorted(names), key=len),
                "name_variants": "|".join(sorted(names)),
                "n_variants": len(names),
            }
            for pid_, names in name_variants.items()
        ]
    ).sort_values("person_id").reset_index(drop=True)

    people_df = _merge_register(people_df, register_path, verbose=verbose)

    # Deterministic ordering; every downstream as-of computation relies on it.
    deliveries_df = deliveries_df.sort_values(
        ["match_date", "match_id", "innings", "ball_seq"], kind="mergesort"
    ).reset_index(drop=True)
    matches_df = matches_df.sort_values(
        ["match_date", "match_id"], kind="mergesort"
    ).reset_index(drop=True)

    _sanity_check(deliveries_df, matches_df)

    deliveries_df.to_parquet(os.path.join(out_dir, "deliveries.parquet"), index=False)
    matches_df.to_parquet(os.path.join(out_dir, "matches.parquet"), index=False)
    people_df.to_parquet(os.path.join(out_dir, "people.parquet"), index=False)
    wickets_df.to_parquet(os.path.join(out_dir, "wickets.parquet"), index=False)
    fielding_df.to_parquet(os.path.join(out_dir, "fielding.parquet"), index=False)

    if verbose:
        print()
        print(f"  matches        {len(matches_df):>10,}")
        print(f"  deliveries     {len(deliveries_df):>10,}")
        print(f"  wickets        {len(wickets_df):>10,}")
        print(f"  people         {len(people_df):>10,}")
        print(f"  skipped (filter) {skipped_filter:>8,}")
        print(f"  errors         {len(errors):>10,}")
        for e in errors[:10]:
            print(f"     ! {e}")
        multi = people_df[people_df["n_variants"] > 1]
        print(f"  people with >1 name variant: {len(multi):,} "
              f"(these are exactly the cases a name-keyed join would have split)")

    return {
        "matches": matches_df,
        "deliveries": deliveries_df,
        "people": people_df,
        "wickets": wickets_df,
        "fielding": fielding_df,
        "errors": errors,
    }


def _merge_register(people_df, register_path, verbose=True):
    """
    Fold the Cricsheet register (people.csv) into the people table.

    The per-match registry gives one name per person -- whatever form Cricsheet
    writes in the scorecards -- so `name_variants` was effectively always a
    single entry, and any downstream matching that relied on "try every
    variant" had nothing to try.  The register adds two things the registry
    cannot:

      * `unique_name`, which disambiguates the five distinct Rashid Khans into
        "Rashid Khan", "Rashid Khan (2)", ... instead of five identical labels;
      * `key_cricinfo`, a stable external id for ~99% of people, which is the
        only sane anchor for resolving an auction roster against this table.

    Missing or unreadable register: warn and carry on with registry-only names,
    so a build never fails because of an optional file.
    """
    keep = ["identifier", "name", "unique_name", "key_cricinfo", "key_cricbuzz"]

    if not register_path or not os.path.exists(register_path):
        if verbose:
            print(f"  WARNING: no register at {register_path!r}; "
                  f"people.parquet will carry registry names only")
        for col in ("unique_name", "key_cricinfo", "key_cricbuzz"):
            people_df[col] = None
        return people_df

    reg = pd.read_csv(register_path)
    missing = [c for c in keep if c not in reg.columns]
    if missing:
        raise ValueError(f"{register_path} is missing columns {missing}")
    reg = reg[keep].drop_duplicates("identifier")

    merged = people_df.merge(
        reg, how="left", left_on="person_id", right_on="identifier"
    ).drop(columns="identifier")

    # Fold the register's names in as additional variants.  Keep this a set
    # union rather than a replacement: the registry name is the one that
    # actually appears in the ball data and must not be lost.
    def _variants(row):
        names = set(str(row["name_variants"]).split("|"))
        for col in ("name", "unique_name"):
            v = row[col]
            if isinstance(v, str) and v.strip():
                names.add(v.strip())
        return sorted(n for n in names if n)

    all_names = merged.apply(_variants, axis=1)
    merged["name_variants"] = ["|".join(v) for v in all_names]
    merged["n_variants"] = [len(v) for v in all_names]
    # Prefer the register's disambiguated label when there is one.
    merged["canonical_name"] = [
        u if isinstance(u, str) and u.strip() else c
        for u, c in zip(merged["unique_name"], merged["canonical_name"])
    ]
    merged = merged.drop(columns="name")

    if verbose:
        matched = int(merged["key_cricinfo"].notna().sum())
        print(f"  register: matched {matched}/{len(merged)} people "
              f"({matched / len(merged):.1%}), "
              f"{int((merged['n_variants'] > 1).sum())} now have >1 name variant")
    return merged


def _sanity_check(d, m):
    """Fail loudly on anything that would silently corrupt downstream stats."""
    problems = []

    if d["match_date"].isna().any():
        problems.append(f"{int(d['match_date'].isna().sum())} deliveries with a null match_date")
    for col in ("striker_id", "bowler_id"):
        if d[col].isna().any():
            problems.append(f"{int(d[col].isna().sum())} deliveries with a null {col}")

    dup = d.duplicated(["match_id", "innings", "ball_seq"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (match_id, innings, ball_seq) rows")

    # A wide is never faced; a no-ball is always faced but never legal.
    if (d["is_wide"] & d["ball_faced"]).any():
        problems.append("wide marked as a ball faced")
    if (d["is_noball"] & d["legal_ball"]).any():
        problems.append("no-ball marked as a legal ball")

    # Runs must reconcile.
    lhs = d["runs_total"]
    rhs = d["runs_batter"] + d["wides"] + d["noballs"] + d["byes"] + d["legbyes"] + d["penalty"]
    bad = int((lhs != rhs).sum())
    if bad:
        problems.append(f"{bad} deliveries where runs_total != batter + all extras")

    if not d["match_date"].is_monotonic_increasing:
        problems.append("deliveries are not sorted by match_date")

    if problems:
        raise ValueError("build_bbb sanity check failed:\n  - " + "\n  - ".join(problems))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zips", "--sources", nargs="*", default=[], dest="zips",
                   help="zip files, directories of json, or individual json "
                        "files. Defaults to the 20 Cricsheet zips in the "
                        "inputs Kaggle dataset.")
    p.add_argument("--download", action="store_true",
                   help="download the default men's T20 sources from Cricsheet first")
    p.add_argument("--raw-dir", default=None,
                   help="where --download writes; defaults to "
                        "<output_dir>/cricsheet")
    p.add_argument("--out-dir", default=None,
                   help="defaults to /kaggle/working/bbb")
    p.add_argument("--include-hundred", action="store_true",
                   help="also include The Hundred (5-ball overs)")
    p.add_argument("--keep-super-overs", action="store_true")
    p.add_argument("--register", default=None,
                   help="Cricsheet people.csv; defaults to <raw-dir>/people.csv")
    args = p.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import data_sources as ds

    out_dir = args.out_dir or ds.output_dir("bbb")

    # raw_dir is resolved LAZILY, inside the --download branch below.
    # output_dir() creates the directory it returns, and calling it here
    # created an empty /kaggle/working/cricsheet on every run -- which then
    # shadowed the real dataset, because /kaggle/working is searched first.
    # An unconditional call to a function with a side effect, used by a branch
    # that usually does not run.

    paths = []
    for pattern in args.zips:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    if args.download:
        raw_dir = args.raw_dir or ds.output_dir("cricsheet_download")
        print(f"Downloading Cricsheet sources to {raw_dir}:")
        paths.extend(download_sources(raw_dir,
                                      include_optional=args.include_hundred))

    if not paths:
        # The default: read the zips straight out of the mounted inputs
        # dataset. iter_match_documents streams json out of a zip, so nothing
        # is extracted into the session first.
        paths = ds.cricsheet_sources()
        print(f"Reading {len(paths)} Cricsheet source(s) from "
              f"{ds.DATASETS['inputs']['slug']}.")

    if not paths:
        p.error("nothing to read: pass --sources and/or --download, or "
                "attach the inputs Kaggle dataset")

    print(f"\nBuilding from {len(paths)} source(s) -> {out_dir}\n")

    # The register is optional. Resolved through data_sources so it is found
    # in the inputs dataset, not next to a raw directory that no longer
    # exists in a data-free repo.
    register_path = args.register or ds.people_register()

    build(paths, out_dir,
          drop_super_overs=not args.keep_super_overs,
          register_path=register_path)
    print(f"\nWrote to {out_dir}.")
    print("Next: python -m pipelines.build_shot_attributes")
    return 0


if __name__ == "__main__":
    sys.exit(main())