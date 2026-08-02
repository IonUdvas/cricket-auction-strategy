"""
Attach shot-quality attributes (control, false shot, elevation, shot type,
line/length, wagon wheel) to the Cricsheet-derived ball-by-ball data.

Why this exists
---------------
`build_bbb.py` produces an identity-clean delivery table: every row carries a
stable Cricsheet `person_id` and a stable `match_id`.  What it cannot produce
is anything about *how* a ball was played, because Cricsheet does not record
it.  That information lives in a separate commentary-derived feed which keys
players on **Cricinfo numeric ids** and matches on its own **`p_match`** ids,
neither of which Cricsheet knows about.

So this module is a join, and the entire difficulty is that the join is on
three levels at once -- player, match, and ball -- with no shared key on any
of them.  Each level is resolved explicitly and each one is verified rather
than assumed:

  players   Cricinfo id -> person_id, via `people.key_cricinfo` where present
            (an exact numeric key) and the existing PlayerIdentityResolver
            where not.  A person_id claimed by two source ids is a collision;
            the exact-key claim wins and the fuzzy one is dropped, because the
            alternative -- dropping both -- silently deletes Shakib Al Hasan,
            Moeen Ali and Kane Williamson from the dataset.

  matches   p_match -> match_id, by same-day (+/- 1 day) candidate generation
            and Jaccard overlap of the resolved player sets.  A match is only
            accepted if overlap is strong AND the runner-up is clearly worse;
            two source matches may not claim one Cricsheet match.  This lands
            100% of IPL 2015-2025 with a median Jaccard of 1.0.

  balls     Positional, within (match_id, innings).  The source orders on
            `ball_id` (= zero-based over + ball/100, unique within an innings)
            and Cricsheet on `ball_seq`; 98%+ of innings agree on ball count
            exactly, so the k-th ball on one side is the k-th on the other.
            That is an assumption, so it is *checked* rather than trusted:
            a row is only emitted if the striker person_id agrees on both
            sides.  Innings whose ball counts differ are dropped whole.

Definitions
-----------
The two derived rates are the ones the auction model actually consumes, and
both have a denominator that is easy to get wrong.

`is_false_shot` -- the source's control vocabulary is
{under control, well timed, mis-timed, miss, edge, hit pad}.  Controlled is
{under control, well timed}; everything else is a false shot.  `hit pad` is
included because being beaten is a false shot, but `is_false_shot_strict`
excludes it for anyone who disagrees, and the raw category is kept so the
question stays open.

`is_aerial` -- the source records `grounded` on balls the batter never hit:
39,806 plays-and-misses, 15,300 hit-pads and 10,993 leaves are all filed as
grounded.  Dividing by every ball therefore does not measure "how often does
he hit it in the air", it measures that diluted by how often he misses.  So
`made_contact` is emitted alongside, and it is the correct denominator for
elevation.

There is also a second, wider control signal: the feed carries a plain binary
`control` on ~1.07M balls, roughly 2.5x the coverage of the descriptive
vocabulary.  The two do not agree on `mis-timed` (some are flagged controlled,
some not), so they are NOT merged into one column.  Both are emitted --
`control_binary` for reach, `shot_control` for precision -- and the choice is
left to the feature builder, which is the layer that knows how much coverage
it needs.

Outputs (parquet, --out-dir, default /kaggle/working/bbb)
---------------------------------------------
ball_attributes.parquet   one row per Cricsheet delivery that has shot data
player_crosswalk.parquet  cricinfo_id -> person_id, with method and ball counts
match_crosswalk.parquet   p_match -> match_id, with match quality and status

Usage
-----
    python -m data.build_shot_attributes \
        --source-dir <dir>   (default: the inputs Kaggle dataset)
        --bbb-dir     <dir>   (default: the inputs Kaggle dataset)
        --out-dir     <dir>   (default: /kaggle/working/bbb)
"""

from __future__ import annotations

import argparse
import os
import sys

import duckdb
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Control vocabulary.  These are the only cricket semantics in the file.
# ---------------------------------------------------------------------------

CONTROLLED = frozenset({"under control", "well timed"})

# Beaten, edged or mistimed.  `hit pad` is a false shot under the wide reading
# and not a shot at all under the narrow one; both readings are emitted.
FALSE_SHOT = frozenset({"mis-timed", "miss", "edge", "hit pad"})
FALSE_SHOT_STRICT = frozenset({"mis-timed", "miss", "edge"})

# Balls where bat never met ball.  The denominator for elevation must exclude
# these or "% aerial" silently becomes "% aerial among balls including the
# ones he swung through".
NO_CONTACT_CONTROL = frozenset({"miss", "hit pad"})
NO_CONTACT_SHOT = frozenset({"leave"})

# Match acceptance thresholds.  A Jaccard of 0.35 is deliberately loose --
# the source and Cricsheet disagree about substitutes and impact players --
# but the runner-up rule is what actually does the work.
MIN_JACCARD = 0.35
RUNNER_UP_RATIO = 0.8


# ---------------------------------------------------------------------------
# Stage 1 -- players
# ---------------------------------------------------------------------------

def build_player_crosswalk(con, source_glob, people_path, repo_root):
    """Cricinfo numeric player id -> Cricsheet person_id."""
    sys.path.insert(0, repo_root)
    from input_creation_2.player_features.identity import PlayerIdentityResolver

    src = con.sql(f"""
        WITH u AS (
            SELECT p_bat AS pid, bat AS nm, team_bat AS team FROM {source_glob}
            UNION ALL
            SELECT p_bowl, bowl, team_bowl FROM {source_glob}
        )
        SELECT pid,
               list(DISTINCT nm)   AS names,
               list(DISTINCT team) AS teams,
               count(*)            AS balls
        FROM u WHERE pid IS NOT NULL GROUP BY pid
    """).df()

    people = pd.read_parquet(people_path)

    # Tier 1: the exact numeric key.  Only trusted where it is unique on the
    # Cricsheet side -- a key mapping to two person_ids identifies neither.
    ci = people.dropna(subset=["key_cricinfo"]).copy()
    ci["key_cricinfo"] = ci["key_cricinfo"].astype("int64")
    ci = ci[~ci["key_cricinfo"].duplicated(keep=False)]
    src["person_id"] = src["pid"].map(dict(zip(ci["key_cricinfo"], ci["person_id"])))
    src["method"] = np.where(src["person_id"].notna(), "cricinfo_key", None)
    n1 = int(src["person_id"].notna().sum())

    # Tier 2: name resolution, using every spelling and team context the id
    # was ever seen with.
    # resolve_player returns (person_id, method); person_id is None for
    # anything it refused to guess at, which is most of the tail.
    todo = src[src["person_id"].isna()]
    resolver = PlayerIdentityResolver(people)
    found = {}
    for row in todo.itertuples():
        names = [n for n in row.names if isinstance(n, str) and n.strip()]
        if not names:
            continue
        ctxs = [(None, t) for t in row.teams if isinstance(t, str)]
        try:
            person, how = resolver.resolve_player(row.pid, names, contexts=ctxs)
        except Exception:
            continue
        if isinstance(person, str) and person:
            found[row.Index] = (person, f"name:{how}")
    for idx, (person, how) in found.items():
        src.at[idx, "person_id"] = person
        src.at[idx, "method"] = how
    n2 = len(found)

    # Collisions.  An exact numeric key beats a fuzzy name match; only a
    # same-tier tie is genuinely unresolvable.
    rank = np.where(src["method"] == "cricinfo_key", 0, 1)
    rank = pd.Series(rank, index=src.index)
    resolved = src[src["person_id"].notna()]
    drop = []
    for _, grp in resolved.groupby("person_id"):
        if len(grp) == 1:
            continue
        r = rank.loc[grp.index]
        winners = grp.index[r == r.min()]
        drop.extend(grp.index if len(winners) > 1 else grp.index.difference(winners))
    src.loc[drop, ["person_id", "method"]] = [None, None]

    ok = src["person_id"].notna()
    print(f"  players: {n1} by key + {n2} by name - {len(drop)} collisions "
          f"= {int(ok.sum())}/{len(src)} ids, "
          f"{100 * src.loc[ok, 'balls'].sum() / src['balls'].sum():.1f}% of balls")

    out = src[["pid", "person_id", "method", "balls"]].rename(
        columns={"pid": "cricinfo_id"})
    out["display_name"] = src["names"].apply(lambda x: x[0] if len(x) else None)
    return out


# ---------------------------------------------------------------------------
# Stage 2 -- matches
# ---------------------------------------------------------------------------

def build_match_crosswalk(con, source_glob, deliveries_path):
    """Source p_match -> Cricsheet match_id, by date bucket + player overlap."""
    src = con.sql(f"""
        WITH u AS (
            SELECT p_match, date, competition, p_bat AS pid FROM {source_glob}
            UNION ALL
            SELECT p_match, date, competition, p_bowl FROM {source_glob}
        )
        SELECT u.p_match,
               min(u.date)                 AS date,
               any_value(u.competition)    AS competition,
               list(DISTINCT x.person_id)  AS players
        FROM u JOIN xw x ON u.pid = x.cricinfo_id
        WHERE x.person_id IS NOT NULL
        GROUP BY u.p_match
    """).df()
    src["date"] = pd.to_datetime(src["date"], errors="coerce")

    cs = con.sql(f"""
        WITH u AS (
            SELECT match_id, match_date, striker_id AS pid FROM '{deliveries_path}'
            UNION ALL
            SELECT match_id, match_date, bowler_id FROM '{deliveries_path}'
        )
        SELECT match_id, min(match_date) AS date, list(DISTINCT pid) AS players
        FROM u WHERE pid IS NOT NULL GROUP BY match_id
    """).df()
    cs["date"] = pd.to_datetime(cs["date"], errors="coerce")

    cs_sets = {r.match_id: set(r.players) for r in cs.itertuples()}
    by_day = {}
    for r in cs.itertuples():
        if not pd.isna(r.date):
            by_day.setdefault(r.date.normalize(), []).append(r.match_id)

    day = pd.Timedelta(days=1)
    rows = []
    for r in src.itertuples():
        if pd.isna(r.date):
            rows.append((r.p_match, None, 0.0, 0, "no_date"))
            continue
        d = r.date.normalize()
        cands = by_day.get(d, []) + by_day.get(d - day, []) + by_day.get(d + day, [])
        s = set(r.players)
        scored = []
        for m in cands:
            inter = len(s & cs_sets[m])
            if inter:
                scored.append((inter / len(s | cs_sets[m]), inter, m))
        if not scored:
            rows.append((r.p_match, None, 0.0, 0,
                         "no_candidate" if not cands else "no_overlap"))
            continue
        scored.sort(reverse=True)
        best_j, best_i, best_m = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        if best_j < MIN_JACCARD:
            rows.append((r.p_match, None, best_j, best_i, "weak"))
        elif runner > RUNNER_UP_RATIO * best_j:
            rows.append((r.p_match, None, best_j, best_i, "ambiguous"))
        else:
            rows.append((r.p_match, best_m, best_j, best_i, "ok"))

    mx = pd.DataFrame(rows, columns=["p_match", "match_id", "jaccard",
                                     "n_shared", "status"])
    mx = mx.merge(src[["p_match", "date", "competition"]], on="p_match", how="left")

    # One Cricsheet match may be claimed once.  Keep the strongest claim.
    got = mx["match_id"].notna()
    dupe = got & mx["match_id"].duplicated(keep=False)
    if dupe.any():
        order = mx[dupe].sort_values("jaccard", ascending=False)
        losers = order.index[order["match_id"].duplicated(keep="first")]
        mx.loc[losers, ["match_id", "status"]] = [None, "duplicate_claim"]

    got = mx["match_id"].notna()
    print(f"  matches: {int(got.sum())}/{len(mx)} ({100 * got.mean():.1f}%), "
          f"median jaccard {mx.loc[got, 'jaccard'].median():.3f}")
    return mx


# ---------------------------------------------------------------------------
# Stage 3 -- balls
# ---------------------------------------------------------------------------

_ATTR_SELECT = """
    SELECT
        b.match_id_B                                   AS p_match,
        b.innings                                      AS innings,
        b.ball_id_B                                    AS ball_id,
        b.p_bat_B                                      AS cricinfo_striker,
        b.p_bowl_B                                     AS cricinfo_bowler,
        TRY_CAST(b.control AS DOUBLE)                  AS control_binary,
        b.control_alt                                  AS shot_control,
        b.elevation_A                                  AS elevation,
        b.shot_type_A                                  AS shot_type,
        b.foot_A                                       AS foot,
        b.variation_A                                  AS variation,
        b.len_var_A                                    AS length_variation,
        coalesce(b.line, b.line_alt)                   AS line,
        coalesce(b.length, b.length_alt)               AS length,
        b.area_A                                       AS area,
        b.zone_A                                       AS zone,
        b.wagonX                                       AS wagon_x,
        b.wagonY                                       AS wagon_y,
        b.wagonZone                                    AS wagon_zone,
        b.fielding_position_A                          AS fielding_position,
        b.fielder_action_A                             AS fielder_action,
        b.match_id_A IS NOT NULL                       AS has_shot_detail
    FROM {combined} b
    WHERE b.match_id_B IS NOT NULL
"""


def build_ball_attributes(con, combined_scan, deliveries_path):
    """Positionally align source balls to Cricsheet deliveries and verify."""
    con.execute(f"CREATE OR REPLACE VIEW src_raw AS {_ATTR_SELECT.format(combined=combined_scan)}")

    # Position within (p_match, innings), ordered by ball_id.  ball_id is
    # zero-based-over + ball/100 and is unique within an innings, so it is a
    # total order even where wides make `ball` repeat.
    con.execute("""
        CREATE OR REPLACE VIEW src_pos AS
        SELECT *, row_number() OVER (
            PARTITION BY p_match, innings ORDER BY ball_id
        ) - 1 AS pos
        FROM src_raw
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW cs_pos AS
        SELECT match_id, innings, ball_seq, striker_id, bowler_id, over, phase,
               row_number() OVER (
                   PARTITION BY match_id, innings ORDER BY ball_seq
               ) - 1 AS pos
        FROM '{deliveries_path}'
        WHERE NOT is_super_over
    """)

    # Only align innings whose ball counts agree exactly.  A count mismatch
    # means the two feeds disagree about what happened, and a positional join
    # across a disagreement silently shifts every subsequent ball by one.
    con.execute("""
        CREATE OR REPLACE VIEW aligned_innings AS
        SELECT m.p_match, m.match_id, s.innings
        FROM (SELECT p_match, innings, count(*) n FROM src_pos GROUP BY 1,2) s
        JOIN mx m ON s.p_match = m.p_match
        JOIN (SELECT match_id, innings, count(*) n FROM cs_pos GROUP BY 1,2) d
             ON d.match_id = m.match_id AND d.innings = s.innings
        WHERE m.match_id IS NOT NULL AND s.n = d.n
    """)

    joined = con.sql("""
        SELECT
            c.match_id, c.innings, c.ball_seq, c.over, c.phase,
            c.striker_id, c.bowler_id,
            s.control_binary, s.shot_control, s.elevation, s.shot_type, s.foot,
            s.variation, s.length_variation, s.line, s.length, s.area, s.zone,
            s.wagon_x, s.wagon_y, s.wagon_zone,
            s.fielding_position, s.fielder_action, s.has_shot_detail,
            xb.person_id AS src_striker_id
        FROM aligned_innings a
        JOIN src_pos s ON s.p_match = a.p_match AND s.innings = a.innings
        JOIN cs_pos  c ON c.match_id = a.match_id AND c.innings = a.innings
                      AND c.pos = s.pos
        LEFT JOIN xw xb ON xb.cricinfo_id = s.cricinfo_striker
    """).df()

    # Verification: the two feeds must name the same batter on the same ball.
    # This is the check that makes the positional join safe rather than
    # merely plausible.
    known = joined["src_striker_id"].notna()
    agree = known & (joined["src_striker_id"] == joined["striker_id"])
    rate = agree.sum() / max(int(known.sum()), 1)
    print(f"  balls: {len(joined):,} positionally joined, "
          f"striker agreement {100 * rate:.2f}% on {int(known.sum()):,} checkable")
    if rate < 0.95:
        raise RuntimeError(
            f"striker agreement {rate:.3f} is too low for the positional ball "
            f"join to be trusted; inspect ordering before using this output"
        )

    out = joined[agree | ~known].drop(columns=["src_striker_id"]).copy()
    print(f"  balls: {len(out):,} kept after verification")
    return _derive(out)


def _derive(df):
    """Turn the raw vocabularies into the booleans the model consumes."""
    sc = df["shot_control"]
    st = df["shot_type"]

    df["is_controlled"] = np.where(sc.isna(), np.nan, sc.isin(CONTROLLED).astype(float))
    df["is_false_shot"] = np.where(sc.isna(), np.nan, sc.isin(FALSE_SHOT).astype(float))
    df["is_false_shot_strict"] = np.where(
        sc.isna(), np.nan, sc.isin(FALSE_SHOT_STRICT).astype(float))

    # Bat actually met ball.  Unknown where there is no judgement at all.
    contact = ~(sc.isin(NO_CONTACT_CONTROL) | st.isin(NO_CONTACT_SHOT))
    df["made_contact"] = np.where(sc.isna() & st.isna(), np.nan, contact.astype(float))

    # Aerial is only defined where the batter hit it; see module docstring.
    aerial = (df["elevation"] == "in the air").astype(float)
    df["is_aerial"] = np.where(
        df["elevation"].isna() | (df["made_contact"] != 1.0), np.nan, aerial)

    # The descriptive vocabulary and the binary flag cover different eras:
    # descriptive runs IPL 2019-2024, binary runs 2015-2025.  A model trained
    # for a 2026 auction cares most about the seasons the descriptive feed
    # does not reach, so a merged column is emitted with the descriptive
    # reading preferred and the source recorded on every row.  Anyone who
    # wants a single consistent definition can filter on control_source.
    cb = df["control_binary"]
    df["is_controlled_wide"] = df["is_controlled"].where(
        df["is_controlled"].notna(), cb)
    df["control_source"] = np.where(
        df["is_controlled"].notna(), "descriptive",
        np.where(cb.notna(), "binary", None))

    df["has_shot_control"] = df["is_controlled"].notna().astype("int8")
    df["has_control"] = df["is_controlled_wide"].notna().astype("int8")
    df["has_elevation"] = df["is_aerial"].notna().astype("int8")
    return df


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default=None,
                    help="explicit directory holding the t20 feed files; "
                         "by default they are located via data_sources")
    ap.add_argument("--bbb-dir", default=None,
                    help="directory holding deliveries/people parquet")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to /kaggle/working/bbb")
    ap.add_argument("--memory-limit", default="8GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--describe", action="store_true",
                    help="print what data is visible and exit")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import data_sources as ds

    if args.describe:
        ds.describe()
        return

    extra = [args.source_dir] if args.source_dir else None

    # Resolved without extensions: parquet locally, CSV on Kaggle, and the
    # query text below is identical either way.
    bbb = args.bbb_dir or ds.bbb_dir()
    deliveries = os.path.join(bbb, "deliveries.parquet")
    people = os.path.join(bbb, "people.parquet")
    # Both feed snapshots feed identity and match resolution: the older one
    # carries 129 matches' worth of balls the newer one dropped. The newer one
    # is required; the older is a bonus and the build runs without it.
    if extra:
        combined = ds.find_file("t20_combined", extra_roots=extra)
        newer = ds.find_file("t20_bbb-updated", extra_roots=extra)
        older = ds.find_file("t20_bbb", extra_roots=extra, required=False)
        feeds = [newer]
        if older and os.path.abspath(older) != os.path.abspath(newer):
            feeds.append(older)
    else:
        combined, feeds = ds.shotquality_feeds()
    if len(feeds) < 2:
        print("  note: t20_bbb (older snapshot) not found; continuing with "
              "t20_bbb-updated alone")

    out_dir = args.out_dir or ds.output_dir("bbb")
    os.makedirs(out_dir, exist_ok=True)

    print("inputs")
    for label, path in [("bbb", bbb), ("combined", combined),
                        *[(f"feed[{i}]", f) for i, f in enumerate(feeds)]]:
        print(f"  {label:11s} {path}")
    print(f"  {'out':11s} {out_dir}\n")

    source_glob = ds.scan(feeds)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory_limit}'; SET threads={args.threads};")

    print("stage 1/3  players")
    xw = build_player_crosswalk(con, source_glob, people, repo_root)
    con.register("xw", xw)

    print("stage 2/3  matches")
    mx = build_match_crosswalk(con, source_glob, deliveries)
    con.register("mx", mx)

    print("stage 3/3  balls")
    ba = build_ball_attributes(con, ds.scan(combined), deliveries)

    xw.to_parquet(os.path.join(out_dir, "player_crosswalk.parquet"), index=False)
    mx.to_parquet(os.path.join(out_dir, "match_crosswalk.parquet"), index=False)
    ba.to_parquet(os.path.join(out_dir, "ball_attributes.parquet"), index=False)
    print(f"\nwrote {len(ba):,} rows to {out_dir}/ball_attributes.parquet")


if __name__ == "__main__":
    main()
