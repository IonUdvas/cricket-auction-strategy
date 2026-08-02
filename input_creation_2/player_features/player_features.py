"""
As-of-date T20 player statistics, computed from Cricsheet-derived ball-by-ball
data (see data_prep/build_bbb.py).

Design notes
------------
1. **Everything is keyed on `person_id`, never on a name.**  Names are carried
   only as display labels.  This is what makes "KL Rahul" / "Lokesh Rahul" one
   player and the two "Amit Mishra"s two players.

2. **Deliveries are folded to player-innings once, up front.**  A T20 innings
   has ~11 batters and ~6 bowlers, so the delivery table (millions of rows)
   collapses to a few hundred thousand.  Every later query runs against that.

3. **As-of queries are cumulative-sum lookups, not dataframe slices.**  For each
   player we hold their innings in date order together with the running totals,
   so "career to date D" is one `searchsorted` plus an array index -- O(log n),
   no copying, and provably identical to filtering-then-summing.

4. **Strictly `< as_of_date`.**  A match played *on* the auction date is not
   included.  This is the leakage boundary and it is tested.

5. **Undefined is `None`, never 0.**  A player who has never bowled has
   `economy = None`, not `0.0` (which would read as the best economy rate in
   the dataset).  The flattening layer turns each of these into a value plus an
   explicit `*_is_missing` indicator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = ("powerplay", "middle", "death")


# ---------------------------------------------------------------------------
# Per-player cumulative store
# ---------------------------------------------------------------------------

class _AsOfIndex:
    """
    Cumulative totals for one player, in date order.

    `dates` is sorted ascending.  `cum[q][k]` is the total of quantity `q` over
    the player's first `k` innings.  `cum[q][0]` is 0, so an empty career and a
    date before the player's debut both fall out naturally.
    """

    __slots__ = ("dates", "cum")

    def __init__(self, dates, cum):
        self.dates = dates
        self.cum = cum

    def upto(self, ts):
        k = int(np.searchsorted(self.dates, ts, side="left"))
        return {q: float(arr[k]) for q, arr in self.cum.items()}

    def window(self, ts, n_matches, match_key="new_match"):
        """
        Totals over the player's most recent `n_matches` matches before `ts`.

        Recent form is a trailing window, not a career total, and the window
        has to be counted in *matches* rather than innings: a bowler who did
        not bat in three of his last twenty games has seventeen batting
        innings in those twenty matches, and counting innings would silently
        reach back an extra three games for him and not for an opener.

        `cum[match_key]` is the running distinct-match count, so the start of
        the window is the first innings whose running count exceeds
        (matches_so_far - n_matches).  That is one searchsorted on an array
        that is monotone by construction, and the subtraction of two prefix
        sums is exact -- no re-summing, no slicing.
        """
        k = int(np.searchsorted(self.dates, ts, side="left"))
        if k == 0:
            return {q: 0.0 for q in self.cum}
        counts = self.cum[match_key]
        target = counts[k] - n_matches
        if target <= 0:
            return {q: float(arr[k]) for q, arr in self.cum.items()}
        # counts is non-decreasing; the window opens at the first innings
        # belonging to the (target+1)-th match.
        start = int(np.searchsorted(counts[: k + 1], target, side="right")) - 1
        start = max(start, 0)
        return {q: float(arr[k] - arr[start]) for q, arr in self.cum.items()}

    def since(self, ts, from_ts):
        """Totals over [from_ts, ts) -- e.g. one named season."""
        k = int(np.searchsorted(self.dates, ts, side="left"))
        j = int(np.searchsorted(self.dates, from_ts, side="left"))
        j = min(j, k)
        return {q: float(arr[k] - arr[j]) for q, arr in self.cum.items()}

    @property
    def empty_totals(self):
        return {q: 0.0 for q in self.cum}


def _build_index(frame, key_col, quantity_cols):
    """
    frame          : player-innings rows, already sorted by (date, match, innings)
    key_col        : the player id column
    quantity_cols  : columns to accumulate

    Returns dict: person_id -> _AsOfIndex
    """
    if len(frame) == 0:
        return {}

    order = np.argsort(frame[key_col].to_numpy(), kind="stable")
    ids = frame[key_col].to_numpy()[order]
    dates = frame["match_date"].to_numpy()[order]

    values = {
        q: frame[q].to_numpy(dtype=np.float64)[order] for q in quantity_cols
    }

    # Group boundaries in the id-sorted array.
    boundaries = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    starts = boundaries
    ends = np.r_[boundaries[1:], len(ids)]

    out = {}
    for s, e in zip(starts, ends):
        d = dates[s:e]
        # Stable-sorting by id preserved the original date ordering, but assert
        # it rather than trust it: every as-of query depends on this.
        if not np.all(d[:-1] <= d[1:]):
            sub = np.argsort(d, kind="stable")
            d = d[sub]
            cum = {
                q: np.r_[0.0, np.cumsum(v[s:e][sub])] for q, v in values.items()
            }
        else:
            cum = {q: np.r_[0.0, np.cumsum(v[s:e])] for q, v in values.items()}
        out[ids[s]] = _AsOfIndex(d, cum)
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

BAT_QUANTITIES = (
    "runs", "balls", "fours", "sixes", "dots", "outs", "innings", "new_match",
    "runs_powerplay", "balls_powerplay",
    "runs_middle", "balls_middle",
    "runs_death", "balls_death",
    # Boundaries split by phase.  Overall boundary% conflates a powerplay
    # opener with a death hitter; the split is what separates them.
    "boundaries_powerplay", "boundaries_middle", "boundaries_death",
    # Shot quality, from ball_attributes.parquet.  Each rate carries its own
    # denominator because coverage is partial and era-dependent: control runs
    # IPL 2015-2025 but elevation only 2019-2024, so a player's control and
    # aerial numbers are NOT computed over the same balls and a shared
    # denominator would quietly misstate both.
    "control_balls", "controlled",
    "elev_balls", "aerial",
)

BOWL_QUANTITIES = (
    "balls", "runs", "wickets", "dots", "wides", "noballs",
    "innings", "new_match", "maidens",
    "balls_powerplay", "runs_powerplay", "wickets_powerplay",
    "balls_middle", "runs_middle", "wickets_middle",
    "balls_death", "runs_death", "wickets_death",
    "boundaries_conceded_powerplay", "boundaries_conceded_middle",
    "boundaries_conceded_death",
    # A bowler's false-shot rate is the mirror of the batter's control rate:
    # the same ball, scored from the other end.
    "control_balls", "false_shots",
    "elev_balls", "aerial_conceded",
)

FIELD_QUANTITIES = ("catches", "stumpings", "run_out_involvements")


class PlayerStatsAggregator:
    """
    Parameters
    ----------
    deliveries : DataFrame
        Output of data_prep.build_bbb (deliveries.parquet).
    competitions : iterable of str, optional
        Restrict to these `competition` values.  `None` uses everything.
    fielding : DataFrame, optional
        fielding.parquet.  Enables catch/stumping/run-out counts.
    """

    def __init__(self, deliveries, competitions=None, fielding=None):
        required = {
            "match_id", "match_date", "competition", "innings",
            "striker_id", "non_striker_id", "bowler_id", "player_out_id",
            "player_out_counts", "runs_batter", "runs_conceded_bowler",
            "ball_faced", "legal_ball", "is_four", "is_six", "is_dot_batter",
            "bowler_credited", "wides", "noballs", "over", "phase",
        }
        missing = required - set(deliveries.columns)
        if missing:
            raise ValueError(
                f"deliveries frame is missing required columns: {sorted(missing)}"
            )

        d = deliveries
        if competitions is not None:
            d = d[d["competition"].isin(list(competitions))]
            if len(d) == 0:
                raise ValueError(f"no deliveries for competitions={competitions!r}")

        dates = pd.to_datetime(d["match_date"]).to_numpy()
        if pd.isna(dates).any():
            n = int(pd.isna(dates).sum())
            raise ValueError(
                f"{n} deliveries have a null match_date; fix the build step "
                f"rather than silently dropping them here"
            )

        self.competitions = competitions

        # One global vocabulary for person ids.  Every fold below works on
        # int32 codes rather than Python strings: at a few million deliveries
        # the string columns alone are the difference between fitting in memory
        # and not, and factorising once means each fold is a bincount rather
        # than a hash join.
        cols = ("striker_id", "non_striker_id", "bowler_id", "player_out_id")
        stacked = pd.concat([d[c] for c in cols], ignore_index=True)
        codes, vocab = pd.factorize(stacked, use_na_sentinel=True)
        n = len(d)
        self._vocab = list(vocab)
        self._code_of = {pid_: i for i, pid_ in enumerate(self._vocab)}
        ids = {c: codes[i * n:(i + 1) * n].astype(np.int32)
               for i, c in enumerate(cols)}
        del stacked, codes

        # Match ids likewise; only their grouping matters, never their value.
        match_codes = pd.factorize(d["match_id"], use_na_sentinel=False)[0].astype(np.int32)

        cooked = {
            "dates": dates,
            "match": match_codes,
            "innings": d["innings"].to_numpy().astype(np.int16),
            "over": d["over"].to_numpy().astype(np.int16),
            "phase": d["phase"].to_numpy(),
            **ids,
        }

        self._bat_innings = self._fold_batting(d, cooked)
        self._bowl_innings = self._fold_bowling(d, cooked)

        self._bat = _build_index(self._bat_innings, "player_code", BAT_QUANTITIES)
        self._bowl = _build_index(self._bowl_innings, "player_code", BOWL_QUANTITIES)

        # Distinct matches/innings across both roles, counted exactly rather
        # than inferred from the two role counts (a player who both bats and
        # bowls in a match must count once).
        self._appear_innings = self._fold_appearances(
            self._bat_innings, self._bowl_innings
        )
        self._appear = _build_index(
            self._appear_innings, "player_code", ("new_match", "innings")
        )

        self._field_innings = self._fold_fielding(d, fielding, cooked)
        self._field = (
            _build_index(self._field_innings, "player_code", FIELD_QUANTITIES)
            if self._field_innings is not None else {}
        )

        # Debut date per player, for a cheap "has any history" test.
        self._first_seen = {}
        for store in (self._bat, self._bowl, self._field):
            for code, idx in store.items():
                pid_ = self._vocab[code]
                first = idx.dates[0]
                if pid_ not in self._first_seen or first < self._first_seen[pid_]:
                    self._first_seen[pid_] = first

        self._cache = {}

    # -- folding -----------------------------------------------------------
    #
    # Every fold collapses delivery-level arrays to one row per
    # (player, match, innings) using integer group codes and np.bincount.
    # pandas groupby on a few million rows with a dozen derived columns costs
    # several gigabytes of intermediates; this costs one float array per
    # quantity and runs in seconds.

    @staticmethod
    def _codes(player_codes, match_codes, innings_nos):
        """Dense group code for (player, match, innings)."""
        if len(player_codes) == 0:
            return np.empty(0, dtype=np.int64), 0
        key = (
            player_codes.astype(np.int64) * (int(match_codes.max()) + 1)
            + match_codes
        ) * 8 + innings_nos.astype(np.int64)
        codes, _ = pd.factorize(key, use_na_sentinel=False)
        return codes.astype(np.int64), int(codes.max()) + 1

    @staticmethod
    def _first_index(codes, ngroups):
        """Index of the first row belonging to each group."""
        first = np.full(ngroups, -1, dtype=np.int64)
        order = np.arange(len(codes) - 1, -1, -1)
        first[codes[order]] = order
        return first

    @classmethod
    def _fold(cls, player_codes, match_codes, innings_nos, dates, quantities):
        codes, ng = cls._codes(player_codes, match_codes, innings_nos)
        if ng == 0:
            return pd.DataFrame(
                {c: np.empty(0) for c in
                 ["player_code", "match_code", "innings_no", "match_date",
                  *quantities]}
            )
        first = cls._first_index(codes, ng)
        out = {
            "player_code": player_codes[first],
            "match_code": match_codes[first],
            "innings_no": innings_nos[first],
            "match_date": dates[first],
        }
        for name, values in quantities.items():
            out[name] = np.bincount(codes, weights=values, minlength=ng)
        return pd.DataFrame(out)

    @staticmethod
    def _optional(d, column, n):
        """
        A ball_attributes column, or zeros when it was never merged in.

        Shot quality is an *enrichment*: the aggregator has to keep working on
        a bare build_bbb delivery frame, and it has to give the same answers
        it always did on one.  A missing column therefore contributes zero to
        both numerator and denominator, which makes every derived rate `None`
        rather than a fabricated 0.0.
        """
        if column not in d.columns:
            return np.zeros(n, dtype=np.float64)
        return d[column].to_numpy(dtype=np.float64, na_value=0.0)

    @classmethod
    def _fold_batting(cls, d, c):
        phase = c["phase"]
        n = len(d)
        runs = d["runs_batter"].to_numpy(dtype=np.float64)
        balls = d["ball_faced"].to_numpy().astype(np.float64)
        fours = d["is_four"].to_numpy().astype(np.float64)
        sixes = d["is_six"].to_numpy().astype(np.float64)

        # `has_control` / `has_elevation` are the coverage flags written by
        # build_shot_attributes; they are the denominators, and they are not
        # interchangeable (see BAT_QUANTITIES).
        ctrl_den = cls._optional(d, "has_control", n)
        ctrl_num = cls._optional(d, "is_controlled_wide", n) * ctrl_den
        elev_den = cls._optional(d, "has_elevation", n)
        elev_num = cls._optional(d, "is_aerial", n) * elev_den

        q = {
            "runs": runs,
            "balls": balls,
            "fours": fours,
            "sixes": sixes,
            "dots": d["is_dot_batter"].to_numpy().astype(np.float64),
            "control_balls": ctrl_den,
            "controlled": ctrl_num,
            "elev_balls": elev_den,
            "aerial": elev_num,
        }
        for ph in PHASES:
            m = (phase == ph).astype(np.float64)
            q[f"runs_{ph}"] = runs * m
            q[f"balls_{ph}"] = balls * m
            q[f"boundaries_{ph}"] = (fours + sixes) * m

        striker = c["striker_id"]
        valid = striker >= 0
        scored = cls._fold(
            striker[valid], c["match"][valid], c["innings"][valid],
            c["dates"][valid], {k: v[valid] for k, v in q.items()},
        )

        # Dismissals come from `player_out_id`, not from "was on strike": a
        # batter run out at the non-striker's end is dismissed on a delivery
        # they never faced, and must still be counted.
        out_mask = (
            (c["player_out_id"] >= 0)
            & d["player_out_counts"].to_numpy().astype(bool)
        )
        outs = cls._fold(
            c["player_out_id"][out_mask], c["match"][out_mask],
            c["innings"][out_mask], c["dates"][out_mask],
            {"outs": np.ones(int(out_mask.sum()), dtype=np.float64)},
        )

        # A batting innings is any innings the player came to the crease in --
        # as striker, as non-striker, or by being dismissed there.
        ap_player = np.concatenate([striker, c["non_striker_id"]])
        keep = ap_player >= 0
        appear = cls._fold(
            ap_player[keep],
            np.tile(c["match"], 2)[keep],
            np.tile(c["innings"], 2)[keep],
            np.tile(c["dates"], 2)[keep],
            {"innings": np.ones(int(keep.sum()), dtype=np.float64)},
        )
        appear["innings"] = 1.0
        del ap_player, keep

        keys = ["player_code", "match_code", "innings_no"]
        merged = (
            appear.merge(scored.drop(columns="match_date"), on=keys, how="left")
                  .merge(outs.drop(columns="match_date"), on=keys, how="left")
        )
        num = [x for x in merged.columns if x not in (*keys, "match_date")]
        merged[num] = merged[num].fillna(0.0)
        return _finalise(merged)

    @classmethod
    def _fold_bowling(cls, d, c):
        phase = c["phase"]
        legal = d["legal_ball"].to_numpy().astype(np.float64)
        runs = d["runs_conceded_bowler"].to_numpy(dtype=np.float64)
        wkts = d["bowler_credited"].to_numpy().astype(np.float64)

        # A bowler's dot ball: a legal delivery from which nothing is charged
        # to the bowler.  Byes and leg-byes are never charged, so they leave
        # the ball a dot for the bowler even though the batting side scored.
        dots = (
            legal.astype(bool)
            & (d["runs_batter"].to_numpy() == 0)
            & ((d["byes"].to_numpy() + d["legbyes"].to_numpy()) == 0)
        ).astype(np.float64)

        n = len(d)
        boundaries = (
            d["is_four"].to_numpy().astype(np.float64)
            + d["is_six"].to_numpy().astype(np.float64)
        )
        ctrl_den = cls._optional(d, "has_control", n)
        # False shot is the complement of control on the balls where a
        # judgement exists -- not `1 - rate`, which would count every
        # unjudged ball as a false shot.
        false_num = (1.0 - cls._optional(d, "is_controlled_wide", n)) * ctrl_den
        elev_den = cls._optional(d, "has_elevation", n)
        elev_num = cls._optional(d, "is_aerial", n) * elev_den

        q = {
            "balls": legal,
            "runs": runs,
            "wickets": wkts,
            "dots": dots,
            "wides": d["wides"].to_numpy(dtype=np.float64),
            "noballs": d["noballs"].to_numpy(dtype=np.float64),
            "control_balls": ctrl_den,
            "false_shots": false_num,
            "elev_balls": elev_den,
            "aerial_conceded": elev_num,
        }
        for ph in PHASES:
            m = (phase == ph).astype(np.float64)
            q[f"balls_{ph}"] = legal * m
            q[f"runs_{ph}"] = runs * m
            q[f"wickets_{ph}"] = wkts * m
            q[f"boundaries_conceded_{ph}"] = boundaries * m

        bowler = c["bowler_id"]
        valid = bowler >= 0
        folded = cls._fold(
            bowler[valid], c["match"][valid], c["innings"][valid],
            c["dates"][valid], {k: v[valid] for k, v in q.items()},
        )

        # Maidens: an over in which the bowler concedes nothing at all.  The
        # over key stays integer -- building a string key here would allocate
        # one Python object per delivery.
        n_over = int(c["over"].max()) + 1
        over_key = c["match"].astype(np.int64) * n_over + c["over"]
        over_codes, ng = cls._codes(
            bowler[valid], over_key[valid].astype(np.int64), c["innings"][valid]
        )
        over_runs = np.bincount(over_codes, weights=runs[valid], minlength=ng)
        over_balls = np.bincount(over_codes, weights=legal[valid], minlength=ng)
        maiden_over = ((over_runs == 0) & (over_balls > 0)).astype(np.float64)
        first = cls._first_index(over_codes, ng)
        maidens = cls._fold(
            bowler[valid][first], c["match"][valid][first],
            c["innings"][valid][first], c["dates"][valid][first],
            {"maidens": maiden_over},
        )

        keys = ["player_code", "match_code", "innings_no"]
        folded = folded.merge(maidens.drop(columns="match_date"), on=keys, how="left")
        folded["maidens"] = folded["maidens"].fillna(0.0)
        folded["innings"] = 1.0
        return _finalise(folded)

    @classmethod
    def _fold_appearances(cls, bat_innings, bowl_innings):
        cols = ["player_code", "match_code", "innings_no", "match_date"]
        both = pd.concat(
            [bat_innings[cols], bowl_innings[cols]], ignore_index=True
        ).drop_duplicates(["player_code", "match_code", "innings_no"])
        both["innings"] = 1.0
        return _finalise(both)

    def _fold_fielding(self, d, fielding, c):
        if fielding is None or len(fielding) == 0:
            return None
        f = fielding.dropna(subset=["fielder_id"])
        if len(f) == 0:
            return None

        # A pure fielder (a specialist keeper who neither batted nor bowled in
        # a match) never appears in the delivery columns, so extend the
        # vocabulary rather than dropping them.
        for pid_ in f["fielder_id"].unique():
            if pid_ not in self._code_of:
                self._code_of[pid_] = len(self._vocab)
                self._vocab.append(pid_)

        # Restricting to `competitions` filters deliveries but not the fielding
        # frame the caller handed us, so drop whole matches this view excludes.
        # That is intended narrowing, not a data fault. It must happen before
        # the codes below are built: every array from here on is positional.
        f = f[f["match_id"].isin(set(d["match_id"].unique()))]
        if len(f) == 0:
            return None

        match_lookup = pd.DataFrame({
            "match_id": d["match_id"].to_numpy(),
            "innings": c["innings"],
            "match_code": c["match"],
            "match_date": c["dates"],
        }).drop_duplicates(["match_id", "innings"])
        f = f.merge(match_lookup, on=["match_id", "innings"], how="left")

        # What is left is a fielding row in a match we kept but an innings with
        # no deliveries. It merges to null, and astype(np.int32) on a null
        # match_code yields INT_MIN rather than raising -- the catches survive,
        # attached to a nonexistent match on a NaT date. Refuse rather than
        # guess; the usual cause is super-overs dropped from deliveries but not
        # from fielding.
        unmatched = int(f["match_code"].isna().sum())
        if unmatched:
            sample = (f.loc[f["match_code"].isna(), ["match_id", "innings"]]
                      .drop_duplicates().head(5).to_dict("records"))
            raise ValueError(
                f"{unmatched} fielding rows sit in a kept match but an innings "
                f"with no deliveries, e.g. {sample}. deliveries and fielding "
                f"were filtered inconsistently at build time; rebuild with "
                f"data/build_bbb.py rather than dropping them here."
            )

        player_code = f["fielder_id"].map(self._code_of).to_numpy().astype(np.int32)

        kind = f["kind"].astype(str).str.lower().to_numpy()
        folded = self._fold(
            player_code,
            f["match_code"].to_numpy().astype(np.int32),
            f["innings"].to_numpy().astype(np.int16),
            f["match_date"].to_numpy(),
            {
                # A caught-and-bowled carries no separate fielder, so it never
                # reaches this table -- it is the bowler's own catch.
                "catches": (kind == "caught").astype(np.float64),
                "stumpings": (kind == "stumped").astype(np.float64),
                "run_out_involvements": (kind == "run out").astype(np.float64),
            },
        )
        return _finalise(folded)

    # -- queries -----------------------------------------------------------

    def known_players(self):
        return set(self._first_seen)

    def get_player_stats(self, player_id, as_of_date, recent_matches=20):
        """
        Career totals and derived metrics for `player_id` strictly *before*
        `as_of_date`, plus the same metrics over the player's most recent
        `recent_matches` matches.

        Returns a nested dict.  Metrics that are undefined for the player's
        record (average with no dismissals, economy with no balls) are `None`.

        The `recent` block answers a different question from the career block
        and the auction cares more about it: a 34-year-old with a great career
        and a bad last season is priced on the last season.  Set
        `recent_matches=None` to skip it.
        """
        stamp = pd.Timestamp(as_of_date)
        key = (player_id, stamp, recent_matches)
        if key in self._cache:
            return self._cache[key]

        ts = stamp.to_datetime64()
        code = self._code_of.get(player_id)
        bat_idx = self._bat.get(code)
        bowl_idx = self._bowl.get(code)
        field_idx = self._field.get(code)
        appear_idx = self._appear.get(code)

        bat = bat_idx.upto(ts) if bat_idx else {q: 0.0 for q in BAT_QUANTITIES}
        bowl = bowl_idx.upto(ts) if bowl_idx else {q: 0.0 for q in BOWL_QUANTITIES}
        field = (field_idx.upto(ts) if field_idx
                 else {q: 0.0 for q in FIELD_QUANTITIES})
        appear = (appear_idx.upto(ts) if appear_idx
                  else {"new_match": 0.0, "innings": 0.0})

        result = {
            "player_id": player_id,
            "as_of_date": pd.Timestamp(as_of_date),
            "has_history": bool(appear["innings"]),
            "experience": {
                "matches": int(appear["new_match"]),
                "batting_matches": int(bat["new_match"]),
                "bowling_matches": int(bowl["new_match"]),
                "batting_innings": int(bat["innings"]),
                "bowling_innings": int(bowl["innings"]),
            },
            "batting": {"raw": _as_int(bat), "metrics": _batting_metrics(bat)},
            "bowling": {"raw": _as_int(bowl), "metrics": _bowling_metrics(bowl)},
            "fielding": _as_int(field),
        }

        if recent_matches:
            r_bat = (bat_idx.window(ts, recent_matches) if bat_idx
                     else {q: 0.0 for q in BAT_QUANTITIES})
            r_bowl = (bowl_idx.window(ts, recent_matches) if bowl_idx
                      else {q: 0.0 for q in BOWL_QUANTITIES})
            result["recent"] = {
                "window_matches": recent_matches,
                "batting_matches": int(r_bat["new_match"]),
                "bowling_matches": int(r_bowl["new_match"]),
                "batting": {"raw": _as_int(r_bat),
                            "metrics": _batting_metrics(r_bat)},
                "bowling": {"raw": _as_int(r_bowl),
                            "metrics": _bowling_metrics(r_bowl)},
            }

        self._cache[key] = result
        return result


def _finalise(frame):
    """Sort a player-innings frame by date and mark first-innings-of-a-match."""
    frame = frame.sort_values(
        ["match_date", "match_code", "innings_no"], kind="mergesort"
    ).reset_index(drop=True)
    # `new_match` is 1 on a player's first innings in a given match, so its
    # cumulative sum is a distinct match count.
    frame["new_match"] = (
        ~frame.duplicated(["player_code", "match_code"])
    ).astype(np.int32)
    return frame


def _as_int(d):
    return {k: int(v) for k, v in d.items()}


def _ratio(num, den, scale=1.0):
    return (scale * num / den) if den else None


def _batting_metrics(b):
    m = {
        "average": _ratio(b["runs"], b["outs"]),
        "strike_rate": _ratio(b["runs"], b["balls"], 100.0),
        "boundary_percentage": _ratio(b["fours"] + b["sixes"], b["balls"]),
        "dot_ball_percentage": _ratio(b["dots"], b["balls"]),
        "balls_per_dismissal": _ratio(b["balls"], b["outs"]),
        "runs_per_innings": _ratio(b["runs"], b["innings"]),
        # Control is the batter-side shot-quality measure; false shot % is
        # its complement and is reported for bowlers.  Both are `None` until
        # the player has a ball with a control judgement on it, which for the
        # IPL means 2015 onwards and for most other leagues means never.
        "control_percentage": _ratio(b["controlled"], b["control_balls"]),
        "false_shot_percentage": _ratio(
            b["control_balls"] - b["controlled"], b["control_balls"]),
        # Share of *shots actually made* that went in the air.  The
        # denominator excludes plays-and-misses, leaves and hit-pads; see
        # data/build_shot_attributes.py.
        "aerial_percentage": _ratio(b["aerial"], b["elev_balls"]),
    }
    for ph in PHASES:
        m[f"strike_rate_{ph}"] = _ratio(b[f"runs_{ph}"], b[f"balls_{ph}"], 100.0)
        m[f"balls_share_{ph}"] = _ratio(b[f"balls_{ph}"], b["balls"])
        m[f"runs_{ph}"] = b[f"runs_{ph}"]
        m[f"boundary_percentage_{ph}"] = _ratio(
            b[f"boundaries_{ph}"], b[f"balls_{ph}"])
    return m


def _bowling_metrics(b):
    m = {
        "economy": _ratio(b["runs"], b["balls"], 6.0),
        "average": _ratio(b["runs"], b["wickets"]),
        "strike_rate": _ratio(b["balls"], b["wickets"]),
        "dot_ball_percentage": _ratio(b["dots"], b["balls"]),
        "wickets_per_innings": _ratio(b["wickets"], b["innings"]),
        "extras_per_ball": _ratio(b["wides"] + b["noballs"], b["balls"]),
        "false_shot_percentage": _ratio(b["false_shots"], b["control_balls"]),
        "control_conceded_percentage": _ratio(
            b["control_balls"] - b["false_shots"], b["control_balls"]),
        "aerial_conceded_percentage": _ratio(b["aerial_conceded"], b["elev_balls"]),
    }
    for ph in PHASES:
        m[f"economy_{ph}"] = _ratio(b[f"runs_{ph}"], b[f"balls_{ph}"], 6.0)
        m[f"balls_share_{ph}"] = _ratio(b[f"balls_{ph}"], b["balls"])
        m[f"strike_rate_{ph}"] = _ratio(b[f"balls_{ph}"], b[f"wickets_{ph}"])
        m[f"runs_{ph}"] = b[f"runs_{ph}"]
        m[f"boundary_percentage_{ph}"] = _ratio(
            b[f"boundaries_conceded_{ph}"], b[f"balls_{ph}"])
    return m


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------

class PlayerFeatureBuilder:
    """
    Flattens `PlayerStatsAggregator` output into one numeric row per player.

    Every metric that can be undefined is emitted as a pair:
        <name>              the value, with a neutral fill when undefined
        <name>_is_missing   1.0 when it was undefined, else 0.0

    so a model can learn a separate offset for "no data" instead of reading the
    fill as a real measurement.
    """

    def __init__(self, aggregator, missing_fill=0.0):
        self.aggregator = aggregator
        self.missing_fill = float(missing_fill)

    def build_feature_table(self, players, as_of_date, id_column="player_id"):
        """
        players : DataFrame with `id_column` (and optionally `playerName`),
                  or a plain iterable of ids.

        Returns one row per *distinct* id, in the input's first-seen order.
        """
        if isinstance(players, pd.DataFrame):
            if id_column not in players.columns:
                raise ValueError(f"players frame has no '{id_column}' column")
            frame = players.drop_duplicates(id_column).reset_index(drop=True)
            ids = frame[id_column].tolist()
            labels = (frame["playerName"].tolist()
                      if "playerName" in frame.columns else [None] * len(ids))
        else:
            ids, seen = [], set()
            for p in players:
                if p not in seen:
                    seen.add(p)
                    ids.append(p)
            labels = [None] * len(ids)

        rows = [
            self.flatten(self.aggregator.get_player_stats(pid_, as_of_date), label)
            for pid_, label in zip(ids, labels)
        ]
        out = pd.DataFrame(rows)
        out.insert(0, id_column, ids)

        assert len(out) == len(ids), "feature table lost or gained rows"
        assert out[id_column].is_unique, "feature table has duplicate player ids"
        return out

    def flatten(self, stats, label=None):
        bat_raw = stats["batting"]["raw"]
        bat_met = stats["batting"]["metrics"]
        bowl_raw = stats["bowling"]["raw"]
        bowl_met = stats["bowling"]["metrics"]
        exp = stats["experience"]
        field = stats["fielding"]

        row = {}
        if label is not None:
            row["playerName"] = label

        row["has_history"] = float(stats["has_history"])

        for k, v in exp.items():
            row[f"exp_{k}"] = float(v)

        for k in ("runs", "balls", "fours", "sixes", "dots", "outs",
                  "runs_powerplay", "balls_powerplay",
                  "runs_middle", "balls_middle",
                  "runs_death", "balls_death"):
            row[f"bat_{k}"] = float(bat_raw[k])

        for k in ("balls", "runs", "wickets", "dots", "wides", "noballs", "maidens",
                  "balls_powerplay", "runs_powerplay", "wickets_powerplay",
                  "balls_middle", "runs_middle", "wickets_middle",
                  "balls_death", "runs_death", "wickets_death"):
            row[f"bowl_{k}"] = float(bowl_raw[k])

        for k, v in field.items():
            row[f"field_{k}"] = float(v)

        for k, v in bat_met.items():
            self._put(row, f"bat_{k}", v)
        for k, v in bowl_met.items():
            self._put(row, f"bowl_{k}", v)

        return row

    def _put(self, row, name, value):
        row[name] = self.missing_fill if value is None else float(value)
        row[f"{name}_is_missing"] = 1.0 if value is None else 0.0


def feature_columns(feature_df, id_column="player_id"):
    """The numeric feature block of a table from `build_feature_table`."""
    return [c for c in feature_df.columns if c not in (id_column, "playerName")]