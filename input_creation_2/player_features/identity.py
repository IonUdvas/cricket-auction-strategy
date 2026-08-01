"""
Bridge from an auction roster (Cricbuzz names + Cricbuzz playerIds) to Cricsheet
person ids.

The ball-by-ball side is now identity-clean: every delivery carries a stable
`person_id`.  The auction side still arrives as free-text names, so exactly one
join remains ambiguous, and this module makes that join explicit, auditable, and
conservative rather than silent.

Resolution order, first match wins:

  1. an entry in the overrides CSV, keyed on the auction `playerId`
  2. an exact match against any Cricsheet **name variant** (not just the
     canonical name -- "Lokesh Rahul" only matches via the variant list)
  3. a normalised match (case, accents, punctuation and spacing removed)
  4. a surname + initials signature match, but only when it is unique

Anything that resolves to more than one person, or to none, is left
**unresolved** and reported.  An unresolved player gets an empty stats row with
`has_history = 0`, which is honest; guessing would silently attach one
cricketer's career to another.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


_TOKEN_SPLIT = re.compile(r"[^A-Za-z]+")


def _tokens(name):
    """Case-preserving tokens. Case is the signal that tells an initials blob
    ("KL") apart from a given name ("Lokesh"), so it must survive here."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return []
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return [t for t in _TOKEN_SPLIT.split(s) if t]


def normalize_name(name):
    """Matching key only -- never displayed, never stored as an identity."""
    return " ".join(t.lower() for t in _tokens(name))


def name_signature(name):
    """
    (surname, initials) -- the key that makes "KL Rahul" and "Lokesh Rahul"
    comparable without making "K Gowtham" and "Krishna Gowda" comparable.

    A short all-caps token is an initials blob and contributes *every* letter:
    "KL" -> "kl", not "k". The old version took only the first letter of each
    whitespace-separated part, so every Cricsheet-style name was silently
    truncated -- "JP Duminy" signed as ("duminy", "j") and could never match
    "Jean-Paul Duminy" -> ("duminy", "jp").
    """
    toks = _tokens(name)
    if not toks:
        return ("", "")
    initials = []
    for t in toks[:-1]:
        if t.isupper() and len(t) <= 4:
            initials.extend(ch.lower() for ch in t)
        else:
            initials.append(t[0].lower())
    return (toks[-1].lower(), "".join(initials))


def _is_subsequence(short, long):
    """Every letter of `short`, in order, appears in `long`."""
    it = iter(long)
    return all(ch in it for ch in short)


class PlayerIdentityResolver:
    def __init__(self, people, overrides=None, resolution=None, squad_index=None):
        """
        people     : people.parquet from data.build_bbb
                     (person_id, canonical_name, name_variants, key_cricinfo)
        overrides  : optional DataFrame/CSV path with columns
                     playerId, person_id, action  (action in {"map", "block"})
        resolution : optional DataFrame/CSV path -- the search-populated
                     identity cache, columns playerId, cricinfo_id (plus any
                     provenance columns, which are carried but not used).

        The resolution cache is keyed on `cricinfo_id`, not `person_id`, on
        purpose.  A Cricinfo id is an external, stable, human-checkable fact
        about a cricketer ("is espncricinfo.com/cricketers/kl-rahul-422108 the
        man in this auction row?"), whereas a Cricsheet person_id is an opaque
        hash that nobody can verify by eye and that would have to be re-derived
        if Cricsheet ever reissued it.  `key_cricinfo` is a bijection onto
        person_id over every person in the ball data, so storing the
        verifiable one costs nothing.
        """
        self.people = people.copy()
        self.squad_index = squad_index

        self._by_variant = {}
        self._by_norm = {}
        self._by_sig = {}
        self._by_surname = {}

        for row in people.itertuples(index=False):
            for variant in str(row.name_variants).split("|"):
                variant = variant.strip()
                if not variant:
                    continue
                self._by_variant.setdefault(variant, set()).add(row.person_id)
                self._by_norm.setdefault(normalize_name(variant), set()).add(row.person_id)
                sig = name_signature(variant)
                self._by_sig.setdefault(sig, set()).add(row.person_id)
                self._by_surname.setdefault(sig[0], set()).add((sig[1], row.person_id))

        self.overrides = {}
        self.blocked = set()
        if overrides is not None:
            ov = pd.read_csv(overrides) if isinstance(overrides, str) else overrides
            for r in ov.itertuples(index=False):
                action = str(getattr(r, "action", "map")).strip().lower()
                if action == "block":
                    self.blocked.add(r.playerId)
                else:
                    self.overrides[r.playerId] = r.person_id

        self.unresolved = []
        self.ambiguous = []

        # ---- search-populated resolution cache ---------------------------
        # cricinfo_id -> person_id, straight off the register.
        self._person_by_cricinfo = {}
        if "key_cricinfo" in people.columns:
            for pid_, key in zip(people["person_id"], people["key_cricinfo"]):
                if pd.notna(key):
                    self._person_by_cricinfo[int(key)] = pid_

        self.resolution = {}
        self.cache_misses = []
        if resolution is not None:
            res = (pd.read_csv(resolution)
                   if isinstance(resolution, str) else resolution)
            for r in res.itertuples(index=False):
                cid = getattr(r, "cricinfo_id", None)
                if pd.isna(cid):
                    continue
                person = self._person_by_cricinfo.get(int(cid))
                if person is None:
                    # A real answer, not a failure: this cricketer has a
                    # Cricinfo profile but has never appeared in the T20 ball
                    # data we built.  Record it so it is not silently retried
                    # as if the search had never run.
                    self.cache_misses.append((r.playerId, int(cid)))
                    continue
                self.resolution[r.playerId] = person

    def resolve_one(self, player_id, player_name, context=None):
        if player_id in self.blocked:
            return None, "blocked"
        if player_id in self.overrides:
            return self.overrides[player_id], "override"
        if player_id in self.resolution:
            return self.resolution[player_id], "cricinfo_cache"

        for table, how in (
            (self._by_variant, "exact"),
            (self._by_norm, "normalised"),
        ):
            key = player_name if how == "exact" else normalize_name(player_name)
            hits = table.get(key)
            if hits and len(hits) == 1:
                return next(iter(hits)), how
            if hits:
                narrowed = self._narrow(hits, context)
                if narrowed:
                    return narrowed, f"{how}+squad"
                self.ambiguous.append((player_id, player_name, how, sorted(hits)))
                return None, "ambiguous"

        hits = self._by_sig.get(name_signature(player_name))
        if hits and len(hits) == 1:
            return next(iter(hits)), "signature"
        if hits:
            narrowed = self._narrow(hits, context)
            if narrowed:
                return narrowed, "signature+squad"
            self.ambiguous.append((player_id, player_name, "signature", sorted(hits)))
            return None, "ambiguous"

        # Indian naming: the auction carries the everyday name ("Dinesh
        # Karthik", "Lokesh Rahul") while Cricsheet carries the full initials
        # ("KD Karthik", "KL Rahul"). The everyday initials are a subsequence
        # of the full ones and never the reverse, so the test is directional.
        # Allowing the reverse too lets a single-initial Cricsheet entry
        # swallow any auction name sharing that letter -- "Usama Mir" captures
        # "Umar Nazir Mir". Uniqueness is still required, so "S Sharma" stays
        # ambiguous rather than picking a bowler at random.
        surname, initials = name_signature(player_name)
        if surname and initials:
            subset = {
                pid_ for cand, pid_ in self._by_surname.get(surname, ())
                if _is_subsequence(initials, cand)
            }
            if len(subset) == 1:
                return next(iter(subset)), "initials_subset"
            if subset:
                narrowed = self._narrow(subset, context)
                if narrowed:
                    return narrowed, "initials_subset+squad"
                self.ambiguous.append(
                    (player_id, player_name, "initials_subset", sorted(subset))
                )
                return None, "ambiguous"

        self.unresolved.append((player_id, player_name))
        return None, "unresolved"

    def _narrow(self, candidates, context):
        """Squad evidence, but only ever to narrow an existing candidate set.
        Returns a single person_id or None -- never a guess."""
        if self.squad_index is None or not context:
            return None
        hit = self.squad_index.narrow(
            candidates, context.get("season"), context.get("team")
        )
        return next(iter(hit)) if len(hit) == 1 else None

    def resolve(self, roster, id_column="playerId", name_column="playerName",
                team_column="playsForTeam", season_column="season_year"):
        """
        roster : DataFrame with the auction's own id and name columns.

        Returns a copy with `person_id` and `match_method` added.  Rows that
        could not be resolved keep `person_id = None` -- they are not dropped,
        so downstream row counts stay stable and the gap stays visible.
        """
        self.unresolved, self.ambiguous = [], []
        out = roster.copy()
        has_ctx = (self.squad_index is not None
                   and team_column in out.columns and season_column in out.columns)
        contexts = (
            [{"team": t, "season": s} for t, s in
             zip(out[team_column], out[season_column])]
            if has_ctx else [None] * len(out)
        )
        resolved = [
            self.resolve_one(pid, name, ctx)
            for pid, name, ctx in zip(out[id_column], out[name_column], contexts)
        ]
        out["person_id"] = [r[0] for r in resolved]
        out["match_method"] = [r[1] for r in resolved]
        return out

    def report(self, roster_resolved=None):
        lines = []
        if roster_resolved is not None:
            counts = roster_resolved["match_method"].value_counts().to_dict()
            total = len(roster_resolved)
            matched = int(roster_resolved["person_id"].notna().sum())
            lines.append(f"resolved {matched}/{total} ({matched / total:.1%})")
            lines.append(f"  by method: {counts}")
        if self.ambiguous:
            lines.append(f"  ambiguous ({len(self.ambiguous)}):")
            for pid, name, how, hits in self.ambiguous[:25]:
                lines.append(f"    {name!r} (auction id {pid}) -> {how} hit {hits}")
        if self.unresolved:
            lines.append(f"  unresolved ({len(self.unresolved)}):")
            for pid, name in self.unresolved[:25]:
                lines.append(f"    {name!r} (auction id {pid})")
        if self.ambiguous or self.unresolved:
            lines.append(
                "  -> add these to data/name_overrides.csv "
                "(playerId, person_id, action) and re-run"
            )
        return "\n".join(lines)
