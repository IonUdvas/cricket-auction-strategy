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


def normalize_name(name):
    """Matching key only -- never displayed, never stored as an identity."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)          # drops apostrophes, hyphens, dots
    return re.sub(r"\s+", " ", s).strip()


def name_signature(name):
    """
    (surname, initials) -- the key that makes "KL Rahul" and "Lokesh Rahul"
    comparable without making "K Gowtham" and "Krishna Gowda" comparable.
    """
    parts = normalize_name(name).split()
    if not parts:
        return ("", "")
    return (parts[-1], "".join(p[0] for p in parts[:-1]))


class PlayerIdentityResolver:
    def __init__(self, people, overrides=None):
        """
        people    : people.parquet from data_prep.build_bbb
                    (person_id, canonical_name, name_variants)
        overrides : optional DataFrame/CSV path with columns
                    playerId, person_id, action  (action in {"map", "block"})
        """
        self.people = people.copy()

        self._by_variant = {}
        self._by_norm = {}
        self._by_sig = {}

        for row in people.itertuples(index=False):
            for variant in str(row.name_variants).split("|"):
                variant = variant.strip()
                if not variant:
                    continue
                self._by_variant.setdefault(variant, set()).add(row.person_id)
                self._by_norm.setdefault(normalize_name(variant), set()).add(row.person_id)
                self._by_sig.setdefault(name_signature(variant), set()).add(row.person_id)

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

    def resolve_one(self, player_id, player_name):
        if player_id in self.blocked:
            return None, "blocked"
        if player_id in self.overrides:
            return self.overrides[player_id], "override"

        for table, how in (
            (self._by_variant, "exact"),
            (self._by_norm, "normalised"),
        ):
            key = player_name if how == "exact" else normalize_name(player_name)
            hits = table.get(key)
            if hits and len(hits) == 1:
                return next(iter(hits)), how
            if hits:
                self.ambiguous.append((player_id, player_name, how, sorted(hits)))
                return None, "ambiguous"

        hits = self._by_sig.get(name_signature(player_name))
        if hits and len(hits) == 1:
            return next(iter(hits)), "signature"
        if hits:
            self.ambiguous.append((player_id, player_name, "signature", sorted(hits)))
            return None, "ambiguous"

        self.unresolved.append((player_id, player_name))
        return None, "unresolved"

    def resolve(self, roster, id_column="playerId", name_column="playerName"):
        """
        roster : DataFrame with the auction's own id and name columns.

        Returns a copy with `person_id` and `match_method` added.  Rows that
        could not be resolved keep `person_id = None` -- they are not dropped,
        so downstream row counts stay stable and the gap stays visible.
        """
        self.unresolved, self.ambiguous = [], []
        out = roster.copy()
        resolved = [
            self.resolve_one(pid, name)
            for pid, name in zip(out[id_column], out[name_column])
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
