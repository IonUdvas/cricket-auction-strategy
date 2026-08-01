"""
Bridge from an auction roster (Cricbuzz names + Cricbuzz playerIds) to Cricsheet
person ids.

The ball-by-ball side is now identity-clean: every delivery carries a stable
`person_id`.  The auction side still arrives as free-text names, so exactly one
join remains ambiguous, and this module makes that join explicit, auditable, and
conservative rather than silent.

Resolution happens **once per auction playerId**, not once per roster row.  A
playerId is one human being across every year it appears, so it gets one answer,
computed from every spelling and every (season, franchise) context that playerId
was ever seen with.  Two things follow from that:

  * Cross-year conflicts are impossible by construction rather than caught after
    the fact.  A 2026 row carries no ball data yet, but the same playerId's 2019
    row does, and pooling means the 2026 row inherits that evidence instead of
    guessing from the name alone.
  * When two spellings of one playerId ("Shivam Dubey" / "Shivam Dube") answer
    differently, that disagreement is recorded and the playerId is left
    unresolved unless squad evidence settles it.

Resolution order for a single spelling, first match wins:

  1. an entry in the overrides CSV, keyed on the auction `playerId`
  2. an entry in the cricinfo resolution cache, also keyed on `playerId`
  3. an exact match against any Cricsheet **name variant** (not just the
     canonical name -- "Lokesh Rahul" only matches via the variant list)
  4. a normalised match (case, accents, punctuation and spacing removed)
  5. a surname + initials signature match
  6. a surname match where the auction initials are a **subsequence** of the
     Cricsheet initials

Tiers 3-6 do not stop at the first tier that produces candidates.  A tier that
produces several candidates and cannot be narrowed by squad evidence **falls
through to the next tier** instead of terminating the search.  That single
change is what lets "Rohit Sharma" be found: the exact tier matches two obscure
domestic namesakes whose Cricsheet names are spelled out in full, neither of
whom ever played for Mumbai, and the man everyone means is filed as "RG Sharma",
four tiers down.  Terminating at the first ambiguous tier hid him -- along with
Suryakumar Yadav, Axar Patel, Harshal Patel and most of the top of the auction.

Anything that resolves to more than one person, or to none, is left
**unresolved** and reported.  An unresolved player gets an empty stats row with
`has_history = 0`, which is honest; guessing would silently attach one
cricketer's career to another.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


# An apostrophe always lives *inside* a token: splitting "D'Arcy" into "D" +
# "Arcy" signed D'Arcy Short as ("short", "da"), which can never reach the real
# "DJM Short".
_APOSTROPHE = re.compile(r"['’]")
# A hyphen is genuinely ambiguous and the two readings pull opposite ways.  In
# a given name it is intra-word -- "Lhuan-dre Pretorius" must sign as
# ("pretorius", "l") to reach "LG Pretorius".  In a surname it is a separator
# in one source and not the other -- "Javon Searles" only reaches
# "JPR Scantlebury-Searles" if that surname is also filed under "searles".
# So neither reading is dropped: both are indexed and both are queried.
_HYPHEN = re.compile(r"-")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z]+")

# Nobiliary particles carry no initial in either naming convention, but they are
# tokens, so they inflate both blobs and do it asymmetrically when one side
# abbreviates the given name and the other does not.  Dropping them makes
# "Faf du Plessis" and "F du Plessis" agree on ("plessis", "f") rather than
# meeting at ("plessis", "fd") by luck.  Deliberately conservative: "al", "bin"
# and "ibn" are NOT here, because in Arabic and Bengali names they sit inside
# the given name rather than in front of the surname ("Shakib Al Hasan"), and
# stripping them would merge people the ball data keeps apart.
_PARTICLES = {
    "van", "von", "der", "den", "de", "du", "di", "da", "das", "dos",
    "la", "le", "ten", "ter", "af", "op",
}


def _tokens(name, join_hyphens=True):
    """Case-preserving tokens. Case is the signal that tells an initials blob
    ("KL") apart from a given name ("Lokesh"), so it must survive here."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return []
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _APOSTROPHE.sub("", s)
    s = _HYPHEN.sub("" if join_hyphens else " ", s)
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

    Nobiliary particles contribute nothing (see `_PARTICLES`); the surname is
    still the final token, so "van der Dussen" keeps "dussen".
    """
    toks = _tokens(name) if isinstance(name, (str, float, type(None))) else name
    if isinstance(toks, str) or toks is None:
        toks = _tokens(toks)
    if not toks:
        return ("", "")
    initials = []
    for t in toks[:-1]:
        # Order matters: an initials blob is checked FIRST, because several
        # particles are also real initial pairs.  "DA Miller" is David Andrew,
        # not a Portuguese particle, and testing the particle list first
        # silently signed him as ("miller", "") -- no initials at all, so no
        # tier could ever reach him.
        if t.isupper() and len(t) <= 4:
            initials.extend(ch.lower() for ch in t)
        elif t.lower() in _PARTICLES:
            continue
        else:
            initials.append(t[0].lower())
    return (toks[-1].lower(), "".join(initials))


def name_signatures(name):
    """
    Every (surname, initials) pair this name could reasonably sign as.

    Exactly one for the overwhelming majority of names.  Two when a hyphen is
    present, because the hyphen is a token separator in one source and not in
    the other and there is no way to know which from the string alone -- so
    both readings are kept and the caller matches against either.
    """
    sigs = {name_signature(_tokens(name))}
    if _HYPHEN.search(str(name) if name is not None else ""):
        sigs.add(name_signature(_tokens(name, join_hyphens=False)))
    return {sig for sig in sigs if sig[0]}


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
                for sig in name_signatures(variant):
                    self._by_sig.setdefault(sig, set()).add(row.person_id)
                    self._by_surname.setdefault(sig[0], set()).add(
                        (sig[1], row.person_id)
                    )

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
        self.conflicts = []

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
                if cid is None or pd.isna(cid):
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

    # ------------------------------------------------------------------
    # candidate tiers
    # ------------------------------------------------------------------

    def _surname_matches(self, player_name):
        """(signature hits, subsequence hits, prefix hits) over every reading
        of `player_name` -- see `name_signatures` on why there can be two."""
        exact_sig, subset, prefix = set(), set(), set()
        for surname, initials in name_signatures(player_name):
            exact_sig |= self._by_sig.get((surname, initials), set())
            if not initials:
                continue
            pool = self._by_surname.get(surname, ())
            subset |= {p for c, p in pool if _is_subsequence(initials, c)}
            prefix |= {p for c, p in pool if c.startswith(initials)}
        return exact_sig, subset, prefix

    def _tiers(self, player_name):
        """(method_name, candidate_set) in strictness order, lazily thin."""
        exact_sig, subset, _ = self._surname_matches(player_name)
        tiers = [
            ("exact", self._by_variant.get(player_name)),
            ("normalised", self._by_norm.get(normalize_name(player_name))),
            ("signature", exact_sig),
        ]
        if subset:
            # Indian naming: the auction carries the everyday name ("Dinesh
            # Karthik", "Lokesh Rahul") while Cricsheet carries the full
            # initials ("KD Karthik", "KL Rahul").  The everyday initials are a
            # subsequence of the full ones and never the reverse, so the test
            # is directional.  Allowing the reverse too lets a single-initial
            # Cricsheet entry swallow any auction name sharing that letter --
            # "Usama Mir" captures "Umar Nazir Mir".
            tiers.append(("initials_subset", subset))
        return tiers

    def _prefix_matches(self, player_name):
        """
        Candidates whose Cricsheet initials *begin* with the auction initials.

        This is a tie-break, never a tier.  Cricsheet writes initials
        given-name-first, so anchoring separates brothers that a subsequence
        test cannot: "Hardik" -> "h" is a prefix of "hh" (Hardik Himanshu) but
        not of "kh" (Krunal Hitesh), while both *contain* "h".

        It is not a tier because plenty of cricketers go by a middle name, and
        for them the anchor points at the wrong man with total confidence:
        "Lasith Malinga" anchors onto LN Malinga and away from SL Malinga, and
        "Lokesh Rahul" would anchor away from KL Rahul.  Squad evidence gets
        the first word for exactly that reason; this only speaks when squad
        evidence is silent and the alternative is giving up.
        """
        return self._surname_matches(player_name)[2]

    def _narrow(self, candidates, contexts):
        """
        Squad evidence, but only ever to narrow an existing candidate set.
        Returns a single person_id or None -- never a guess.

        `contexts` is every (season, franchise) pair this playerId was ever
        listed with, not just the one on the row being resolved.  A player
        bought in the 2026 auction has no 2026 ball data, but usually has a
        2019 season somewhere in the same playerId's history, and that is the
        same evidence about the same person.
        """
        if self.squad_index is None or not contexts or not candidates:
            return None
        hits = set()
        for ctx in contexts:
            if not ctx:
                continue
            hits |= self.squad_index.narrow(
                candidates, ctx.get("season"), ctx.get("team")
            )
        return next(iter(hits)) if len(hits) == 1 else None

    def _resolve_name(self, player_name, contexts=()):
        """
        Resolve one spelling.

        Returns (person_id | None, method), or (None, "ambiguous", evidence).

        The shape of this is deliberate and was arrived at by watching it get
        two famous cricketers wrong.

        *A unique name match wins outright.*  The strictest tier that produces
        any candidate at all is the one that speaks: if it names exactly one
        person, that is the answer.  Looser tiers are not consulted, because a
        tier that matched two people matched them on a *better* test than the
        tier below it, and skipping past them to a fuzzy singleton is how you
        pick a stranger.

        *Otherwise squad evidence chooses, from the union of every tier.*  This
        is the part that matters.  Narrowing tier-by-tier looks safer and is
        not: "Rohit Sharma" matches two obscure full-name namesakes exactly,
        and one more at the signature tier who really did turn out for Mumbai,
        so a per-tier narrow returns Raghu Sharma with total confidence while
        RG Sharma -- the man the auction is bidding on, four crore a year, one
        tier further down -- was never in the pool being narrowed.  Pooling
        first puts him in the pool.  Same story for Rinku Singh, who a per-tier
        narrow hands to his KKR team-mate Ramandeep Singh.

        A failure is reported as "ambiguous" when some tier had candidates and
        "unresolved" when nothing matched at all -- the two need different
        fixes, so they keep different names.
        """
        tiers = [(how, hits) for how, hits in self._tiers(player_name) if hits]
        if not tiers:
            return None, "unresolved"

        how, hits = tiers[0]
        if len(hits) == 1:
            return next(iter(hits)), how

        union = set().union(*(h for _, h in tiers))
        narrowed = self._narrow(union, contexts)
        if narrowed:
            return narrowed, f"{how}+squad"

        anchored = union & self._prefix_matches(player_name)
        if len(anchored) == 1:
            return next(iter(anchored)), f"{how}+prefix"

        return None, "ambiguous", (how, sorted(union))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def resolve_player(self, player_id, player_names, contexts=()):
        """
        One answer for one auction playerId, from every spelling it has ever
        been listed under and every (season, franchise) it has ever carried.
        """
        if player_id in self.blocked:
            return None, "blocked"
        if player_id in self.overrides:
            return self.overrides[player_id], "override"
        if player_id in self.resolution:
            return self.resolution[player_id], "cricinfo_cache"

        names = [n for n in dict.fromkeys(player_names) if pd.notna(n)]
        answers, methods, misses = {}, {}, []
        for name in names:
            result = self._resolve_name(name, contexts)
            person, how = result[0], result[1]
            if person is None:
                if how == "ambiguous":
                    misses.append((name, result[2]))
                continue
            answers.setdefault(person, how)
            methods.setdefault(person, how)

        if len(answers) == 1:
            person = next(iter(answers))
            return person, methods[person]

        if len(answers) > 1:
            # Two spellings of one playerId disagreed.  That is evidence at
            # least one of them is wrong, so it is never majority-voted -- but
            # squad membership is an independent fact and may settle it.
            narrowed = self._narrow(set(answers), contexts)
            if narrowed:
                return narrowed, "spelling_conflict+squad"
            self.conflicts.append(
                (player_id, names[0], sorted(answers))
            )
            return None, "conflict"

        if misses:
            name, (how, hits) = misses[0]
            self.ambiguous.append((player_id, name, how, hits))
            return None, "ambiguous"

        self.unresolved.append((player_id, names[0] if names else None))
        return None, "unresolved"

    def resolve_one(self, player_id, player_name, context=None):
        """Single-row convenience wrapper, kept for callers that have one row
        and one context.  `resolve` is the real entry point."""
        return self.resolve_player(
            player_id, [player_name], [context] if context else ()
        )

    def resolve(self, roster, id_column="playerId", name_column="playerName",
                team_column="playsForTeam", season_column="season_year"):
        """
        roster : DataFrame with the auction's own id and name columns.  It may
                 span several auction years; that is the intended use.

        Returns a copy with `person_id` and `match_method` added.  Rows that
        could not be resolved keep `person_id = None` -- they are not dropped,
        so downstream row counts stay stable and the gap stays visible.

        Every row sharing a playerId gets the same answer, because it is
        computed once for the playerId rather than once per row.
        """
        self.unresolved, self.ambiguous, self.conflicts = [], [], []
        out = roster.copy()
        has_ctx = (self.squad_index is not None
                   and team_column in out.columns and season_column in out.columns)

        decisions = {}
        for pid_, grp in out.groupby(id_column, sort=False):
            contexts = (
                [{"team": t, "season": s} for t, s in
                 zip(grp[team_column], grp[season_column])]
                if has_ctx else ()
            )
            decisions[pid_] = self.resolve_player(
                pid_, list(grp[name_column]), contexts
            )

        out["person_id"] = [decisions[p][0] for p in out[id_column]]
        out["match_method"] = [decisions[p][1] for p in out[id_column]]
        return out

    def report(self, roster_resolved=None):
        lines = []
        if roster_resolved is not None:
            counts = roster_resolved["match_method"].value_counts().to_dict()
            total = len(roster_resolved)
            matched = int(roster_resolved["person_id"].notna().sum())
            ids = roster_resolved.drop_duplicates(subset=["playerId"]) \
                if "playerId" in roster_resolved.columns else roster_resolved
            n_ids = len(ids)
            id_hits = int(ids["person_id"].notna().sum())
            lines.append(f"resolved {matched}/{total} rows ({matched / total:.1%}), "
                         f"{id_hits}/{n_ids} playerIds ({id_hits / n_ids:.1%})")
            lines.append(f"  by method: {counts}")
        if self.conflicts:
            lines.append(f"  spelling conflicts ({len(self.conflicts)}):")
            for pid, name, hits in self.conflicts[:25]:
                lines.append(f"    {name!r} (auction id {pid}) -> spellings "
                             f"disagreed between {hits}")
        if self.ambiguous:
            lines.append(f"  ambiguous ({len(self.ambiguous)}):")
            for pid, name, how, hits in self.ambiguous[:25]:
                lines.append(f"    {name!r} (auction id {pid}) -> {how} hit {hits}")
        if self.unresolved:
            lines.append(f"  unresolved ({len(self.unresolved)}):")
            for pid, name in self.unresolved[:25]:
                lines.append(f"    {name!r} (auction id {pid})")
        if self.conflicts or self.ambiguous or self.unresolved:
            lines.append(
                "  -> resolve these by adding a row to "
                "data/identity/cricinfo_resolution.csv "
                "(playerId, playerName, cricinfo_id, dob, method, note) "
                "and re-run"
            )
        return "\n".join(lines)