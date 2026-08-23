"""
Cricbuzz player profiles -> international debut AND last-match dates
-> as-of capped status, including the five-year reversion rule.

WHY
---
`cappedStatus` is one of the strongest signals in the dataset -- capped
players clear at 6-12x the uncapped median in every edition where the
column is populated -- and it is UNPOPULATED for 2018, 2019, 2020 and
2021 (every player reads UNCAPPED). That is 690 in-pool players.

Cricbuzz profiles are keyed on the auction playerId itself:

    https://www.cricbuzz.com/profiles/10045/=
                                      ^^^^^ == auction playerId

so no identity resolution is needed. The trailing slug segment is
required but its contents are not checked; "=" works for every player.

WHAT THIS READS, AND WHY BOTH HALVES MATTER
-------------------------------------------
The profile carries, per format, BOTH a debut and a last-match date:

    Test debut / Last Test
    ODI debut  / Last ODI
    T20 debut  / Last T20
    IPL debut  / Last IPL          (kept, but never counts as a cap)

Debut alone answers "was he ever capped". It cannot answer "is he
STILL capped", and BCCI treats a player who has not appeared
internationally in the preceding five years as uncapped again. Piyush
Chawla (last India cap 2012), Karn Sharma (2014) and Mayank Markande
(2019) all read CAPPED from debut dates alone and UNCAPPED on the 2025
roster -- the roster is right, and the last-match date is what makes
that decidable.

An earlier version of this module tried to recover last-appearance
dates by matching auction playerIds to cricsheet person_ids and
reading the ball data. That was a large amount of machinery
(name-key candidate generation, debut-date confirmation to suppress
collisions) to recover a field that is on the page being fetched
anyway, and it could only ever see T20 internationals, so it would
have wrongly uncapped Test and ODI specialists. It is gone. Read the
field.

ON LEAKAGE
----------
A debut and a last-match date are both dated events, so

    capped(t) = debuted before t, AND last appearance not older than
                t - reversion_years

is the same strictly-before discipline `last_salary` uses.

One honest caveat, in `capped_as_of`'s docstring in full: when a
player's last international POST-DATES the auction being scored, the
profile cannot tell us whether he appeared in the five years just
before it. Reversion is not applied in that case -- he is left capped.
For a genuine forward prediction this branch cannot arise (the scrape
precedes the auction); it only affects backtests, and it errs toward
"he was a real international around then", which is nearly always
right.

RUNNING IT ON KAGGLE
--------------------
Internet is OFF by default: Notebook -> Settings -> Internet -> On.

    from pipelines.scrape_cricbuzz_profiles import (
        scrape, reparse_cache, drop_impossible_dates,
        validate, sweep_reversion, sanity_checks, write_capped_table)

    debuts = scrape(ids[:5], inspect=True)   # parser test first
    debuts = scrape(ids)                     # ~20 min, resumable
    debuts = drop_impossible_dates(debuts)
    validate(debuts)                         # scores vs the real column
    sweep_reversion(debuts)                  # picks reversion_years

IF YOU ALREADY SCRAPED WITH THE PREVIOUS VERSION: the raw HTML is
cached, and the last-match fields were always in it -- only the parser
ignored them. Call `reparse_cache()`. Do NOT re-scrape.
"""

from __future__ import annotations

import os
import re
import time
import random

import pandas as pd

CACHE_DIR = "/kaggle/working/cricbuzz_profiles"

DELAY_SECONDS = 1.5
DELAY_JITTER = 0.7
TIMEOUT = 20
MAX_RETRIES = 3

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

DEBUT_FIELDS = ("test_debut", "odi_debut", "t20i_debut", "ipl_debut")
LAST_FIELDS = ("last_test", "last_odi", "last_t20i", "last_ipl")
PROFILE_FIELDS = DEBUT_FIELDS + LAST_FIELDS

# International only. An IPL appearance is not a cap, in either direction.
INTERNATIONAL_DEBUTS = ("test_debut", "odi_debut", "t20i_debut")
INTERNATIONAL_LASTS = ("last_test", "last_odi", "last_t20i")

# BCCI's window. Swept rather than assumed -- see sweep_reversion.
DEFAULT_REVERSION_YEARS = 5

URL_SLUG_PLACEHOLDER = "="


def profile_url(player_id, slug=URL_SLUG_PLACEHOLDER):
    """/profiles/<id> alone 404s; the slug segment is required."""
    return f"https://www.cricbuzz.com/profiles/{int(player_id)}/{slug}"


def probe_url_patterns(player_id, timeout=TIMEOUT):
    """Try several URL shapes against one id when fetches start failing."""
    import requests
    pid = int(player_id)
    patterns = {
        "no slug":        f"https://www.cricbuzz.com/profiles/{pid}",
        "slug '='":       f"https://www.cricbuzz.com/profiles/{pid}/=",
        "slug 'x'":       f"https://www.cricbuzz.com/profiles/{pid}/x",
        "trailing slash": f"https://www.cricbuzz.com/profiles/{pid}/",
    }
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    out = {}
    for label, url in patterns.items():
        try:
            r = requests.get(url, headers=headers, timeout=timeout,
                             allow_redirects=True)
            out[label] = r.status_code
            print(f"  {'OK  ' if r.status_code == 200 else '    '}"
                  f"{label:16s} -> HTTP {r.status_code} ({len(r.text):,} bytes)")
        except Exception as exc:
            out[label] = f"{type(exc).__name__}"
            print(f"      {label:16s} -> {type(exc).__name__}")
        time.sleep(1.0)
    return out


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

_DATE_PATTERNS = (
    re.compile(r"([A-Za-z]{3})[a-z]*\s+(\d{1,2}),?\s+(\d{4})"),
    re.compile(r"(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})"),
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
)


def parse_date(text):
    """First date in `text` as ISO 'YYYY-MM-DD', or None."""
    if not isinstance(text, str):
        return None
    m = _DATE_PATTERNS[0].search(text)
    if m:
        mon = _MONTHS.get(m.group(1).lower())
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}"
    m = _DATE_PATTERNS[1].search(text)
    if m:
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(1)):02d}"
    m = _DATE_PATTERNS[2].search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------
#
# ORDER MATTERS. "last test" must be tried before "test debut" would be,
# and more importantly a label like "Last Test" must not fall through to
# the "test" branch of anything else. Longest / most specific first.

_LABEL_MAP = (
    (("last test", "test last"), "last_test"),
    (("last odi", "odi last"), "last_odi"),
    (("last t20", "last t20i", "last t20 international", "t20 last"),
     "last_t20i"),
    (("last ipl", "ipl last"), "last_ipl"),
    (("test debut", "tests debut", "debut test"), "test_debut"),
    (("odi debut", "odis debut", "debut odi"), "odi_debut"),
    (("t20 debut", "t20i debut", "t20is debut",
      "t20 international debut", "debut t20"), "t20i_debut"),
    (("ipl debut", "debut ipl"), "ipl_debut"),
)


def _field_for_label(label):
    norm = re.sub(r"[^a-z0-9 ]", " ", (label or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    for spellings, field in _LABEL_MAP:
        for s in spellings:
            if norm == s or norm.startswith(s):
                return field
    return None


def parse_debuts(html):
    """
    Extract debut and last-match dates from a Cricbuzz profile.

    Two independent strategies, merged: a DOM walk over label/value
    pairs, then a regex sweep over the flattened text for anything the
    walk missed. Returns {field: iso_date or None} plus 'profile_name'.
    """
    out = {f: None for f in PROFILE_FIELDS}
    out["profile_name"] = None
    if not html:
        return out

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        soup = None

    if soup is not None:
        h1 = soup.find(["h1"])
        if h1:
            out["profile_name"] = h1.get_text(" ", strip=True) or None

        for node in soup.find_all(["div", "td", "th", "span", "li"]):
            label = node.get_text(" ", strip=True)
            if not label or len(label) > 60:
                continue
            field = _field_for_label(label)
            if not field or out[field]:
                continue
            candidates = []
            sib = node.find_next_sibling()
            if sib is not None:
                candidates.append(sib.get_text(" ", strip=True))
            parent = node.parent
            if parent is not None:
                ptext = parent.get_text(" ", strip=True)
                candidates.append(
                    ptext[len(label):] if ptext.startswith(label) else ptext)
            for c in candidates:
                d = parse_date(c)
                if d:
                    out[field] = d
                    break

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    for spellings, field in _LABEL_MAP:
        if out[field]:
            continue
        for s in spellings:
            m = re.search(re.escape(s) + r"(.{0,120})", text, flags=re.I)
            if m:
                d = parse_date(m.group(1))
                if d:
                    out[field] = d
                    break
    return out


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _cache_path(player_id, cache_dir):
    return os.path.join(cache_dir, f"{int(player_id)}.html")


def fetch_profile(player_id, cache_dir=CACHE_DIR, session=None,
                  delay=DELAY_SECONDS, force=False):
    """(html, from_cache). html is None on failure; failures aren't cached."""
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(player_id, cache_dir)
    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(), True

    sess = session or requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    for attempt in range(MAX_RETRIES):
        try:
            resp = sess.get(profile_url(player_id), headers=headers,
                            timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code == 200 and resp.text:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(resp.text)
                time.sleep(delay + random.uniform(0, DELAY_JITTER))
                return resp.text, False
            if resp.status_code == 404:
                print(f"    playerId {player_id}: 404, no profile")
                time.sleep(delay)
                return None, False
            print(f"    playerId {player_id}: HTTP {resp.status_code} "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
        except Exception as exc:
            print(f"    playerId {player_id}: {type(exc).__name__} "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(delay * (attempt + 2) + random.uniform(0, 1.0))
    return None, False


def scrape(player_ids, cache_dir=CACHE_DIR, limit=None, inspect=False,
           delay=DELAY_SECONDS):
    """
    Fetch + parse profiles. Resumable; cached players aren't re-fetched.
    Use inspect=True on a handful first -- it is a parser test.
    """
    ids = list(dict.fromkeys(int(p) for p in player_ids))
    if limit:
        ids = ids[:limit]

    rows, n_cached, n_failed = [], 0, 0
    for i, pid in enumerate(ids, 1):
        html, cached = fetch_profile(pid, cache_dir=cache_dir, delay=delay)
        n_cached += int(cached)
        if html is None:
            n_failed += 1
            rows.append({"playerId": pid, "profile_name": None,
                         "_fetched": False,
                         **{f: None for f in PROFILE_FIELDS}})
            continue
        parsed = parse_debuts(html)
        rows.append({"playerId": pid, "_fetched": True, **parsed})
        if inspect:
            got = {k: v for k, v in parsed.items() if v}
            print(f"  [{i}/{len(ids)}] playerId {pid}: {got or 'NOTHING PARSED'}")
        elif i % 50 == 0:
            print(f"  {i}/{len(ids)} ({n_cached} cached, {n_failed} failed)",
                  flush=True)

    out = pd.DataFrame(rows)
    print(f"\nscraped {len(out)} profiles "
          f"({n_cached} from cache, {n_failed} failed)")
    _parse_health(out)
    return out.drop(columns=["_fetched"], errors="ignore")


def reparse_cache(player_ids=None, cache_dir=CACHE_DIR):
    """
    Re-parse already-cached HTML. Free, no network.

    This is how to pick up the last-match fields if the profiles were
    scraped by the earlier parser that ignored them: the data was
    always in the cached pages.
    """
    if player_ids is None:
        player_ids = [int(f[:-5]) for f in os.listdir(cache_dir)
                      if f.endswith(".html")]
    rows = []
    for pid in player_ids:
        path = _cache_path(pid, cache_dir)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            rows.append({"playerId": int(pid), "_fetched": True,
                         **parse_debuts(fh.read())})
    out = pd.DataFrame(rows)
    print(f"re-parsed {len(out)} cached profiles (no network)")
    _parse_health(out)
    return out.drop(columns=["_fetched"], errors="ignore")


def _parse_health(debuts):
    """Fetch failures and parse failures reported separately."""
    if not len(debuts):
        print("  PARSE HEALTH: no rows.")
        return
    if "_fetched" in debuts.columns:
        fetched = debuts[debuts["_fetched"]]
        n_unfetched = len(debuts) - len(fetched)
    else:
        fetched, n_unfetched = debuts, 0

    if n_unfetched:
        print(f"  fetch: {n_unfetched}/{len(debuts)} page(s) NOT retrieved.")
        if n_unfetched == len(debuts):
            print("    ^ EVERY fetch failed -- a FETCH problem, not a parser "
                  "problem. 404s => run probe_url_patterns(<id>). 403s => "
                  "Kaggle internet off, or rate limited (raise "
                  "DELAY_SECONDS). Failures are not cached; re-run retries.")
            return

    print(f"  parse health (over {len(fetched)} fetched page(s)):")
    for f in PROFILE_FIELDS:
        got = fetched[f].notna().sum() if f in fetched.columns else 0
        print(f"    {f:12s} {got:4d}/{len(fetched)} ({got / max(len(fetched),1):.0%})")

    intl_cols = [c for c in INTERNATIONAL_DEBUTS if c in fetched.columns]
    none_intl = fetched[intl_cols].isna().all(axis=1).sum() if intl_cols else 0
    print(f"    no international debut at all: {none_intl}/{len(fetched)} "
          f"(expected -- most of the pool is uncapped domestic)")

    last_cols = [c for c in INTERNATIONAL_LASTS if c in fetched.columns]
    if last_cols:
        have_debut = ~fetched[intl_cols].isna().all(axis=1)
        have_last = ~fetched[last_cols].isna().all(axis=1)
        n_d = int(have_debut.sum())
        n_both = int((have_debut & have_last).sum())
        print(f"    capped players with a LAST-match date: {n_both}/{n_d}")
        if n_d and n_both == 0:
            print("    ^ debuts parsed but NO last-match dates. The reversion "
                  "rule cannot work. Check the 'Last Test/ODI/T20' labels in "
                  "a cached page and fix _LABEL_MAP, then reparse_cache().")


# ---------------------------------------------------------------------------
# Capped status
# ---------------------------------------------------------------------------

def _earliest(row, fields):
    vals = [row.get(f) for f in fields]
    vals = [v for v in vals if isinstance(v, str) and v]
    return min(vals) if vals else None


def _latest(row, fields):
    vals = [row.get(f) for f in fields]
    vals = [v for v in vals if isinstance(v, str) and v]
    return max(vals) if vals else None


def capped_as_of(debut_row, auction_date,
                 reversion_years=DEFAULT_REVERSION_YEARS):
    """
    CAPPED as of `auction_date`, with the five-year reversion rule.

    Rules, in order:
      1. No international debut on record   -> UNCAPPED, known=False.
         "No debut" is genuinely ambiguous -- a real domestic player, or
         a failed parse -- so it is flagged rather than asserted.
      2. Earliest international debut is not before the auction
                                            -> UNCAPPED, known=True.
      3. reversion_years is None            -> CAPPED (rule off).
      4. Last international appearance is on/after the auction date
                                            -> CAPPED. See the caveat.
      5. Last international appearance older than
         auction_date - reversion_years     -> UNCAPPED (reverted).
      6. Otherwise                          -> CAPPED.

    When no last-match date was parsed for a capped player, the debut
    date stands in for it: a player with one cap twenty years ago and
    no last-match record should revert, and using his debut as his last
    known appearance gets that right.

    THE CAVEAT ON RULE 4. If the player's last international post-dates
    the auction being scored, this data cannot say whether he appeared
    in the five years immediately before it -- only that he was active
    at some point after. Reversion is therefore not applied and he is
    left capped. For a genuine forward prediction the branch cannot
    arise, since the scrape precedes the auction. It only affects
    backtests, and it errs toward "he was a real international around
    then", which is nearly always correct. The alternative -- ignoring
    post-auction appearances entirely -- would revert every still-active
    veteran whose debut happens to be more than five years before the
    auction, which is badly wrong.
    """
    debut = _earliest(debut_row, INTERNATIONAL_DEBUTS)
    if debut is None:
        return "UNCAPPED", False

    asof = str(auction_date)
    if debut >= asof:
        return "UNCAPPED", True

    if reversion_years is None:
        return "CAPPED", True

    last = _latest(debut_row, INTERNATIONAL_LASTS) or debut
    if last >= asof:
        return "CAPPED", True

    cutoff = f"{int(asof[:4]) - int(reversion_years)}{asof[4:]}"
    return ("UNCAPPED" if last < cutoff else "CAPPED"), True


# ---------------------------------------------------------------------------
# Cleaning and checking
# ---------------------------------------------------------------------------

def drop_impossible_dates(debuts, archetypes_path=None, min_age=12.0,
                          verbose=True):
    """
    Blank any date that cannot be real: before age `min_age`, in the
    future, or a last-match date earlier than its own debut.

    Name collisions and parse slips produce exactly these. Blanking
    lets the missing-flag carry it instead of keeping a known-false
    number.
    """
    import data_sources as ds
    if archetypes_path is None:
        archetypes_path = ds.archetypes_path()
    A = pd.read_csv(archetypes_path)

    out = debuts.copy()
    dob = out["playerId"].map(dict(zip(
        A.player_id, pd.to_datetime(A.date_of_birth, errors="coerce"))))
    today = pd.Timestamp.today()

    n = 0
    for f in PROFILE_FIELDS:
        if f not in out.columns:
            continue
        d = pd.to_datetime(out[f], errors="coerce")
        bad = ((d - dob).dt.days < 365 * min_age) | (d > today)
        bad = bad.fillna(False)
        n += int(bad.sum())
        out.loc[bad, f] = None

    # last-before-debut is incoherent; drop the last, keep the debut
    for deb, lst in (("test_debut", "last_test"), ("odi_debut", "last_odi"),
                     ("t20i_debut", "last_t20i"), ("ipl_debut", "last_ipl")):
        if deb in out.columns and lst in out.columns:
            bad = (pd.to_datetime(out[lst], errors="coerce")
                   < pd.to_datetime(out[deb], errors="coerce")).fillna(False)
            n += int(bad.sum())
            out.loc[bad, lst] = None

    if verbose:
        print(f"drop_impossible_dates: blanked {n} impossible date(s) "
              f"across {len(out)} players")
    return out


def sanity_checks(debuts, archetypes_path=None):
    """Internal consistency checks needing no ground truth."""
    import data_sources as ds
    if archetypes_path is None:
        archetypes_path = ds.archetypes_path()
    A = pd.read_csv(archetypes_path)
    d = debuts.merge(A[["player_id", "auction_name", "date_of_birth"]],
                     left_on="playerId", right_on="player_id", how="left")

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    problems = []
    for f in PROFILE_FIELDS:
        if f not in d.columns:
            continue
        have = d[d[f].notna()]
        for _, r in have[have[f] > today].iterrows():
            problems.append((r.playerId, r.auction_name, f, f"future: {r[f]}"))
        dob = pd.to_datetime(have["date_of_birth"], errors="coerce")
        deb = pd.to_datetime(have[f], errors="coerce")
        for _, r in have[(deb - dob).dt.days < 365 * 12].iterrows():
            problems.append((r.playerId, r.auction_name, f,
                             f"{r[f]} vs DOB {r.date_of_birth}"))

    def surname(n):
        return n.split()[-1].lower() if isinstance(n, str) and n.split() else None
    named = d[d.profile_name.notna() & d.auction_name.notna()]
    for _, r in named.iterrows():
        a, b = surname(r.auction_name), surname(r.profile_name)
        if a and b and a != b:
            problems.append((r.playerId, r.auction_name, "profile_name",
                             f"profile says {r.profile_name!r}"))

    print(f"=== sanity checks: {len(problems)} problem(s) ===")
    for p in problems[:25]:
        print(f"  playerId {p[0]} ({p[1]}) [{p[2]}]: {p[3]}")
    if len(problems) > 25:
        print(f"  ... and {len(problems) - 25} more")
    return pd.DataFrame(problems,
                        columns=["playerId", "auction_name", "field", "problem"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _scorable_editions(player_template, auction_dates):
    """
    Editions whose cappedStatus is actually populated.

    A minority class under 5% is a scrape artifact, not a real
    distribution: 2019 has 269 UNCAPPED and exactly one CAPPED, which
    has two distinct values but is not usable truth.
    """
    out = {}
    for year, asof in auction_dates.items():
        path = player_template.format(year=year)
        if not os.path.exists(path):
            continue
        roster = pd.read_csv(path)
        truth = roster["cappedStatus"].astype(str).str.upper()
        minority = min((truth == "CAPPED").mean(), (truth == "UNCAPPED").mean())
        out[year] = (roster, truth, asof,
                     not (truth.nunique() <= 1 or minority < 0.05))
    return out


def validate(debuts, reversion_years=DEFAULT_REVERSION_YEARS,
             player_template=None, verbose=True):
    """Score derived capped status against the editions that have it."""
    from src.training import AUCTION_DATES
    import data_sources as ds
    if player_template is None:
        player_template = ds.player_template()

    lookup = debuts.set_index("playerId").to_dict("index")
    eds = _scorable_editions(player_template, AUCTION_DATES)

    rows, disagreements = [], []
    for year, (roster, truth, asof, usable) in sorted(eds.items()):
        res = [capped_as_of(lookup.get(int(p), {}), asof, reversion_years)
               for p in roster["playerId"]]
        derived = pd.Series([r[0] for r in res], index=roster.index)
        known = pd.Series([r[1] for r in res], index=roster.index)

        rows.append({
            "year": year, "n": len(roster),
            "truth_capped": int((truth == "CAPPED").sum()),
            "truth_usable": usable,
            "derived_capped": int((derived == "CAPPED").sum()),
            "accuracy": float((derived == truth).mean()) if usable else float("nan"),
            "baseline_all_uncapped": float((truth == "UNCAPPED").mean()) if usable else float("nan"),
            "missing_rate": float(1 - known.mean()),
        })
        if usable:
            for i in roster.index[derived != truth]:
                disagreements.append({
                    "year": year, "playerName": roster.at[i, "playerName"],
                    "country": roster.at[i, "country"],
                    "truth": truth.at[i], "derived": derived.at[i]})

    report = pd.DataFrame(rows)
    if verbose:
        print(f"=== capped status (reversion_years={reversion_years}) ===")
        print(report.to_string(index=False))
        u = report[report.truth_usable]
        if len(u):
            print(f"\nmean accuracy: {u.accuracy.mean():.1%}   "
                  f"baseline: {u.baseline_all_uncapped.mean():.1%}")
            if u.accuracy.mean() < u.baseline_all_uncapped.mean():
                print("  ^ WORSE THAN BASELINE -- do not ship; the join or "
                      "the parse is wrong.")
        deg = report[~report.truth_usable]
        if len(deg):
            print(f"\nunpopulated editions (the gap being filled): "
                  f"{list(deg.year)}")
            print(deg[["year", "n", "derived_capped", "missing_rate"]]
                  .to_string(index=False))
        if disagreements:
            d = pd.DataFrame(disagreements)
            print(f"\n{len(d)} disagreements, by country:")
            print(d.country.value_counts().head(8).to_string())
            print(d.head(12).to_string(index=False))
    return report, pd.DataFrame(disagreements)


def sweep_reversion(debuts, windows=(None, 3, 4, 5, 6, 7, 8),
                    player_template=None):
    """
    Score every reversion window and report which wins.

    The rule is not assumed to help -- it is measured. If `none` wins,
    ship without reversion; that is a real outcome, not a failure.
    """
    from src.training import AUCTION_DATES
    import data_sources as ds
    if player_template is None:
        player_template = ds.player_template()

    lookup = debuts.set_index("playerId").to_dict("index")
    eds = _scorable_editions(player_template, AUCTION_DATES)

    rows = []
    for w in windows:
        accs = []
        for year, (roster, truth, asof, usable) in eds.items():
            if not usable:
                continue
            derived = pd.Series(
                [capped_as_of(lookup.get(int(p), {}), asof, w)[0]
                 for p in roster["playerId"]], index=roster.index)
            accs.append(float((derived == truth).mean()))
        rows.append({"reversion_years": "none" if w is None else w,
                     "mean_accuracy": sum(accs) / len(accs) if accs else float("nan"),
                     "editions_scored": len(accs)})

    out = pd.DataFrame(rows)
    print("=== reversion window sweep ===")
    print(out.to_string(index=False))
    best = out.loc[out.mean_accuracy.idxmax()]
    print(f"\nbest: reversion_years={best.reversion_years} "
          f"at {best.mean_accuracy:.1%}")
    if str(best.reversion_years) == "none":
        print("  -> reversion does NOT help here. Ship without it.")
    else:
        print(f"  -> pass reversion_years={best.reversion_years} to "
              f"write_capped_table().")
    return out


def write_capped_table(debuts, out_path,
                       reversion_years=DEFAULT_REVERSION_YEARS,
                       player_template=None):
    """
    Write the per-(playerId, edition) capped table for the pipeline.

    Per EDITION, not per player: capped status is time-varying, and a
    single per-player column cannot express "uncapped in 2018, capped
    in 2021, uncapped again in 2026". Baking one in would repeat the
    age_at_last_auction mistake the config already drops as leaky.
    """
    from src.training import AUCTION_DATES
    import data_sources as ds
    if player_template is None:
        player_template = ds.player_template()

    lookup = debuts.set_index("playerId").to_dict("index")
    rows = []
    for year, asof in sorted(AUCTION_DATES.items()):
        path = player_template.format(year=year)
        if not os.path.exists(path):
            continue
        roster = pd.read_csv(path)
        for pid in roster["playerId"].unique():
            status, known = capped_as_of(lookup.get(int(pid), {}), asof,
                                         reversion_years)
            rows.append({"playerId": int(pid), "auction_year": int(year),
                         "capped": 1.0 if status == "CAPPED" else 0.0,
                         "capped_is_missing": 0.0 if known else 1.0})

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} (player, edition) rows to {out_path}")
    print(out.groupby("auction_year")[["capped", "capped_is_missing"]]
          .mean().to_string())
    return out
