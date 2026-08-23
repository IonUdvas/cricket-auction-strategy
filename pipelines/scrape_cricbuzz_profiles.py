"""
Cricbuzz player profiles -> international debut dates -> as-of capped status.

WHY
---
`cappedStatus` in the auction files is one of the strongest signals in
the dataset -- capped players sell at 6-12x the uncapped median in
every edition where the column is populated:

    2022  capped median 240L vs uncapped 20L
    2024  capped median 200L vs uncapped 20L
    2025  capped median 320L vs uncapped 30L
    2026  capped median 200L vs uncapped 30L

and it is ENTIRELY ABSENT for 2018, 2019, 2020 and 2021: every player
in those four rosters is labelled UNCAPPED (2019 has exactly one
CAPPED, itself an artifact). That is 690 in-pool players, roughly half
of them genuinely capped internationals, all given the same value.

WHY CRICBUZZ RATHER THAN THE BALL DATA
--------------------------------------
Two other sources were tried first and are documented here so they are
not re-attempted:

  * Matching cricsheet international appearances BY NAME fails:
    cricsheet stores "SM Curran", the archetype table stores
    "Sam Curran". Naive matching scored ~50% (chance).

  * people.csv's `key_cricbuzz` column would be a direct join to the
    auction playerId, but it is populated for only 50 of 18,362 rows
    (1.2% roster coverage). Unusable.

  * A hybrid (archetype `capped_status` snapshot + cricsheet T20I
    debut) reached 89-96% on validation. Its residual error is almost
    entirely players capped via TESTS OR ODIs ONLY -- Will Sutherland,
    Kuldeep Sen, Akash Deep, Dan Lawrence -- who are invisible to a
    T20-only ball dataset. There is no way to fix that from the
    cricsheet zips, which are all T20 formats.

Cricbuzz profiles carry Test, ODI and T20I debut dates SEPARATELY,
which is exactly the missing piece. And the profile URL is keyed on
the auction playerId itself:

    https://www.cricbuzz.com/profiles/10045/liam-livingstone
                                      ^^^^^ == auction playerId

so no identity resolution is needed at all -- the single biggest
source of error in every other approach disappears.

ON LEAKAGE
----------
A debut date is a DATED EVENT, so

    capped_as_of(auction_date) = any(debut_date < auction_date)

is the same strictly-before discipline `last_salary` already uses. No
hindsight. This is the property that makes scraping this safe, and it
is why this is worth doing while hand-labelling archetypes from
general knowledge is NOT (that would encode what a player later became).

RUNNING IT ON KAGGLE
--------------------
Internet is OFF by default in Kaggle notebooks. Turn it on:
    Notebook -> Settings (right panel) -> Internet -> On
(requires a phone-verified account).

    from pipelines.scrape_cricbuzz_profiles import scrape, validate

    # 1. START SMALL. Five players, inspect what came back.
    debuts = scrape(player_ids[:5], inspect=True)

    # 2. Then the whole roster. Resumable: re-running only fetches
    #    players not already cached on disk.
    debuts = scrape(player_ids)

    # 3. CHECK IT before believing it.
    validate(debuts)

Raw HTML is cached to disk before parsing, so if the parser is wrong
the pages do not need re-fetching -- fix `parse_debuts` and re-run
`reparse_cache()`.

A NOTE ON THE PARSER
--------------------
The selectors below were written WITHOUT access to a live Cricbuzz
page (the authoring sandbox blocks the domain: `x-deny-reason:
host_not_allowed`). They are deliberately written as several
independent strategies over the page text rather than one brittle CSS
path, and `inspect=True` prints what each player yielded. Treat the
first five-player run as a parser test, not a formality.
"""

from __future__ import annotations

import os
import re
import time
import random

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = "/kaggle/working/cricbuzz_profiles"

# Be polite. This is ~800 requests against someone else's site; there is
# no reason to do it fast, and a burst is both rude and the quickest way
# to get blocked. ~1.5s mean with jitter is ~20 minutes for a full run,
# once, cached forever after.
DELAY_SECONDS = 1.5
DELAY_JITTER = 0.7
TIMEOUT = 20
MAX_RETRIES = 3

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# The four debut fields worth having. IPL debut is not used for capped
# status (it is not international) but is cheap to keep and is a useful
# cross-check on whether the right profile was fetched.
DEBUT_FIELDS = ("test_debut", "odi_debut", "t20i_debut", "ipl_debut")

# International formats only. IPL debut deliberately excluded.
INTERNATIONAL_FIELDS = ("test_debut", "odi_debut", "t20i_debut")


# Cricbuzz's profile route REQUIRES a slug segment after the id, but does
# not validate its contents -- /profiles/10069/= serves the same page as
# /profiles/10069/liam-livingstone, while /profiles/10069 (no segment)
# returns 404 rather than redirecting.
#
# That matters because we do not know each player's real slug and do not
# want to scrape an index to find out. Any placeholder works. "=" is used
# because it was verified against the live site; if Cricbuzz ever tightens
# the route, `probe_url_patterns` below finds the new shape against one id
# instead of failing 800 times.
URL_SLUG_PLACEHOLDER = "="


def profile_url(player_id, slug=URL_SLUG_PLACEHOLDER):
    """
    Profile URL for an auction playerId.

    The trailing slug segment is required (see URL_SLUG_PLACEHOLDER).
    Omitting it 404s -- that was the first version of this function and
    it failed on every player.
    """
    return f"https://www.cricbuzz.com/profiles/{int(player_id)}/{slug}"


def probe_url_patterns(player_id, patterns=None, timeout=TIMEOUT):
    """
    Try several URL shapes against ONE id and report the status of each.

    Run this first if a scrape starts returning 404s. It costs a handful
    of requests and tells you which pattern is live, instead of guessing
    and re-running the whole roster.

    Returns {pattern_label: status_code_or_error}.
    """
    import requests

    pid = int(player_id)
    if patterns is None:
        patterns = {
            "no slug":            f"https://www.cricbuzz.com/profiles/{pid}",
            "slug '='":           f"https://www.cricbuzz.com/profiles/{pid}/=",
            "slug 'x'":           f"https://www.cricbuzz.com/profiles/{pid}/x",
            "trailing slash":     f"https://www.cricbuzz.com/profiles/{pid}/",
        }

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    out = {}
    for label, url in patterns.items():
        try:
            r = requests.get(url, headers=headers, timeout=timeout,
                             allow_redirects=True)
            out[label] = r.status_code
            marker = "OK  " if r.status_code == 200 else "    "
            print(f"  {marker}{label:16s} -> HTTP {r.status_code}  "
                  f"({len(r.text):,} bytes)  {r.url}")
        except Exception as exc:
            out[label] = f"{type(exc).__name__}: {exc}"
            print(f"      {label:16s} -> {type(exc).__name__}")
        time.sleep(1.0)
    return out


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# "Aug 04, 2016" / "04 Aug 2016" / "2016-08-04"
_DATE_PATTERNS = (
    re.compile(r"([A-Za-z]{3})[a-z]*\s+(\d{1,2}),?\s+(\d{4})"),   # Aug 04, 2016
    re.compile(r"(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})"),      # 04 Aug 2016
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),                        # 2016-08-04
)


def parse_date(text):
    """First date-looking thing in `text`, as ISO 'YYYY-MM-DD', or None."""
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

# Label -> field. Cricbuzz has used several spellings across redesigns,
# so match on a normalised label rather than an exact string.
_LABEL_MAP = (
    (("test debut", "tests debut"), "test_debut"),
    (("odi debut", "odis debut"), "odi_debut"),
    (("t20 debut", "t20i debut", "t20is debut", "t20 international debut"),
     "t20i_debut"),
    (("ipl debut",), "ipl_debut"),
)


def _field_for_label(label):
    norm = re.sub(r"[^a-z0-9 ]", " ", (label or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    for spellings, field in _LABEL_MAP:
        for s in spellings:
            if norm.startswith(s) or norm == s:
                return field
    return None


def parse_debuts(html):
    """
    Extract debut dates from a Cricbuzz profile page.

    Three independent strategies, tried in order and merged, because
    the page structure is not verifiable from the authoring
    environment:

      1. BeautifulSoup over the profile's label/value pairs
      2. BeautifulSoup over any <table> rows
      3. A pure-regex sweep of the raw text, as a last resort

    Returns {field: iso_date or None} plus 'name' when it can be found.
    """
    out = {f: None for f in DEBUT_FIELDS}
    out["profile_name"] = None

    if not html:
        return out

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        soup = None

    if soup is not None:
        # Name, for the cross-check that we fetched the right player.
        h1 = soup.find(["h1"])
        if h1:
            out["profile_name"] = h1.get_text(" ", strip=True) or None

        # Strategy 1 + 2: any element whose text starts with a known
        # label, paired with its sibling / parent's remaining text.
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
                if ptext.startswith(label):
                    candidates.append(ptext[len(label):])
                else:
                    candidates.append(ptext)

            for c in candidates:
                d = parse_date(c)
                if d:
                    out[field] = d
                    break

    # Strategy 3: regex over the flattened text. Catches layouts the
    # DOM walk missed; only fills fields still empty.
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
    """
    Raw HTML for one profile, from disk cache when available.

    Returns (html, from_cache). html is None when the fetch failed;
    failures are NOT cached, so a re-run retries them.
    """
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(player_id, cache_dir)

    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(), True

    sess = session or requests.Session()
    headers = {"User-Agent": USER_AGENT,
               "Accept-Language": "en-US,en;q=0.9"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = sess.get(profile_url(player_id), headers=headers,
                            timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code == 200 and resp.text:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(resp.text)
                time.sleep(delay + random.uniform(0, DELAY_JITTER))
                return resp.text, False

            # 404 is a real answer (no such profile) -- do not retry it.
            if resp.status_code == 404:
                print(f"    playerId {player_id}: 404, no profile")
                time.sleep(delay)
                return None, False

            print(f"    playerId {player_id}: HTTP {resp.status_code} "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
        except Exception as exc:
            print(f"    playerId {player_id}: {type(exc).__name__} "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")

        # Back off before retrying; a 403/429 usually means slow down.
        time.sleep(delay * (attempt + 2) + random.uniform(0, 1.0))

    return None, False


def scrape(player_ids, cache_dir=CACHE_DIR, limit=None, inspect=False,
           delay=DELAY_SECONDS):
    """
    Fetch + parse profiles for `player_ids`.

    limit   : stop after this many (use 5 for the first run)
    inspect : print what was parsed for each player -- USE THIS FIRST.
              The parser was written without access to a live page, so
              the first run is a parser test.

    Returns a DataFrame: playerId, profile_name, <the four debut cols>.
    Resumable: cached players are not re-fetched.
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
                         **{f: None for f in DEBUT_FIELDS}})
            continue

        parsed = parse_debuts(html)
        rows.append({"playerId": pid, "_fetched": True, **parsed})

        if inspect:
            got = {k: v for k, v in parsed.items() if v}
            print(f"  [{i}/{len(ids)}] playerId {pid}: {got or 'NOTHING PARSED'}")
        elif i % 50 == 0:
            print(f"  {i}/{len(ids)} ({n_cached} from cache, "
                  f"{n_failed} failed)", flush=True)

    out = pd.DataFrame(rows)
    print(f"\nscraped {len(out)} profiles "
          f"({n_cached} from cache, {n_failed} failed)")
    _parse_health(out)
    return out.drop(columns=["_fetched"], errors="ignore")


def reparse_cache(player_ids=None, cache_dir=CACHE_DIR):
    """
    Re-run `parse_debuts` over already-cached HTML. Free, no network.
    This is the whole point of caching raw HTML: if the parser is
    wrong, fix it and call this instead of re-fetching 800 pages.
    """
    if player_ids is None:
        player_ids = [
            int(f[:-5]) for f in os.listdir(cache_dir) if f.endswith(".html")
        ]
    rows = []
    for pid in player_ids:
        path = _cache_path(pid, cache_dir)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            rows.append({"playerId": int(pid), "_fetched": True,
                         **parse_debuts(fh.read())})
    out = pd.DataFrame(rows)
    _parse_health(out)
    return out.drop(columns=["_fetched"], errors="ignore")


def _parse_health(debuts):
    """
    Did the parser actually work? Printed after every scrape.

    Fetch failures and parse failures are reported SEPARATELY and the
    parse rate is computed over fetched rows only. The first version
    pooled them, so five 404s printed "the parser is not matching this
    page layout" -- pointing at the wrong problem entirely, when the
    real cause was a URL that 404s without a slug segment. A diagnostic
    that names the wrong culprit is worse than none.
    """
    if not len(debuts):
        print("  PARSE HEALTH: no rows.")
        return

    if "_fetched" in debuts.columns:
        fetched = debuts[debuts["_fetched"]]
        n_unfetched = len(debuts) - len(fetched)
    else:
        fetched, n_unfetched = debuts, 0

    if n_unfetched:
        print(f"  fetch: {n_unfetched}/{len(debuts)} page(s) could NOT be "
              f"retrieved (nothing to parse for those).")
        if n_unfetched == len(debuts):
            print("    ^ EVERY fetch failed -- this is a FETCH problem, not "
                  "a parser problem.")
            print("      If they were 404s, the URL shape is wrong: run")
            print("        probe_url_patterns(<any playerId>)")
            print("      to find which pattern is live. If they were 403s, "
                  "Kaggle internet may be off, or you are being rate "
                  "limited -- raise DELAY_SECONDS.")
            print("      Failures are NOT cached, so re-running retries them.")
            return

    print(f"  parse health (over the {len(fetched)} page(s) actually fetched):")
    for f in DEBUT_FIELDS:
        got = fetched[f].notna().sum()
        pct = got / len(fetched) if len(fetched) else 0
        print(f"    {f:12s} parsed for {got:4d}/{len(fetched)} ({pct:.0%})")

    none_at_all = fetched[list(DEBUT_FIELDS)].isna().all(axis=1).sum()
    print(f"    no debut of any kind: {none_at_all}/{len(fetched)}")

    # Some players genuinely have no debut of any kind -- uncapped
    # domestic players are most of the pool -- so a nonzero count here
    # is expected. All of them being empty is not.
    if len(fetched) and none_at_all == len(fetched):
        print("    ^ pages fetched but NOTHING parsed from any of them -- "
              "now this IS the parser. Inspect a cached file and fix "
              "parse_debuts, then call reparse_cache(). Do NOT re-scrape.")


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def capped_as_of(debut_row, auction_date):
    """
    CAPPED iff any INTERNATIONAL debut is strictly before auction_date.

    Strictly-before is the leakage boundary, same rule as last_salary.
    IPL debut is not international and is ignored here.

    Returns (status, known). `known` is False when the player has no
    debut of any kind on record -- which is genuinely ambiguous (a true
    uncapped domestic player, or a failed parse) and should be carried
    as a missing-flag rather than silently treated as UNCAPPED.
    """
    dates = [debut_row.get(f) for f in INTERNATIONAL_FIELDS]
    dates = [d for d in dates if isinstance(d, str) and d]
    if not dates:
        return "UNCAPPED", False
    return ("CAPPED" if min(dates) < str(auction_date) else "UNCAPPED"), True


def build_capped_table(debuts, rosters):
    """
    Long table: one row per (playerId, auction_year).

    rosters : {year: completed_players DataFrame}

    Returns playerId, auction_year, cb_capped (1/0), cb_capped_is_missing.
    This is the shape build_demographic_features wants -- per (player,
    edition), NOT one row per player. Capped status is time-varying and
    a per-player column cannot express it (that is the same mistake as
    age_at_last_auction, which the config already drops as leaky).
    """
    from src.training import AUCTION_DATES

    lookup = debuts.set_index("playerId").to_dict("index")
    rows = []
    for year, roster in rosters.items():
        asof = AUCTION_DATES[year]
        for pid in roster["playerId"].unique():
            row = lookup.get(int(pid), {})
            status, known = capped_as_of(row, asof)
            rows.append({"playerId": int(pid), "auction_year": int(year),
                         "cb_capped": 1.0 if status == "CAPPED" else 0.0,
                         "cb_capped_is_missing": 0.0 if known else 1.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation -- run this before believing any of the above
# ---------------------------------------------------------------------------

def validate(debuts, player_template=None, verbose=True):
    """
    Score the derived capped status against the editions that HAVE a
    real cappedStatus (2022-2026), and report what it would produce
    for the four broken ones (2018-2021).

    Prints, and returns, a per-edition frame. Three things to look at:

      accuracy      -- vs the scraped cappedStatus. Should beat the
                       89-96% the cricsheet hybrid managed. If it is
                       near 50%, the join or the parse is broken.
      baseline      -- accuracy of "call everyone UNCAPPED", which is
                       what 2018-2021 currently does. The derived
                       column has to beat this to be worth anything.
      missing_rate  -- players with no debut on record at all.

    A per-player disagreement list is returned too, so a systematic
    failure (e.g. all Australians wrong) is visible rather than
    averaged away.
    """
    from src.training import AUCTION_DATES
    import data_sources as ds

    if player_template is None:
        player_template = ds.player_template()

    lookup = debuts.set_index("playerId").to_dict("index")
    rows, disagreements = [], []

    for year in sorted(AUCTION_DATES):
        path = player_template.format(year=year)
        if not os.path.exists(path):
            continue
        roster = pd.read_csv(path)
        asof = AUCTION_DATES[year]

        derived, known = [], []
        for pid in roster["playerId"]:
            s, k = capped_as_of(lookup.get(int(pid), {}), asof)
            derived.append(s)
            known.append(k)
        roster["derived"] = derived
        roster["known"] = known

        truth = roster["cappedStatus"].astype(str).str.upper()
        n_truth_capped = int((truth == "CAPPED").sum())

        ####################################################
        # An edition whose truth is degenerate cannot score
        # anything, and must not be averaged in.
        #
        # `nunique() <= 1` alone is not enough: 2019 has 269
        # UNCAPPED and exactly ONE CAPPED, which is a scrape
        # artifact rather than a real distribution, but it has
        # two distinct values so a naive check calls it usable.
        # Scoring against it produced a meaningless 42.9%
        # accuracy against a 99.6% "baseline" and dragged the
        # reported mean down by six points. Any edition where
        # the minority class is under 5% of the roster is
        # treated as unpopulated.
        ####################################################
        minority = min((truth == "CAPPED").mean(),
                       (truth == "UNCAPPED").mean())
        degenerate = truth.nunique() <= 1 or minority < 0.05

        acc = float("nan") if degenerate else float((roster.derived == truth).mean())
        base = float("nan") if degenerate else float((truth == "UNCAPPED").mean())

        rows.append({
            "year": year,
            "n": len(roster),
            "truth_capped": n_truth_capped,
            "truth_usable": not degenerate,
            "derived_capped": int((roster.derived == "CAPPED").sum()),
            "accuracy": acc,
            "baseline_all_uncapped": base,
            "missing_rate": float(1 - roster["known"].mean()),
        })

        if not degenerate:
            bad = roster[roster.derived != truth]
            for _, r in bad.iterrows():
                disagreements.append({
                    "year": year, "playerName": r.get("playerName"),
                    "country": r.get("country"),
                    "truth": r.get("cappedStatus"), "derived": r["derived"],
                })

    report = pd.DataFrame(rows)

    if verbose:
        print("=== capped status: derived vs the real column ===")
        print(report.to_string(index=False))
        usable = report[report.truth_usable]
        if len(usable):
            print(f"\nmean accuracy on scorable editions: "
                  f"{usable.accuracy.mean():.1%}")
            print(f"baseline (call everyone UNCAPPED):   "
                  f"{usable.baseline_all_uncapped.mean():.1%}")
            if usable.accuracy.mean() < usable.baseline_all_uncapped.mean():
                print("  ^ WORSE THAN THE BASELINE. Do not ship this; the "
                      "join or the parse is wrong.")
            elif usable.accuracy.mean() < 0.85:
                print("  ^ below the 89-96% the cricsheet hybrid already "
                      "achieved -- check the disagreement list before "
                      "preferring this source.")

        deg = report[~report.truth_usable]
        if len(deg):
            print(f"\neditions with no usable truth (this is the gap being "
                  f"filled): {list(deg.year)}")
            print(deg[["year", "n", "derived_capped", "missing_rate"]]
                  .to_string(index=False))

        if disagreements:
            d = pd.DataFrame(disagreements)
            print(f"\n{len(d)} disagreements. By country (top 10):")
            print(d.country.value_counts().head(10).to_string())
            print("\nsample:")
            print(d.head(15).to_string(index=False))

    return report, pd.DataFrame(disagreements)


def sanity_checks(debuts, archetypes_path=None):
    """
    Cheap internal consistency checks that need no ground truth.

    Catches the failure modes that a raw accuracy number hides: a
    debut before the player was born, a debut in the future, or a
    profile whose name does not resemble the player we asked for
    (i.e. the wrong page was fetched).
    """
    import data_sources as ds
    if archetypes_path is None:
        archetypes_path = ds.archetypes_path()
    A = pd.read_csv(archetypes_path)

    d = debuts.merge(
        A[["player_id", "auction_name", "date_of_birth"]],
        left_on="playerId", right_on="player_id", how="left")

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    problems = []

    for f in INTERNATIONAL_FIELDS:
        have = d[d[f].notna()]
        future = have[have[f] > today]
        for _, r in future.iterrows():
            problems.append((r.playerId, r.auction_name, f,
                             f"debut in the future: {r[f]}"))
        dob = pd.to_datetime(have["date_of_birth"], errors="coerce")
        deb = pd.to_datetime(have[f], errors="coerce")
        too_young = have[(deb - dob).dt.days < 365 * 12]
        for _, r in too_young.iterrows():
            problems.append((r.playerId, r.auction_name, f,
                             f"debut {r[f]} vs DOB {r.date_of_birth}"))

    # Did we fetch the right page? Compare surnames.
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
    return pd.DataFrame(
        problems, columns=["playerId", "auction_name", "field", "problem"])


def drop_impossible_debuts(debuts, archetypes_path=None, min_age=12.0,
                           verbose=True):
    """
    Blank any debut date that cannot be real, and return the cleaned frame.

    `sanity_checks` REPORTS problems; this one ACTS on the subset that
    is unambiguously wrong -- a debut before the player was twelve, or
    a debut in the future. Both mean the date belongs to a different
    person (a name collision) rather than being a mis-typed date, so
    the right move is to blank the field and let the missing-flag carry
    it, not to keep a number known to be false.

    This matters: on the cricsheet-name-matched stand-in it caught
    Kartik Sharma "debuting" in 2014 aged 8 and Rinku Singh in 2007
    aged 9 -- both collisions with other players of the same name, and
    both would otherwise have marked a genuinely uncapped teenager as
    a capped international in every edition.

    Real Cricbuzz data is keyed on the auction playerId so collisions
    should be rare, but "should be" is not "is", and this costs one
    pass over 800 rows.
    """
    import data_sources as ds
    if archetypes_path is None:
        archetypes_path = ds.archetypes_path()
    A = pd.read_csv(archetypes_path)

    out = debuts.copy()
    dob = out["playerId"].map(
        dict(zip(A.player_id, pd.to_datetime(A.date_of_birth, errors="coerce")))
    )
    today = pd.Timestamp.today()

    n_dropped = 0
    for f in DEBUT_FIELDS:
        if f not in out.columns:
            continue
        deb = pd.to_datetime(out[f], errors="coerce")
        too_young = (deb - dob).dt.days < 365 * min_age
        in_future = deb > today
        bad = (too_young | in_future).fillna(False)
        n_dropped += int(bad.sum())
        out.loc[bad, f] = None

    if verbose:
        print(f"drop_impossible_debuts: blanked {n_dropped} impossible "
              f"debut date(s) across {len(out)} players")
    return out
