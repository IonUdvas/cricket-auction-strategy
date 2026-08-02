"""
Where the data actually is.

Nothing that matters lives in this repository.  The repo holds code; every
byte of data comes from a Kaggle Dataset mounted read-only under
/kaggle/input, or from /kaggle/working when it was built earlier in the same
session.  This module is the only place that knows how to find any of it.

Two datasets are required, one is optional, and one directory is written:

    inputs      udvasbasak2/ipl-auction-model-inputs          REQUIRED
                Every source byte this project owns:
                    cricsheet/    the 20 Cricsheet men's T20 json zips,
                                  plus people.csv (the register)
                    shotquality/  t20_bbb, t20_bbb-updated, t20_combined
                    identity/     cricinfo_resolution.csv
                    auction/      player_archetypes.csv

    auction     udvasbasak2/ipl-auction-trail-data            REQUIRED
                results/{completed_players,auction_trail,earnings}/*_<year>.csv
                -- the scraped auction record, one file per auction year.

    hawkeye     rhitankar21/t20-data-for-auction-project      OPTIONAL
                The IPL 2026 Hawkeye ball-tracking feed.  Nothing in the
                training path reads it yet.  It is NOT the build source for
                the ball-by-ball set: its Cricsheet snapshot is older than
                the zips in `inputs` (no Major League Tournament at all, and
                several hundred fewer recent matches), so building from it
                would quietly shrink the delivery table.

    /kaggle/working
                Session scratch, and where the two BUILT artifacts live:
                    bbb/          the five parquet tables, from
                                  pipelines/build_bbb.py
                    bbb/ball_attributes.parquet
                                  shot quality, from
                                  pipelines/build_shot_attributes.py
                Neither is stored in a dataset.  Both are derived, both are
                reproducible from `inputs` alone, and /kaggle/working is
                searched BEFORE /kaggle/input so a fresh build always wins
                over a stale mount.

Why there is no repo fallback
-----------------------------
There used to be one, and it is what this module is now written to prevent.
A `<repo>/data` root means a half-finished local build, or a file left over
from a previous experiment, can satisfy a lookup that was meant to hit a
versioned dataset -- silently, with no error, producing numbers that cannot
be reproduced from the dataset version the run recorded.  The only escape
hatch is CRICKET_DATA_DIR, which is explicit, has to be set deliberately, and
is printed by `describe()`.

Nothing here reads a data file.  It resolves paths and raises loudly when it
cannot, naming the dataset to attach, because a data-loading failure that
surfaces three functions later as an empty DataFrame is the most expensive
kind of bug in this pipeline.
"""

from __future__ import annotations

import glob
import os

# Kaggle mounts a dataset at /kaggle/input/<slug>, but attaching through the
# newer datasets browser produces /kaggle/input/datasets/<owner>/<slug>.
# Both are globbed rather than picked between.
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

# Set CRICKET_DATA_DIR to one directory, or several separated by os.pathsep,
# to run off local copies.  This is the ONLY way to read data that is not on
# Kaggle, and it has to be set on purpose.
ENV_OVERRIDE = "CRICKET_DATA_DIR"

# Every extension we know how to read, in preference order.  Parquet first:
# where both exist it is smaller, typed, and an order of magnitude faster to
# scan than the CSV of the same table.
DATA_EXTENSIONS = (".parquet", ".csv", ".csv.gz", ".zip")

# How deep to look inside a mounted dataset.  Kaggle datasets routinely wrap
# their payload in one or two folders named after the upload, and the
# cricsheet one nests competitions four levels down.
MAX_DEPTH = 6

# Directory names that are never worth walking into.
_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}

# This repository's own directory.
#
# It is EXCLUDED from every search, and that exclusion is the whole point of
# this module. The obvious version of "no repo fallback" -- simply not listing
# the repo among the roots -- is not enough, and here is the exact way it
# fails:
#
#     %cd /kaggle/working
#     !git clone .../cricket-auction-strategy.git
#
# The clone now sits INSIDE /kaggle/working, which is a legitimate search root
# (it is where this session's builds go) and is searched BEFORE /kaggle/input
# so that a fresh build beats a stale mount. So the walk descends into the
# clone, finds data/raw/shotquality, and cheerfully reports the repo as the
# "inputs dataset" -- while the real, mounted dataset sits there unread. Every
# path resolves, nothing errors, and the run is silently reading a copy whose
# version nobody recorded.
#
# That is not hypothetical. It happened on the first real Kaggle run.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# The datasets this project needs
# ---------------------------------------------------------------------------

DATASETS = {
    # `marker` is a directory name that exists ONLY in that dataset. Mounts
    # are recognised by it rather than by slug, because Kaggle's mount path
    # depends on how the dataset was attached and changes on rename -- and a
    # rename should not break a run.
    "inputs": {
        "slug": "udvasbasak2/ipl-auction-model-inputs",
        "marker": "shotquality",
        "holds": "cricsheet zips, shot-quality feeds, curated CSVs",
        "required": True,
    },
    "auction": {
        "slug": "udvasbasak2/ipl-auction-trail-data",
        "marker": "completed_players",
        "holds": "scraped auction trail, completed players, earnings",
        "required": True,
    },
    "hawkeye": {
        "slug": "rhitankar21/t20-data-for-auction-project",
        "marker": "hawkeye_ipl2026",
        "holds": "IPL 2026 Hawkeye ball tracking (optional)",
        "required": False,
    },
}


class DataNotFound(FileNotFoundError):
    """Raised with the dataset slug to attach, not just a missing path."""


def _attach_hint(dataset=None):
    if dataset:
        d = DATASETS[dataset]
        return f"\nAttach the Kaggle dataset  {d['slug']}  ({d['holds']})."
    slugs = "\n  ".join(f"{d['slug']:48s} {d['holds']}"
                        for d in DATASETS.values())
    return ("\nAttach the Kaggle datasets this project reads:\n  " + slugs +
            f"\n\nOr set {ENV_OVERRIDE} to a directory that contains the file.")


# ---------------------------------------------------------------------------
# Search roots
# ---------------------------------------------------------------------------

def _inside_repo(path):
    """True if `path` is the repo directory or anything under it."""
    try:
        p = os.path.realpath(path)
        r = os.path.realpath(REPO_ROOT)
    except OSError:
        return False
    return p == r or p.startswith(r + os.sep)


def _env_roots():
    env = os.environ.get(ENV_OVERRIDE)
    return [p for p in (env.split(os.pathsep) if env else []) if p]


def _kaggle_roots():
    """Every mounted Kaggle input directory."""
    if not os.path.isdir(KAGGLE_INPUT):
        return []
    roots = []
    for pattern in (
        os.path.join(KAGGLE_INPUT, "datasets", "*", "*"),
        os.path.join(KAGGLE_INPUT, "*"),
    ):
        roots.extend(sorted(p for p in glob.glob(pattern) if os.path.isdir(p)))
    # /kaggle/input/datasets itself is a container, not a dataset.
    return [r for r in roots if os.path.basename(r) != "datasets"]


def data_roots(extra=None):
    """
    Ordered search path, most-specific first.

    Explicit argument, then CRICKET_DATA_DIR, then /kaggle/working, then every
    mounted Kaggle dataset.  /kaggle/working comes before /kaggle/input on
    purpose: a file rebuilt this session is the one the caller means.
    """
    roots = []
    if extra:
        roots.extend([extra] if isinstance(extra, str) else list(extra))
    roots.extend(_env_roots())
    if os.path.isdir(KAGGLE_WORKING):
        roots.append(KAGGLE_WORKING)
    roots.extend(_kaggle_roots())

    seen, out = set(), []
    for r in roots:
        if not r:
            continue
        r = os.path.abspath(r)
        # A root that IS the repo, or lives inside it, is dropped outright.
        # See the REPO_ROOT comment above.
        if _inside_repo(r):
            continue
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def _walk(root, max_depth=MAX_DEPTH):
    """Yield (dirpath, dirnames, filenames) under root, depth-capped."""
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    # followlinks=True: a Kaggle mount, and any local staging directory you
    # point CRICKET_DATA_DIR at, may be a symlink. os.walk skips those by
    # default, which presents as "dataset NOT ATTACHED" with the dataset
    # plainly sitting there. Depth is capped below, so there is no runaway
    # even if a link points at an ancestor.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _SKIP_DIRS
            and not _inside_repo(os.path.join(dirpath, d))
        )
        if dirpath.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
        yield dirpath, dirnames, filenames


# ---------------------------------------------------------------------------
# Resolving one dataset
# ---------------------------------------------------------------------------

def dataset_root(name, required=True):
    """
    The mount point of a named dataset, identified by its marker directory.

    Returns the directory that CONTAINS the marker, so callers can go on to
    look for siblings of it.  The marker is used rather than the slug because
    Kaggle's mount name depends on how the dataset was attached and on any
    rename, and a rename should not break a run.
    """
    spec = DATASETS[name]
    marker = spec["marker"]
    for root in data_roots():
        for dirpath, dirnames, _ in _walk(root):
            if marker in dirnames:
                return dirpath
    if not required:
        return None
    raise DataNotFound(
        f"could not find the {name!r} dataset (looked for a directory "
        f"{marker!r}).\nSearched:\n  "
        + "\n  ".join(data_roots() or ["(nothing mounted)"])
        + _attach_hint(name)
    )


# ---------------------------------------------------------------------------
# Resolving files and directories
# ---------------------------------------------------------------------------

def find_file(name, extensions=None, extra_roots=None, required=True,
              dataset=None):
    """
    Resolve a logical file name to a path.

    `name` may carry an extension or not.  Without one, `extensions` is tried
    in order, so the same call finds a parquet in one dataset and a CSV in
    another.  Every root is walked to MAX_DEPTH, because Kaggle datasets nest.

    Preference between candidate extensions beats preference between roots --
    one full pass per candidate -- otherwise a CSV in the first-listed dataset
    shadows the parquet of the same table in the second, which is slower to
    read and loses the column types.
    """
    exts = list(extensions or DATA_EXTENSIONS)
    stem, ext = os.path.splitext(name)
    candidates = [name] if ext else [stem + e for e in exts]

    roots = data_roots(extra_roots)
    for cand in candidates:
        target = cand.lower()
        for root in roots:
            direct = os.path.join(root, cand)
            if os.path.isfile(direct):
                return direct
            for dirpath, _, filenames in _walk(root):
                for f in filenames:
                    if f.lower() == target:
                        return os.path.join(dirpath, f)

    if not required:
        return None
    raise DataNotFound(
        f"could not find {name!r} (tried {', '.join(candidates)}).\n"
        f"Searched:\n  " + "\n  ".join(roots or ["(nothing mounted)"])
        + _attach_hint(dataset)
    )


def find_dir(name, extra_roots=None, required=True, contains=None,
             dataset=None):
    """
    Resolve a directory by name.

    `contains` names a file that must be present inside it, which is what
    stops an empty `bbb/` created by a previous half-finished run from being
    returned in preference to the real dataset.
    """
    for root in data_roots(extra_roots):
        if os.path.basename(root) == name and (
                not contains or os.path.isfile(os.path.join(root, contains))):
            return root
        for dirpath, dirnames, _ in _walk(root):
            for d in dirnames:
                if d != name:
                    continue
                cand = os.path.join(dirpath, d)
                if contains and not os.path.isfile(os.path.join(cand, contains)):
                    continue
                return cand
    if not required:
        return None
    raise DataNotFound(
        f"could not find a directory {name!r}"
        + (f" containing {contains!r}" if contains else "")
        + ".\nSearched:\n  "
        + "\n  ".join(data_roots(extra_roots) or ["(nothing mounted)"])
        + _attach_hint(dataset)
    )


# ---------------------------------------------------------------------------
# The named things this pipeline actually asks for
# ---------------------------------------------------------------------------

def bbb_dir(extra_roots=None, required=True):
    """
    The build_bbb output set: deliveries / fielding / people / wickets /
    matches parquet.

    This is BUILT, not stored. It normally lives at /kaggle/working/bbb and
    is produced by `python -m pipelines.build_bbb` earlier in the session.
    No dataset carries it, because it is fully derived from the cricsheet
    zips in `inputs` and a stored copy is one more thing to keep in sync.

    `contains=` matters here. build_shot_attributes writes
    ball_attributes.parquet into that same /kaggle/working/bbb, and on a
    session where build_bbb has NOT run yet that leaves a directory called
    `bbb` holding none of the five tables. Without the check it would win the
    search -- /kaggle/working is searched first -- and every delivery would
    silently disappear.
    """
    got = find_dir("bbb", extra_roots, required=False,
                   contains="deliveries.parquet")
    if got or not required:
        return got
    raise DataNotFound(
        "no bbb parquet set found (looked for a directory 'bbb' containing "
        "deliveries.parquet).\n"
        "It is built, not downloaded. Run this first, in the same session:\n"
        "    python -m pipelines.build_bbb\n"
        "which reads the cricsheet zips from "
        f"{DATASETS['inputs']['slug']} and writes to "
        f"{os.path.join(KAGGLE_WORKING, 'bbb')}.\n"
        "Searched:\n  " + "\n  ".join(data_roots(extra_roots)
                                       or ["(nothing mounted)"])
    )


def resolution_path(required=False):
    """
    The hand-verified cricinfo identity cache.

    Optional by default: the pipeline runs without it, just worse -- every
    identity it settles goes back to being guessed from name tiers.
    `describe()` says whether it was found, and it should always be found.
    """
    return find_file("cricinfo_resolution.csv", required=required,
                     dataset="inputs")


def archetypes_path(required=True):
    """The curated player archetype table (24 boolean tags per player)."""
    return find_file("player_archetypes.csv", required=required,
                     dataset="inputs")


def people_register(required=False):
    """
    Cricsheet's people.csv register.

    Only adds cross-site ids (Cricinfo, CricketArchive) on top of the
    per-match registry, so build_bbb runs without it.
    """
    return find_file("people.csv", extensions=(".csv",), required=required,
                     dataset="inputs")


def shotquality_feeds(required=True):
    """
    (combined, [feeds]) for build_shot_attributes.

    Both feed snapshots are used: the older one carries 129 matches' worth of
    balls the newer one dropped.  The newer is required, the older is a bonus.
    """
    combined = find_file("t20_combined", required=required, dataset="inputs")
    newer = find_file("t20_bbb-updated", required=required, dataset="inputs")
    older = find_file("t20_bbb", required=False, dataset="inputs")
    feeds = [f for f in (newer,) if f]
    if older and newer and os.path.abspath(older) != os.path.abspath(newer):
        feeds.append(older)
    return combined, feeds


def ball_attributes_path(required=False):
    """
    Shot quality per delivery.

    Built, not stored: pipelines/build_shot_attributes.py writes it to
    /kaggle/working/bbb during a session.  Absent, the aggregator reports
    `None` for every shot-quality metric rather than inventing zeros -- so
    this is `required=False` and the caller degrades.
    """
    return find_file("ball_attributes.parquet", required=required,
                     dataset="inputs")


# --- the auction record ----------------------------------------------------

_AUCTION_SETS = {
    "players": ("completed_players", "completed_players_{year}.csv"),
    "bids": ("auction_trail", "auction_trail_{year}.csv"),
    "earnings": ("earnings", "earnings_{year}.csv"),
}


def auction_template(kind, required=True):
    """
    A "{year}"-templated path into the auction dataset.

    `kind` is one of players / bids / earnings.  Returned as a template rather
    than a resolved path because the whole pipeline is keyed on auction year
    and every caller formats it themselves.
    """
    if kind not in _AUCTION_SETS:
        raise KeyError(
            f"kind must be one of {sorted(_AUCTION_SETS)}, got {kind!r}")
    subdir, pattern = _AUCTION_SETS[kind]
    d = find_dir(subdir, required=required, dataset="auction")
    return os.path.join(d, pattern) if d else None


def player_template(required=True):
    """completed_players_{year}.csv -- the roster of every player at auction."""
    return auction_template("players", required)


def bid_template(required=True):
    """auction_trail_{year}.csv -- the bid ladder."""
    return auction_template("bids", required)


def earnings_template(required=False):
    """earnings_{year}.csv."""
    return auction_template("earnings", required)


# --- cricsheet source json -------------------------------------------------

def cricsheet_sources(required=True):
    """
    Everything build_bbb should read, from the `inputs` dataset.

    Returns a list of paths: the 20 `*_json.zip` archives under
    `cricsheet/`, or -- if you ever upload them unzipped instead --
    the directories of match json.  `iter_match_documents` accepts both,
    so nothing has to be extracted into the session first.

    people.csv is deliberately NOT returned here: it is the register, not a
    match source, and it is fetched separately by `people_register()`.

    Why the zips rather than the Hawkeye dataset's unzipped copy: this
    snapshot is newer. It carries Major League Tournament (134 matches, absent
    there entirely) and several hundred more recent matches across t20s, t20
    blast, MLC and LPL. Building from the older copy silently produces a
    smaller delivery table with no error anywhere.
    """
    root = find_dir("cricsheet", required=required, dataset="inputs")
    if not root:
        return []

    zips = sorted(glob.glob(os.path.join(root, "*_json.zip")))
    if zips:
        return zips

    # Fallback: an unzipped upload. Any directory holding match json.
    dirs = []
    for dirpath, _, filenames in _walk(root):
        if any(f.endswith(".json") and f != "README.json" for f in filenames):
            dirs.append(dirpath)
    if dirs:
        return sorted(dirs)

    if required:
        raise DataNotFound(
            f"{root} holds no *_json.zip and no directory of match json."
            + _attach_hint("inputs")
        )
    return []


def hawkeye_dir(required=False):
    """The IPL 2026 Hawkeye ball-tracking feed.  Optional; nothing reads it yet."""
    return find_dir("hawkeye_ipl2026", required=required, dataset="hawkeye")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def output_dir(subdir=None):
    """
    Where this run may write.

    /kaggle/input is read-only, so anything built during a session goes to
    /kaggle/working, which is also what Kaggle offers for download at the end.
    Off Kaggle it falls back to ./outputs relative to the current working
    directory -- never into the repo, which holds no data by construction.
    """
    base = KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else \
        os.path.abspath(os.environ.get("CRICKET_OUTPUT_DIR", "outputs"))
    path = os.path.join(base, subdir) if subdir else base
    if _inside_repo(path):
        raise ValueError(
            f"refusing to write data into the repository ({path}). "
            f"Set CRICKET_OUTPUT_DIR to a directory outside it."
        )
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def _duckdb_at_least(major, minor):
    try:
        import duckdb
        parts = duckdb.__version__.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


def _csv_options():
    """
    Explicit CSV dialect, so nothing depends on the sniffer guessing right.

    DuckDB auto-detects delimiter, quote and escape by searching a candidate
    space, and when that search fails on a multi-file read it raises
    `Error when sniffing file ""` -- with an empty filename, because the
    multi-file reader does not know which of the files it was working on.
    That is unactionable.  These files are ordinary comma-separated exports,
    so the dialect is simply stated and there is no search to fail.

    `sample_size` is deliberately left at the default.  An earlier version
    forced -1 to stop sparse columns being typed as all-NULL, but a full-file
    check showed the default sniffer already recovers every one of the
    1,069,790 `control` values, and -1 makes the sniffer scan the whole file
    before reading a row of it.

    `null_padding` is deliberately NOT set.  It looks like free robustness
    against ragged rows, and DuckDB's own error message recommends it, but it
    is rejected by the parallel scanner whenever a file contains a newline
    inside a quoted field -- which these do, in the commentary text columns.
    Setting it turns a file that reads fine into `CSV Error on Line: 1807`.
    """
    opts = [
        "header=true",
        "delim=','",
        "quote='\"'",
        "escape='\"'",
    ]
    if _duckdb_at_least(1, 2):
        # Tolerates rows that do not strictly comply with RFC 4180 --
        # unescaped quotes mid-field, which hand-assembled exports collect.
        opts.append("strict_mode=false")
    return ", ".join(opts)


def scan(path, csv_kwargs=None):
    """
    A DuckDB table expression for `path`, whatever its format.

    A list is expanded into one read per file combined with UNION ALL BY NAME
    rather than handed to `read_csv([...])`.  It costs nothing and it means a
    failure names the file that caused it instead of reporting `""`.
    """
    if isinstance(path, (list, tuple)):
        parts = [f"SELECT * FROM {scan(p, csv_kwargs)}" for p in path]
        return "(" + " UNION ALL BY NAME ".join(parts) + ")"

    p = str(path)
    if p.endswith(".parquet"):
        return f"read_parquet('{p}')"
    opts = _csv_options()
    if csv_kwargs:
        opts += ", " + ", ".join(f"{k}={v}" for k, v in csv_kwargs.items())
    return f"read_csv('{p}', {opts})"


def probe(path, con=None, min_columns=10):
    """
    Try to read one file's header and first rows.

    Returns (ok, message).  Run this over each input before a long build: a
    dialect failure surfaces here, named, in a second, instead of thirty
    seconds into stage 1 with an empty filename attached to it.
    """
    import duckdb
    con = con or duckdb.connect()
    try:
        rel = con.sql(f"SELECT * FROM {scan(path)} LIMIT 5")
        cols = len(rel.columns)
        # count(*) over the sample rather than fetching it: pulling real
        # values into Python drags in the timestamp conversion path, and a
        # probe that fails because pytz is missing tells you nothing about
        # the file.
        n = con.sql(
            f"SELECT count(*) FROM (SELECT * FROM {scan(path)} LIMIT 5)"
        ).fetchone()[0]
        # A file whose dialect was misread does not raise -- it parses as one
        # giant column. That is the failure mode worth catching, because it
        # survives all the way to a confusing join error much later.
        if cols < min_columns:
            return False, (f"MISPARSED: {cols} column(s), expected at least "
                           f"{min_columns} -- wrong delimiter or encoding")
        if n == 0:
            return False, f"EMPTY: {cols} columns but no rows"
        return True, f"ok: {cols} columns, read {n} sample rows"
    except Exception as exc:
        first = str(exc).strip().split("\n")[0]
        return False, f"UNREADABLE: {first}"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe():
    """
    Print what is visible.  First thing to run when a path fails, and the
    first cell of any Kaggle notebook using this repo.
    """
    print(f"{ENV_OVERRIDE:20s}: {os.environ.get(ENV_OVERRIDE) or '(unset)'}")
    print(f"{'repo root':20s}: {REPO_ROOT}  (EXCLUDED from all searches)")
    print(f"{'output dir':20s}: {output_dir()}")

    stale = os.path.join(REPO_ROOT, "data")
    if os.path.isdir(stale):
        n = sum(len(f) for _, _, f in os.walk(stale))
        print(f"\n  WARNING: {stale} still exists and holds {n} files.")
        print("  It is ignored -- nothing below reads from it -- but it means")
        print("  the history purge has not run yet, so every clone still")
        print("  drags ~231 MB. See scripts/purge_data_from_history.sh.")

    print("\nsearch roots (in order):")
    for r in data_roots() or ["(nothing mounted)"]:
        print(f"  {r}")

    print("\ndatasets:")
    for name, spec in DATASETS.items():
        root = dataset_root(name, required=False)
        print(f"  {name:10s} {spec['slug']:48s} {root or 'NOT ATTACHED'}")

    print("\nresolved paths:")
    checks = [
        ("bbb dir", lambda: bbb_dir(required=False)),
        ("ball_attributes", ball_attributes_path),
        ("cricinfo_resolution", resolution_path),
        ("player_archetypes", lambda: archetypes_path(required=False)),
        ("cricsheet people.csv", people_register),
        ("players template", lambda: player_template(required=False)),
        ("bids template", lambda: bid_template(required=False)),
        ("earnings template", earnings_template),
        ("t20_combined", lambda: find_file("t20_combined", required=False)),
        ("t20_bbb-updated", lambda: find_file("t20_bbb-updated", required=False)),
        ("t20_bbb", lambda: find_file("t20_bbb", required=False)),
    ]
    for label, fn in checks:
        try:
            got = fn()
        except Exception as exc:  # noqa: BLE001
            got = f"ERROR: {type(exc).__name__}"
        print(f"  {label:22s} {got or 'MISSING'}")

    try:
        srcs = cricsheet_sources(required=False)
    except Exception:  # noqa: BLE001
        srcs = []
    print(f"  {'cricsheet sources':22s} {len(srcs)} files "
          f"(expect 20 zips)")

    if not bbb_dir(required=False):
        print("\nbbb is NOT built in this session. Run:\n"
              "    python -m pipelines.build_bbb\n"
              "    python -m pipelines.build_shot_attributes")


if __name__ == "__main__":
    describe()