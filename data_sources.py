"""
Where the data actually is.

The repo is cloned into a fresh Kaggle session on every run, so nothing that
matters can live inside it: the ball-by-ball feed alone is larger than the
whole codebase, and a git clone that has to drag it along is slow to pull and
impossible to update without a commit.  The data lives in Kaggle Datasets,
which are versioned separately and mounted read-only, and this module is the
single place that knows how to find it.

The problem it solves is that the same logical file sits at a different path
depending on where the code is running, and there is more than one Kaggle
mount convention in the wild:

    /kaggle/input/<slug>/t20_bbb-updated.csv
    /kaggle/input/datasets/<owner>/<slug>/t20_bbb-updated.csv
    <repo>/data/raw/shotquality/t20_bbb-updated.parquet

Rather than encode any of them, `find_file` searches an ordered list of roots
and returns the first hit.  Roots are ordered most-specific-first, so an
explicit override always beats a Kaggle mount, which always beats whatever
happens to be sitting in the working tree -- a stale local copy can never
silently shadow the dataset the run was configured with.

Format is resolved the same way and for the same reason.  The Kaggle copies
are CSV and the local ones are parquet; `find_file("t20_bbb-updated")` takes
no extension and returns whichever exists.  `scan()` then turns that path into
the right DuckDB table function, so the query text does not care either.

Nothing here reads a file.  It resolves paths and raises loudly when it
cannot, because a data-loading failure that surfaces three functions later as
an empty DataFrame is the most expensive kind of bug in this pipeline.
"""

from __future__ import annotations

import glob
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Every extension we know how to read, in preference order.  Parquet first:
# where both exist it is smaller, typed, and an order of magnitude faster to
# scan than the CSV of the same table.
DATA_EXTENSIONS = (".parquet", ".csv", ".csv.gz", ".zip")

# Set CRICKET_DATA_DIR to point a run at a specific directory -- the escape
# hatch for a local rebuild, or for a Kaggle mount this module has not seen.
ENV_OVERRIDE = "CRICKET_DATA_DIR"


def _kaggle_roots():
    """
    Every mounted Kaggle input directory, deepest-usable-level first.

    Kaggle mounts a dataset at /kaggle/input/<slug>, but attaching through the
    newer datasets browser produces /kaggle/input/datasets/<owner>/<slug>.
    Both are globbed rather than picked between, so attaching a dataset either
    way works without a code change.
    """
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return []
    roots = []
    for pattern in (
        os.path.join(base, "datasets", "*", "*"),
        os.path.join(base, "*"),
    ):
        roots.extend(sorted(p for p in glob.glob(pattern) if os.path.isdir(p)))
    # /kaggle/input/datasets itself is a container, not a dataset.
    return [r for r in roots if os.path.basename(r) != "datasets"]


def data_roots(extra=None):
    """Ordered search path: explicit override, Kaggle mounts, then the repo."""
    roots = []
    if extra:
        roots.extend([extra] if isinstance(extra, str) else list(extra))
    env = os.environ.get(ENV_OVERRIDE)
    if env:
        roots.extend(env.split(os.pathsep))
    roots.extend(_kaggle_roots())
    roots.extend([
        os.path.join(REPO_ROOT, "data"),
        os.path.join(REPO_ROOT, "data", "bbb"),
        os.path.join(REPO_ROOT, "data", "raw"),
        os.path.join(REPO_ROOT, "data", "raw", "shotquality"),
        os.path.join(REPO_ROOT, "data", "auction"),
        os.path.join(REPO_ROOT, "data", "identity"),
    ])
    seen, out = set(), []
    for r in roots:
        r = os.path.abspath(r)
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def find_file(name, extensions=None, extra_roots=None, required=True):
    """
    Resolve a logical file name to a path.

    `name` may carry an extension or not.  Without one, `extensions` is tried
    in order, so the same call finds the parquet locally and the CSV on
    Kaggle.  Each root is searched one level deep as well as at the top, since
    a Kaggle dataset often wraps its files in a folder named after itself.
    """
    exts = list(extensions or DATA_EXTENSIONS)
    stem, ext = os.path.splitext(name)
    candidates = [name] if ext else [stem + e for e in exts]

    for root in data_roots(extra_roots):
        for cand in candidates:
            direct = os.path.join(root, cand)
            if os.path.isfile(direct):
                return direct
            hits = sorted(glob.glob(os.path.join(root, "*", cand)))
            if hits:
                return hits[0]

    if not required:
        return None
    raise FileNotFoundError(
        f"could not find {name!r} (tried {', '.join(candidates)}).\n"
        f"Searched:\n  " + "\n  ".join(data_roots(extra_roots)) +
        f"\n\nAttach the Kaggle dataset holding it, or set {ENV_OVERRIDE} to "
        f"a directory that contains it."
    )


def find_dir(name, extra_roots=None, required=True, contains=None):
    """
    Resolve a directory (e.g. the bbb parquet set).

    `contains` names a file that must be present, which is what stops an empty
    `data/bbb` created by a previous half-finished run from being returned in
    preference to a real Kaggle mount.
    """
    for root in data_roots(extra_roots):
        for cand in (os.path.join(root, name), root):
            if not os.path.isdir(cand):
                continue
            if contains and not os.path.isfile(os.path.join(cand, contains)):
                continue
            if not contains and os.path.basename(cand) != name:
                continue
            return cand
    if not required:
        return None
    raise FileNotFoundError(
        f"could not find a directory {name!r}"
        + (f" containing {contains!r}" if contains else "") +
        f".\nSearched:\n  " + "\n  ".join(data_roots(extra_roots))
    )


def bbb_dir(extra_roots=None, required=True):
    """The build_bbb output set, wherever it is mounted."""
    return find_dir("bbb", extra_roots, required, contains="deliveries.parquet")


def output_dir(subdir=None):
    """
    Where this run may write.

    /kaggle/input is read-only, so anything built during a session goes to
    /kaggle/working, which is also what Kaggle offers for download at the end.
    """
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else \
        os.path.join(REPO_ROOT, "data")
    path = os.path.join(base, subdir) if subdir else base
    os.makedirs(path, exist_ok=True)
    return path


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


def _duckdb_at_least(major, minor):
    try:
        import duckdb
        parts = duckdb.__version__.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


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



def describe():
    """Print what is visible.  First thing to run when a path fails."""
    print(f"repo root : {REPO_ROOT}")
    print(f"{ENV_OVERRIDE} : {os.environ.get(ENV_OVERRIDE) or '(unset)'}")
    print(f"output dir: {output_dir()}")
    print("\nsearch roots:")
    for r in data_roots():
        files = sorted(os.listdir(r))[:8]
        print(f"  {r}")
        if files:
            print(f"      {', '.join(files)}"
                  + (" ..." if len(os.listdir(r)) > 8 else ""))