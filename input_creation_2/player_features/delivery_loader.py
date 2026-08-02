"""
Load the delivery table with shot-quality attributes merged in.

`PlayerStatsAggregator` treats shot quality as optional: given a bare
build_bbb frame it returns exactly what it always returned, and given an
enriched one it additionally reports control %, false shot % and aerial %.
This module is the thing that produces the enriched frame, and it exists
separately so that the merge -- which is where a left join can silently
become an inner one and delete two thirds of the deliveries -- happens in
one place with one assertion on it.
"""

from __future__ import annotations

import os

import pandas as pd

# Only the columns the aggregator folds.  ball_attributes carries another
# dozen (shot type, line, length, wagon wheel, fielding position) that are
# useful for analysis and are deliberately not loaded here: they would
# multiply the memory cost of the delivery frame for features nothing
# currently consumes.
ATTRIBUTE_COLUMNS = (
    "is_controlled_wide", "has_control",
    "is_aerial", "has_elevation",
)

KEYS = ("match_id", "innings", "ball_seq")


def load_deliveries(bbb_dir=None, with_shot_quality=True, columns=None,
                    attributes_path=None):
    """
    Parameters
    ----------
    bbb_dir : str, optional
        Directory holding deliveries.parquet.  Resolved via `data_sources`
        when omitted, which finds it under /kaggle/input or in the repo.
    with_shot_quality : bool
        Merge ball_attributes if it can be found.  When it cannot, this is a
        no-op and the caller gets the plain delivery table -- the aggregator
        degrades to `None` for every shot-quality metric rather than
        inventing zeros.
    columns : list of str, optional
        Restrict the delivery columns read.  The aggregator's required set
        plus the merge keys must be included.
    attributes_path : str, optional
        Explicit path to ball_attributes.parquet.  It is resolved separately
        from `bbb_dir` because it is *built* rather than downloaded, so on
        Kaggle it usually arrives as its own dataset while deliveries comes
        from the original bbb one.  Assuming the two live together is the
        obvious shortcut and it silently drops shot quality on exactly the
        setup this pipeline runs in.
    """
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import data_sources as ds

    bbb_dir = bbb_dir or ds.bbb_dir()
    deliveries = pd.read_parquet(
        os.path.join(bbb_dir, "deliveries.parquet"), columns=columns)
    if not with_shot_quality:
        return deliveries

    path = attributes_path or os.path.join(bbb_dir, "ball_attributes.parquet")
    if not os.path.exists(path):
        path = ds.find_file("ball_attributes.parquet", required=False)
    if not path:
        return deliveries

    attrs = pd.read_parquet(path, columns=list(KEYS) + list(ATTRIBUTE_COLUMNS))

    # ball_attributes holds at most one row per delivery by construction
    # (it is a positional join onto this same table), but "by construction"
    # is what every duplicated-key bug has been described as, so check.
    dupes = attrs.duplicated(list(KEYS)).sum()
    if dupes:
        raise ValueError(
            f"ball_attributes has {dupes} duplicated {KEYS} rows; the merge "
            f"below would multiply deliveries. Rebuild with "
            f"data/build_shot_attributes.py."
        )

    before = len(deliveries)
    merged = deliveries.merge(attrs, on=list(KEYS), how="left")
    if len(merged) != before:
        raise ValueError(
            f"merge changed the delivery count {before} -> {len(merged)}"
        )

    # Coverage flags must be 0 rather than NaN on unmatched deliveries: they
    # are denominators, and a NaN denominator propagates into every phase
    # total it is summed into.
    for flag in ("has_control", "has_elevation"):
        merged[flag] = merged[flag].fillna(0).astype("int8")
    return merged


def coverage_report(deliveries):
    """Per-competition shot-quality coverage, for sanity-checking a build."""
    if "has_control" not in deliveries.columns:
        return pd.DataFrame()
    g = deliveries.groupby("competition")
    out = pd.DataFrame({
        "balls": g.size(),
        "control_pct": 100 * g["has_control"].mean(),
        "elevation_pct": 100 * g["has_elevation"].mean(),
    })
    return out.sort_values("balls", ascending=False).round(1)
