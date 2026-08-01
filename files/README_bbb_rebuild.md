# Rebuilt ball-by-ball pipeline

Replaces `t20-men.parquet` and `input_creation_2/player_features/player_features.py`.
Scope is deliberately narrow: **get the statistics right**. Auction ordering, RTM
semantics, retentions, purse accounting and schema plumbing are untouched.

```
data_prep/build_bbb.py                              Cricsheet JSON -> parquet
input_creation_2/player_features/player_features.py as-of-date statistics
input_creation_2/player_features/identity.py        auction roster -> person_id
tests/test_player_features.py                       25 tests, all hand-verified
```

## 1. Build the data

```bash
pip install requests pyarrow pandas numpy
python -m data_prep.build_bbb --download --out-dir data/bbb
```

Downloads the men's T20 sources (T20Is, non-official T20Is, IPL, BBL, T20 Blast,
Syed Mushtaq Ali, BPL, CPL, PSL, CSA T20, Super Smash, LPL, ILT20, SA20, MLC,
MLT, MSL, NPL, Ireland Inter-Pro T20, Major Clubs T20) and writes:

| file | contents |
|---|---|
| `deliveries.parquet` | one row per ball, 49 columns |
| `matches.parquet` | one row per match |
| `people.parquet` | `person_id`, canonical name, **all name variants** |
| `wickets.parquet` | one row per wicket (a ball can carry two) |
| `fielding.parquet` | one row per (wicket, fielder) |

The Hundred is excluded by default — 5-ball overs make its balls and
over-based phases non-comparable. Add `--include-hundred` if you want it.
Super-over deliveries are dropped by default (`--keep-super-overs` to retain).

`build()` ends with a sanity check that raises rather than writing a bad file:
null dates, null player ids, duplicate `(match, innings, ball)`, a wide marked
as faced, a no-ball marked legal, and `runs_total != batter + all extras`.

## 2. Compute statistics

```python
import pandas as pd
from input_creation_2.player_features.player_features import (
    PlayerStatsAggregator, PlayerFeatureBuilder, feature_columns)

deliveries = pd.read_parquet("data/bbb/deliveries.parquet")
fielding   = pd.read_parquet("data/bbb/fielding.parquet")

agg = PlayerStatsAggregator(deliveries, fielding=fielding)
# IPL-only view, if you want one:
# agg_ipl = PlayerStatsAggregator(deliveries, competitions=["Indian Premier League"])

builder = PlayerFeatureBuilder(agg)
features = builder.build_feature_table(roster, "2018-01-27")   # roster has player_id
```

`build_feature_table` returns exactly one row per distinct id and asserts it.
91 numeric columns: raw counts, derived metrics, powerplay/middle/death splits,
fielding, and a `*_is_missing` indicator beside every metric that can be
undefined.

**Undefined is never zero.** A player who has not bowled gets
`economy = None` from `get_player_stats` and `bowl_economy = 0.0` **plus**
`bowl_economy_is_missing = 1.0` in the flattened table, so the model can learn
a separate offset for "no data" instead of reading the fill as a measurement.

**The as-of cutoff is strict.** A match played *on* the auction date is
excluded. This is tested (`test_as_of_date_is_strict`).

## 3. Bridge the auction roster

Your auction CSVs carry Cricbuzz names and ids; the ball data carries Cricsheet
person ids. One join remains, and `identity.py` makes it explicit:

```python
from input_creation_2.player_features.identity import PlayerIdentityResolver

people = pd.read_parquet("data/bbb/people.parquet")
resolver = PlayerIdentityResolver(people, overrides="data/name_overrides.csv")
roster = resolver.resolve(player_df)          # adds person_id, match_method
print(resolver.report(roster))
```

Matching runs against **every name variant**, not just a canonical form — that
alone fixes `Lokesh Rahul`. Anything ambiguous or unmatched is left `None` and
reported rather than guessed; put those in `data/name_overrides.csv`
(`playerId, person_id, action` where action is `map` or `block`) and re-run.
An unresolved player still gets a row, flagged `has_history = 0`.

## What the rules now are

| rule | treatment |
|---|---|
| wide | not faced, not a legal ball, **charged to the bowler** |
| no-ball | **faced**, not a legal ball, charged to the bowler |
| byes / leg-byes | faced, legal, **never charged to the bowler** |
| run out | dismissal for the batter, **no wicket for the bowler** |
| run out at non-striker's end | still a dismissal, on a ball they never faced |
| retired hurt | **not an out** |
| retired out | an out |
| 4/6 with `non_boundary` | counts as runs, **not** as a boundary |
| batting dismissal | from `player_out`, never from "was on strike" |
| bowler's wicket | bowled, caught, caught and bowled, lbw, stumped, hit wicket |
| batter dot | ball faced, no run off the bat |
| bowler dot | legal ball, nothing charged to the bowler |
| maiden | an over conceding nothing |

## Verification

```bash
python -m pytest tests/test_player_features.py -q      # 25 passed
```

Two synthetic Cricsheet matches, written to the real schema, cover every rule
above plus the two identity failure modes: one person under two names, two
people under one name.

Beyond the fixtures I cross-checked the cumulative-sum aggregator against a
brute-force filter-then-sum on randomly generated deliveries —
**480 (player, date) pairs across 15 quantities, zero mismatches** — and ran it
at realistic scale:

```
2,640,000 deliveries
aggregator build        7.3 s      peak RSS 1.8 GB (incl. 1.2 GB source frame)
2,000 as-of queries     106 ms     (53 us each)
feature table 400x91    0.02 s
```

For comparison, the old aggregator took 40 s to group a 16-column subset and
ran out of memory on the full frame.

## Design notes

Everything is keyed on `person_id`; names are display labels only.

Deliveries fold once to player-innings (~2.6M rows becomes ~120k batting +
~66k bowling rows), then each player's innings are held in date order with
running totals. "Career to date D" is one `searchsorted` plus an array
index — no filtering, no copying, and provably identical to
filter-then-sum. Folding uses integer group codes and `np.bincount` rather
than pandas groupby; at a few million rows the string columns alone are the
difference between fitting in memory and not.

## Two things to decide

**Competition scope.** The aggregator currently weights a Syed Mushtaq Ali
innings the same as an IPL one. `competitions=` lets you restrict it, but the
better answer is probably two aggregators — one IPL-only, one all-T20 — with
their outputs prefixed and concatenated, so the model can weigh them itself.

**Phase boundaries.** `PHASES` uses powerplay = overs 1–6, middle = 7–15,
death = 16–20. The 16-vs-17 death boundary is a convention; it is one constant
at the top of both files if you want to move it.
