# CHANGES

Two commits were missing from your codebase. Everything else — the
demographics wiring, the verification wiring, the config keys, the
`verification.py` docstring correction — you had already applied
correctly, and I found no conflicts.

Applying either group by hand is mechanical. Line numbers below are
*after* all edits are applied; apply top-down within each file and they
will hold.

---

# GROUP A — Archetype wiring (the substantive one)

## Why

`AuctionReplayEngine.__init__` has accepted `archetype_df=None` since the
archetype counters were written. **No call site anywhere in the repo ever
passed it**, and `build_training_samples` did not even take the argument,
so `self.archetype_tags` was `None` on every build. The engine then took
its documented fallback, quoted from its own constructor comment:

> Optional: without the table the engine keeps emitting the legacy
> BATTER/BOWLER/ALL-ROUNDER counters and nothing downstream changes.

Nothing downstream changed. Every model trained to date saw three coarse
role counters instead of the twelve-archetype supply/demand/scarcity
block — the granularity `archetypes.py` was written specifically to
replace, and the granularity the writeup claims for `S_t^(r)` and
`A^(r)`.

Measured on a synthetic three-team auction:

| block | before | after |
|---|---|---|
| `team_state` | 20 columns | 59 (`*_bought`, `*_needed` per archetype) |
| `auction_state` | 20 columns | 56 (`*_remaining`, `*_demand`, `*_scarcity`) |

## A1. `input_creation_2/auction_dataset_utils.py`

**A1a — add the parameter** (~line 591). In the signature of
`build_training_samples`, after `player_context_columns=None,`:

```python
    archetype_df=None,
```

**A1b — pass it to the engine** (~line 647). Find:

```python
    engine = AuctionReplayEngine(
        bid_df=bid_df,
        player_df=player_df,
        auction_max_purse=auction_max_purse,
    )
```

Replace with (comment included — this is the fact worth preserving):

```python
    ####################################################################
    # archetype_df is what switches the engine from three coarse role
    # counters (batters/bowlers/allrounders bought) to the twelve
    # squad-construction archetypes, and it is what makes the
    # supply/demand/scarcity block exist at all. Without it the engine
    # silently falls back to the legacy counters -- which is how it ran
    # for the whole history of this pipeline, because no call site ever
    # passed the table.
    #
    # It must be the RAW archetype table, not the filtered role frame
    # handed to build_role_table as player_role_df. build_archetype_tags
    # needs `pace`, `RA` and `LA` to derive right_arm_pace/left_arm_pace,
    # and the filtered frame drops them. Passing the wrong one raises
    # rather than silently degrading, which is the intent.
    ####################################################################

    engine = AuctionReplayEngine(
        bid_df=bid_df,
        player_df=player_df,
        auction_max_purse=auction_max_purse,
        archetype_df=archetype_df,
    )
```

## A2. `src/training.py`

**A2a — new loader** (~line 195), immediately after
`default_bid_template()`:

```python
def default_archetype_df():
    """
    The RAW player_archetypes.csv, for the replay engine.

    Distinct from the `player_role_df` passed to build_role_table. That
    one is filtered down to the tag columns that become the model's role
    vector; this one must still carry `pace`, `RA` and `LA`, because
    build_archetype_tags derives right_arm_pace / left_arm_pace from
    their conjunction and asserts that every paced player has exactly
    one arm set. Hand it the filtered frame and it raises.

    Loaded by default rather than left to the caller: the engine's
    archetype_df argument has existed since the archetype counters were
    written and no call site ever passed it, so every model to date
    trained on the three legacy role counters instead. A default that
    has to be opted out of fails in the safer direction.
    """
    return pd.read_csv(ds.archetypes_path())
```

**A2b — `build_training_df` signature** (~line 248). Add after
`player_context_columns=None,`:

```python
        archetype_df=None,
```

**A2c — resolve the default** (~line 349), immediately before
`data_cfg = config.get("data", {}) or {}`:

```python
    ####################################################################
    # The archetype table defaults to loading rather than to None.
    #
    # None is what every call site has effectively passed since the
    # archetype counters were written, and the engine treats it as
    # "use the legacy three role counters" without complaint. That
    # default is the reason the supply/demand/scarcity block has never
    # appeared in a training frame. Pass archetype_df=False to opt out
    # deliberately -- distinct from None, which now means "load it".
    ####################################################################

    if archetype_df is None:
        archetype_df = default_archetype_df()
    elif archetype_df is False:
        archetype_df = None
```

**A2d — forward it** (~line 371). In the `build_training_samples(...)`
call inside the per-year loop, after `player_context_columns=...`:

```python
            archetype_df=archetype_df,
```

**A2e — `run_training_pipeline`** (~line 473). Add `archetype_df=None,`
to the signature after `player_role_df=None,`, then change (~line 480):

```python
    full_training_df = build_training_df(player_template, bid_template, bbb_dir)
```

to:

```python
    full_training_df = build_training_df(
        player_template, bid_template, bbb_dir, archetype_df=archetype_df,
    )
```

**A2f — `prepare_holdout_data`** (~line 573). Add `archetype_df=None,` to
the signature after `player_role_df=None,`. Then both
`build_training_df` calls (~lines 646–656) gain
`archetype_df=archetype_df`:

```python
    # archetype_df goes to BOTH splits or the team/auction state blocks
    # come out different widths and the cross-split attrs check below
    # fires. It is threaded explicitly rather than left to each call's
    # default so that opting out (archetype_df=False) opts both out.
    train_df = build_training_df(
        player_template, bid_template, bbb_dir, years=train_years,
        feature_context=feature_context, archetype_df=archetype_df,
    )

    val_df = build_training_df(
        player_template, bid_template, bbb_dir, years=val_years,
        feature_context=feature_context, archetype_df=archetype_df,
    )
```

**A2g — `run_training_pipeline_with_holdout`** (~line 900). Add
`archetype_df=None,` to the signature after `player_role_df=None,`, and
add `archetype_df=archetype_df,` to the `prepare_holdout_data(...)` call
(~line 946), after `player_role_df=player_role_df,`.

## A3. `src/sweep.py`

In `_build_key`, after the `role_df` block and **before** the final
`return json.dumps(...)`:

```python
    # archetype_df switches the engine between the legacy three role
    # counters and the twelve-archetype supply/demand/scarcity block, so
    # it changes the WIDTH of the team-state and auction-state blocks.
    # Omitting it from the key would let a cache warmed before the
    # archetype wiring serve pre-archetype frames to a sweep that thinks
    # it is measuring them -- a stale-cache bug that looks like a null
    # result rather than like an error.
    arch_df = pipeline_kwargs.get("archetype_df")
    if arch_df is False:
        parts["archetypes"] = "disabled"
    elif arch_df is None:
        parts["archetypes"] = "default"
    else:
        parts["archetypes"] = (tuple(arch_df.columns), len(arch_df))
```

**This one is not optional if you use the sweep.** Any cache warmed
before this change will otherwise serve pre-archetype frames to a sweep
that believes it is measuring them — which reads as a null result, not
as an error.

## A4. `tests/test_archetype_wiring.py` — new file

Copy it from the zip. Six tests, all covering the *join* rather than the
module. `test_archetypes.py` already tested `archetypes.py` thoroughly
and passed throughout the entire period the wiring was dead, because a
unit test of a module nothing calls passes forever.

The tests assert: the state blocks widen when the table is passed; the
legacy counters survive alongside so an ablation is a column selection
rather than a rebuild; multi-label counts do not collapse to a
partition; and the training intervals are byte-identical with and
without archetypes — the interval labels are a property of the bid
ladder and must not move.

---

# GROUP B — Cleanup (cosmetic, but B1 has teeth)

## B1. `.gitignore` — RESTORE IT

Currently absent from the repo entirely. Two consequences:

1. `__pycache__` is unignored, so `git add -A` stages `.pyc` files.
2. Every `data/`, `*.parquet`, `*.csv` guard is gone — the backstop
   behind the history purge you completed.

Copy the 82-line file from the zip. It is the GitHub Python template
trimmed of ~180 lines of boilerplate for frameworks this repo does not
use (Django, Flask, Scrapy, Celery, RabbitMQ, SageMath, Streamlit), with
the hand-written DATA section preserved intact.

## B2. `README.md` — delete two stale lines

In the repository-layout block, remove:

```
scripts/              staging and purge tooling
older_versions/       superseded code, kept for reference
```

`scripts/` is deleted; `older_versions/` does not exist in this repo.

## B3. `data_sources.py` — fix a dangling reference

In `describe()`, the stale-data warning points at a script that no
longer exists. Replace:

```python
        print("  the history purge has not run yet, so every clone still")
        print("  drags ~231 MB. See scripts/purge_data_from_history.sh.")
```

with:

```python
        print("  this clone predates the history purge and still drags the old")
        print("  data. Delete it and re-clone. Do NOT pull into a pre-purge")
        print("  clone: the rewrite gave every commit a new hash, so a pull")
        print("  merges the old history back in rather than replacing it.")
```

---

# Verification

After applying, from the repo root:

```bash
python3 -m pytest tests/ -q          # expect: 49 passed, 1 skipped
```

Then check the wiring is actually live:

```python
import inspect
import src.training as T
from input_creation_2.auction_dataset_utils import build_training_samples as B

assert hasattr(T, "default_archetype_df")
for fn in (T.build_training_df, T.run_training_pipeline,
           T.prepare_holdout_data, T.run_training_pipeline_with_holdout, B):
    assert "archetype_df" in inspect.signature(fn).parameters, fn.__name__
assert "archetype_df=archetype_df" in inspect.getsource(B)
print("archetype wiring live")
```

# What to expect on the first real build

- **Your sweep cache is stale.** The key changed, so it rebuilds rather
  than serving wrong data — but the first `run_seeds` pays full stage-1
  cost.
- **`build_archetype_tags` may raise** on the first year if any paced
  player in the raw table lacks an arm tag, or if any `player_id` is
  duplicated. That is the assertion doing its job, not a wiring failure.
- **Check the scarcity columns before trusting them.** Verify the pool
  counts reconcile against `players_remaining`; that reconciliation is
  the whole reason the `untagged` bucket exists.
- **Nothing in your notebook changes.** The default is to load the table,
  so existing calls pick the wiring up automatically. `archetype_df=False`
  is the explicit opt-out, which is also how you run the ablation.

# For the paper

Block 9 in `revisions.tex` is now unblocked — but only once you have
**retrained**. Your existing `S_t^(r)` and `A^(r)` text becomes true as
written; it needs only the vocabulary cross-reference from block 2. Do
not paste it against results produced before this change.
