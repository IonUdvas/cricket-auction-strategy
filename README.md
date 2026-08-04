# cricket-auction-strategy

A valuation model for IPL auction bidding, trained on ball-by-ball T20 data
replayed against historical auction trails.

**This repository contains no data.** Not ignored — absent. Every byte lives in
a Kaggle Dataset, and no code path resolves a data file against the repo, so a
file dropped into the working tree is never read even if it is never committed.
`data_sources.py` is the only module that knows where anything is.

---

## The data

| Dataset | Required | Holds |
|---|---|---|
| `udvasbasak2/ipl-auction-model-inputs` | yes | Cricsheet zips, shot-quality feeds, curated CSVs |
| `udvasbasak2/ipl-auction-trail-data` | yes | Auction trail, completed players, earnings |
| `rhitankar21/t20-data-for-auction-project` | no | IPL 2026 Hawkeye ball tracking |

### `ipl-auction-model-inputs` layout

```
cricsheet/      20 × *_json.zip          the Cricsheet men's T20 match feed
                people.csv               the Cricsheet register
shotquality/    t20_bbb.parquet
                t20_bbb-updated.parquet
                t20_combined.parquet
identity/       cricinfo_resolution.csv  hand-verified identity decisions
auction/        player_archetypes.csv    24 curated tags per player
```

The directory names are load-bearing: `data_sources` recognises the mount by
looking for `shotquality/`, and `cricsheet_sources()` globs `cricsheet/*.zip`.

### Built, not stored

Two artifacts are derived and rebuilt each session into `/kaggle/working/bbb`:

| Artifact | Built by | Cost |
|---|---|---|
| `deliveries/matches/people/wickets/fielding.parquet` | `pipelines/build_bbb.py` | ~3–6 min |
| `ball_attributes.parquet` | `pipelines/build_shot_attributes.py` | ~2–4 min |

They are not stored because a stored copy is a second thing to keep in sync
with the zips it came from, and the failure mode when it drifts is silent — a
delivery table quietly short by a season, giving wrong career totals for
exactly the players who played most recently.

`data_sources` searches `/kaggle/working` **before** `/kaggle/input`, so if you
do choose to cache `bbb` as its own dataset, a fresh in-session build still
wins over the stored copy. Keep it in a separate dataset from the sources.

---

## Running on Kaggle

Attach the two required datasets, then:

```python
!git clone -q https://github.com/IonUdvas/cricket-auction-strategy.git
%cd cricket-auction-strategy

from kaggle_session import prepare
paths = prepare()          # describes the mounts, builds bbb + shot quality
```

`prepare()` is idempotent — re-running the cell skips any stage whose output
already exists.

```python
import pandas as pd
from src.training import run_training_pipeline_with_holdout

out = run_training_pipeline_with_holdout(
    train_years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    val_years=[2026],
    player_role_df=pd.read_csv(paths["archetypes"]),
)
```

No paths are passed. Every default resolves through `data_sources`.

When something is missing, the error names the dataset to attach rather than
the path it wanted. `python -m data_sources` prints the whole picture.

### Running off Kaggle

Set `CRICKET_DATA_DIR` to a directory holding the same layout. It accepts
several paths separated by `:`. This is the only way to read data that is not
on Kaggle, and it has to be set deliberately:

```bash
CRICKET_DATA_DIR=/local/kaggle-mirror python -m pipelines.build_bbb
```

---

## Layout

```
data_sources.py       the only module that resolves a data path
kaggle_session.py     one-call session bootstrap
configs/default.yaml  hyperparameters (code, not data — it stays in the repo)
pipelines/            code that BUILDS data. Holds no data.
  build_bbb.py                  Cricsheet json -> the five parquet tables
  build_shot_attributes.py      shot-quality feeds -> ball_attributes.parquet
  identity/                     tools for curating cricinfo_resolution.csv
input_creation_2/     replay engine, player features, identity resolution
valuation_model/      the model, losses, scaling, training loop
src/                  training entry points, checks, multi-seed experiments
tests/                pytest suite + standalone audit scripts
```

`pipelines/` was called `data/`. It was renamed because a directory called
`data` is where data ends up, whatever the intent.

---

## Notes on the data itself

- The Cricsheet zips here are a **newer snapshot** than the unzipped copy in
  `rhitankar21/t20-data-for-auction-project`: +183 T20 internationals, +115
  T20 Blast, +33 MLC, +16 LPL. Build from these, not from there.
- `mlt_male_json.zip` (Major League Tournament, 134 matches) is absent from
  that dataset entirely — but all 134 are dropped by `match_passes_filter`
  anyway, so it contributes zero deliveries. It is uploaded for completeness.
- A full build over all 20 zips needs several GB of RAM. Kaggle is fine; a
  small container is not.
