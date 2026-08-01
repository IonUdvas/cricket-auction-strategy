"""
Trace basePrice / cappedStatus / isPlayerOverseas from the frame the
replay engine produces all the way to the tensor the model reads.

No auction CSVs needed: the frame is synthesised to the same contract
build_training_samples produces (same columns, same dtypes, same
attrs), so every stage after the replay engine is the real code.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from input_creation_2.auction_dataset_utils import (
    add_player_context_features,
    build_role_table,
    build_encoders,
)
from input_creation_2.auction_dataset import IPLAuctionDataset
from valuation_model.scaling import fit_scalers
from valuation_model.models import ValuationModel

PLAYER_COLS = ["bat_runs", "bat_strike_rate", "bat_runs_is_missing"]
TEAM_COLS = ["remaining_purse", "players_bought"]
AUCTION_COLS = ["auction_order", "players_remaining"]


def make_frame(n=40, overseas_dtype="bool", capped_style="upper", seed=0):
    rng = np.random.default_rng(seed)

    base = rng.choice([30.0, 75.0, 100.0, 200.0], size=n)

    if overseas_dtype == "bool":
        overseas = rng.random(n) > 0.5
    elif overseas_dtype == "object_with_nan":
        # What pd.read_csv gives when the scraper wrote a blank for
        # one player: dtype object, holding True / False / nan.
        overseas = np.array(
            [True, False] * (n // 2), dtype=object
        )
        overseas[3] = np.nan
        overseas = pd.Series(overseas)
    elif overseas_dtype == "yes_no":
        overseas = pd.Series(rng.choice(["Yes", "No"], size=n))
    else:
        raise ValueError(overseas_dtype)

    capped = rng.choice(["CAPPED", "UNCAPPED"], size=n)
    if capped_style == "title":
        capped = np.array([c.title() for c in capped])

    df = pd.DataFrame({
        "playerId": np.arange(n) // 4,
        "playerName": [f"p{i // 4}" for i in range(n)],
        "team": rng.choice(["CSK", "MI", "RCB", "KKR"], size=n),
        "role": rng.choice(["Batter", "Bowler"], size=n),
        "country": rng.choice(["India", "Australia"], size=n),
        "countryId": 2,
        "cappedStatus": capped,
        "isPlayerOverseas": overseas,
        "basePrice": base,
        "auctionPrice": base * 2,
        "auctionStatus": "SOLD",
        "playsForTeam": "CSK",
        "observation_type": rng.choice(["left", "right", "interval"], size=n),
        "bat_runs": rng.integers(0, 9000, n).astype(float),
        "bat_strike_rate": rng.random(n) * 200,
        "bat_runs_is_missing": 0.0,
        "remaining_purse": rng.random(n) * 11000,
        "players_bought": rng.integers(0, 20, n).astype(float),
        "auction_order": np.arange(n, dtype=float),
        "players_remaining": np.arange(n, 0, -1, dtype=float),
        "lower": 1.0,
        "upper": base,
        "winner": False,
    })

    df.attrs["player_feature_columns"] = list(PLAYER_COLS)
    df.attrs["team_state_columns"] = list(TEAM_COLS)
    df.attrs["auction_state_columns"] = list(AUCTION_COLS)
    return df


def report(title, **kw):
    print(f"\n=== {title} ===")
    for k, v in kw.items():
        print(f"  {k}: {v}")


def trace(df, label):
    """Full path: context -> roles -> scalers -> dataset -> model."""

    df = add_player_context_features(df)
    cols = df.attrs["player_feature_columns"]

    role_frame, role_columns = build_role_table(df, verbose=False)
    attrs = dict(df.attrs)
    df = pd.concat([df.reset_index(drop=True), role_frame], axis=1)
    df.attrs = attrs
    df.attrs["role_columns"] = role_columns

    enc = build_encoders(df)
    scalers = fit_scalers(df)
    ds = IPLAuctionDataset(df, enc, scalers=scalers)

    idx = {c: i for i, c in enumerate(scalers["player_feature_columns"].columns)}

    out = {"player_feature_columns": cols}
    for name in ("ctx_basePrice", "ctx_cappedStatus", "ctx_isPlayerOverseas"):
        if name not in idx:
            out[name] = "ABSENT FROM MODEL INPUT"
            continue
        col = ds.player_features[:, idx[name]].numpy()
        raw = df[name.replace("ctx_", "")]
        out[name] = (
            f"tensor col {idx[name]:>2} | distinct={len(np.unique(col))} "
            f"| std={col.std():.4f} | "
            f"corr_with_raw={_corr(col, df[name]):.4f}"
        )

    report(label, **out)
    return ds, df


def _corr(scaled, raw):
    raw = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    if np.nanstd(raw) == 0 or scaled.std() == 0:
        return float("nan")
    ok = np.isfinite(raw)
    return float(np.corrcoef(scaled[ok], raw[ok])[0, 1])


if __name__ == "__main__":

    # 1. The happy path: booleans as pandas reads them from a clean CSV.
    ds, df = trace(make_frame(overseas_dtype="bool"), "bool overseas / CAPPED strings")

    print("\n  spot check, first 5 rows (raw -> scaled):")
    idx = {c: i for i, c in enumerate(df.attrs["player_feature_columns"])}
    for name in ("ctx_basePrice", "ctx_cappedStatus", "ctx_isPlayerOverseas"):
        i = idx[name]
        print(f"    {name:>24}: raw={list(df[name].head(5))} "
              f"scaled={np.round(ds.player_features[:5, i].numpy(), 3).tolist()}")

    # 2. One blank in the scraped column -> object dtype.
    trace(make_frame(overseas_dtype="object_with_nan"),
          "overseas column arrives as object dtype (one blank)")

    # 3. A different truthy spelling.
    trace(make_frame(overseas_dtype="yes_no"), "overseas column arrives as Yes/No")

    # 4. Title-case capped status.
    trace(make_frame(capped_style="title"), "cappedStatus arrives Title-Case")

    # 5. Idempotence: calling twice must not double the columns.
    d = add_player_context_features(add_player_context_features(make_frame()))
    report("called twice", player_feature_columns=d.attrs["player_feature_columns"])


def extra():
    """Failure modes that must now be loud rather than silent."""

    # 6. An unrecognised spelling must not become False.
    df = make_frame()
    df["isPlayerOverseas"] = "Foreigner"
    report("unrecognised token", note="expect UNRECOGNISED + CONSTANT below")
    df = add_player_context_features(df)
    print("   ctx_isPlayerOverseas values:",
          df["ctx_isPlayerOverseas"].unique().tolist(),
          "| flag mean:", df["ctx_isPlayerOverseas_is_missing"].mean())

    # 7. Full forward pass, to prove the widened block matches player_dim.
    ds, df = trace(make_frame(), "forward pass")
    model = ValuationModel(
        player_dim=len(df.attrs["player_feature_columns"]),
        team_state_dim=len(TEAM_COLS),
        auction_state_dim=len(AUCTION_COLS),
        num_role_features=len(df.attrs["role_columns"]),
        num_teams=4,
    )
    with torch.no_grad():
        out = model(
            player_features=ds.player_features,
            role_features=ds.role_features,
            team=ds.team,
            team_state=ds.team_state,
            auction_state=ds.auction_state,
        )
    print("   forward ok:", {k: tuple(v.shape) for k, v in out.items()}
          if isinstance(out, dict) else tuple(o.shape for o in out))

    # 8. Two splits built independently must present the same block.
    a = add_player_context_features(make_frame(seed=1))
    b = add_player_context_features(
        make_frame(seed=2, overseas_dtype="object_with_nan")
    )
    same = a.attrs["player_feature_columns"] == b.attrs["player_feature_columns"]
    report("train/val schema", identical=same,
           train_width=len(a.attrs["player_feature_columns"]),
           val_width=len(b.attrs["player_feature_columns"]))


extra()