"""
tests/audit_data_loss.py -- where rows and players go missing.

Run: python -m tests.audit_data_loss     (needs only the auction CSVs)

Every number in the report this file was written for is reproducible
from here, including the two claims that turned out to be WRONG on
closer inspection and are checked explicitly so they cannot be
re-asserted from memory.
"""
import sys, io, contextlib
import pandas as pd

sys.path.insert(0, ".")
from input_creation_2.money import parse_money
from input_creation_2.auction_replay_engine import AuctionReplayEngine

YEARS = range(2018, 2027)
PURSE = {2018: 8000, 2019: 8200, 2020: 8500, 2021: 8500, 2022: 9000,
         2023: 9500, 2024: 10000, 2025: 12000, 2026: 12500}


def load(trail, kind, year):
    return pd.read_csv(f"{trail}/{kind}/{kind}_{year}.csv")


def audit(trail, archetypes_path):
    ARCH = pd.read_csv(archetypes_path)
    rows = []
    for y in YEARS:
        cp = load(trail, "completed_players", y)
        tr = load(trail, "auction_trail", y)
        eng = AuctionReplayEngine(bid_df=tr, player_df=cp,
                                  auction_max_purse=PURSE[y],
                                  archetype_df=ARCH)
        with contextlib.redirect_stdout(io.StringIO()):
            out = eng.replay()
        t = out["training"]
        n_teams = len(eng.teams)
        st = cp.auctionStatus.str.upper().value_counts()
        pre = int(st.get("RETAINED", 0) + st.get("TRADED", 0) + st.get("DRAFTED", 0))
        pool = len(cp) - pre
        vc = t.observation_type.value_counts()
        rows.append(dict(
            year=y, teams=n_teams, roster=len(cp), never_emitted=pre,
            in_pool=pool, potential=pool * n_teams, actual=len(t),
            dropped_unknown=pool * n_teams - len(t),
            left=int(vc.get("left", 0)), right=int(vc.get("right", 0)),
            interval=int(vc.get("interval", 0)),
        ))
    return pd.DataFrame(rows)


def retained_are_degenerate(trail):
    """
    The 940 retained players look like free training data. They are not:
    a retention price is the prior season's salary, which is exactly
    what the `last_salary` FEATURE holds -- so every such row would
    have target == feature and teach the model to echo last_salary.
    """
    out = []
    for y in (2023, 2024, 2026):
        cp = load(trail, "completed_players", y)
        r = cp[cp.auctionStatus.str.upper() == "RETAINED"].copy()
        r["p"] = r.auctionPrice.apply(parse_money)
        e = load(trail, "earnings", y)
        e["amt"] = e.Amount.apply(parse_money)
        prior = e[e.Season == y - 1].drop_duplicates("playerId").set_index("playerId")["amt"]
        r["prev"] = r.playerId.map(prior)
        both = r.dropna(subset=["prev"])
        identical = float(((both.p - both.prev).abs() < 1e-6).mean())
        out.append(dict(year=y, retained=len(r), with_prior=len(both),
                        pct_identical=round(identical * 100)))
    return pd.DataFrame(out)


def ladder_completeness(trail):
    """
    A single-row bid ladder looks like a truncated record. Check
    before believing it: 479 of 481 are players who sold AT their
    base price, i.e. genuinely uncontested and correctly recorded.
    Only 2 across nine editions are real gaps.
    """
    keep = []
    for y in YEARS:
        cp = load(trail, "completed_players", y)
        tr = load(trail, "auction_trail", y)
        n_teams = len(set(cp.playsForTeam.dropna()) | set(tr.Team.dropna()))
        s = cp[cp.auctionStatus.str.upper().isin(["SOLD", "RTM"])].copy()
        s["base"] = s.basePrice.apply(parse_money)
        s["price"] = s.auctionPrice.apply(parse_money)
        s["n"] = s.playerId.map(tr.groupby("playerId").size()).fillna(0)
        s["year"] = y
        s["teams"] = n_teams
        keep.append(s)
    s = pd.concat(keep)
    single = s[s.n <= 1].copy()
    single["ratio"] = single.price / single.base
    gaps = single[single.ratio > 1.0001]
    return single, gaps


def no_ball_record(trail, archetypes_path):
    A = pd.read_csv(archetypes_path)
    missing = set(A.loc[A.has_bbb_record != True, "player_id"])
    out = []
    for y in YEARS:
        cp = load(trail, "completed_players", y)
        pool = cp[~cp.auctionStatus.str.upper().isin(
            ["RETAINED", "TRADED", "DRAFTED"])]
        n = int(pool.playerId.isin(missing).sum())
        out.append(dict(year=y, in_pool=len(pool), no_ball_record=n,
                        pct=round(n / len(pool) * 100, 1)))
    return pd.DataFrame(out)


def main(trail, archetypes_path):
    d = audit(trail, archetypes_path)
    print("=== rows and players ===")
    print(d.to_string(index=False))
    print(f"\nplayers that never produce a training row: "
          f"{d.never_emitted.sum()} of {d.roster.sum()} "
          f"({d.never_emitted.sum()/d.roster.sum()*100:.1f}%) -- all "
          f"retained/traded/drafted, removed before the replay")
    print(f"rows dropped as 'unknown' inside the pool: "
          f"{d.dropped_unknown.sum()}  <- expected 0")
    assert d.dropped_unknown.sum() == 0, "in-pool rows are being lost"
    print(f"informative rows (right+interval): {d.right.sum()+d.interval.sum()}"
          f" | left: {d.left.sum()}")

    print("\n=== are the retained players recoverable? ===")
    print(retained_are_degenerate(trail).to_string(index=False))
    print("100% identical => target would equal the last_salary feature. "
          "Adding them would train a last_salary echo, not a valuation.")

    print("\n=== are single-row bid ladders truncated records? ===")
    single, gaps = ladder_completeness(trail)
    print(f"single-row ladders: {len(single)} | sold exactly at base price: "
          f"{int((single.ratio<=1.0001).sum())} "
          f"({(single.ratio<=1.0001).mean()*100:.0f}%)")
    print(f"genuine gaps (sold above base, no ladder): {len(gaps)}")
    if len(gaps):
        print(gaps[["year", "playerName", "basePrice", "auctionPrice",
                    "playsForTeam"]].to_string(index=False))

    print("\n=== players with no ball-by-ball record ===")
    print(no_ball_record(trail, archetypes_path).to_string(index=False))


if __name__ == "__main__":
    import data_sources as ds
    trail = ds.player_template().rsplit("/completed_players/", 1)[0]
    main(trail, ds.archetypes_path())
