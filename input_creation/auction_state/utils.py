import numpy as np
import pandas as pd

def parse_price(price):
    if pd.isna(price):
        return np.nan

    if price == "--":
        return np.nan

    price = price.strip()

    if price.endswith("Cr"):
        return float(price.replace("Cr", "")) * 100

    if price.endswith("L"):
        return float(price.replace("L", ""))

    return float(price)


def build_bid_summary(player_bid_df):

    """
    Build interval-censored valuation data for one player's auction.

    Parameters
    ----------
    player_bid_df : pd.DataFrame
        Bid-by-bid dataframe for ONE player.

    Returns
    -------
    pd.DataFrame

    Columns
    -------
    playerId
    playerName
    team
    last_bid
    lower
    upper
    winner
    """

    if len(player_bid_df) == 0:
        return pd.DataFrame()

    ############################################################
    # Sort auction chronologically
    ############################################################

    bids = player_bid_df.copy()

    bids["BidAmount"] = bids["BidAmount"].apply(parse_price)

    bids = (
        bids
        .sort_values("BidAmount", kind="stable")
        .reset_index(drop=True)
    )

    ############################################################

    first_row = bids.iloc[0]

    sold = first_row["auctionStatus"] == "sold"

    winner = (
        first_row["playsForTeam"]
        if sold else None
    )

    ############################################################
    # Last bid index for every team
    ############################################################

    last_bid_idx = {}

    for idx, row in bids.iterrows():

        last_bid_idx[row["Team"]] = idx

    ############################################################
    # Construct summary
    ############################################################

    summary = []

    for team, idx in last_bid_idx.items():

        last_bid = bids.loc[idx, "BidAmount"]

        if idx == len(bids) - 1:

            upper = np.inf

        else:

            upper = bids.loc[idx + 1, "BidAmount"]

        summary.append({

            "playerId":
                first_row["playerId"],

            "playerName":
                first_row["playerName"],

            "team":
                team,

            "last_bid":
                last_bid,

            "lower":
                last_bid,

            "upper":
                upper,

            "winner":
                sold and (team == winner)

        })

    summary = pd.DataFrame(summary)

    summary.sort_values(
        "lower",
        ascending=False,
        inplace=True
    )

    summary.reset_index(
        drop=True,
        inplace=True
    )

    return summary

def build_bid_summary_for_all(player_bid_df, all_teams):
    """
    Build interval-censored valuation data for one player's auction.
    """

    if len(player_bid_df) == 0:
        return pd.DataFrame()

    ############################################################
    # Sort auction chronologically
    ############################################################

    bids = player_bid_df.copy()

    bids["BidAmount"] = bids["BidAmount"].apply(parse_price)
    bids["basePrice"] = bids["basePrice"].apply(parse_price)

    bids = (
        bids
        .sort_values("BidAmount", kind="stable")
        .reset_index(drop=True)
    )

    first_row = bids.iloc[0]

    sold = first_row["auctionStatus"] == "sold"

    winner = first_row["playsForTeam"] if sold else None

    base_price = first_row["basePrice"]

    ############################################################
    # Last bid index for every participating team
    ############################################################

    last_bid_idx = {}

    for idx, row in bids.iterrows():
        last_bid_idx[row["Team"]] = idx

    ############################################################
    # Construct summary
    ############################################################

    summary = []

    participating_teams = set(last_bid_idx.keys())

    ############################################################
    # Teams that participated
    ############################################################

    for team, idx in last_bid_idx.items():

        last_bid = bids.loc[idx, "BidAmount"]

        if idx == len(bids) - 1:
            upper = 40
        else:
            upper = bids.loc[idx + 1, "BidAmount"]

        summary.append({

            "playerId": first_row["playerId"],
            "playerName": first_row["playerName"],

            "team": team,

            "last_bid": last_bid,

            "lower": last_bid,
            "upper": upper,

            "winner": sold and (team == winner)

        })

    ############################################################
    # Teams that never entered the bidding
    ############################################################

    for team in all_teams:

        if team in participating_teams:
            continue

        summary.append({

            "playerId": first_row["playerId"],
            "playerName": first_row["playerName"],

            "team": team,

            "last_bid": np.nan,

            "lower": 0.01,
            "upper": base_price,

            "winner": False

        })

    summary = pd.DataFrame(summary)

    summary.sort_values(
        ["lower", "upper"],
        ascending=[False, True],
        inplace=True
    )

    summary.reset_index(drop=True, inplace=True)

    return summary