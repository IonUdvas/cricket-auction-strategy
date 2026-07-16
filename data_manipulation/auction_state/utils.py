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