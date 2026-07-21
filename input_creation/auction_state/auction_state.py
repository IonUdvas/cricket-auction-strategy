import numpy as np
import pandas as pd

from .utils import parse_price

class AuctionReplayEngine:
    TEAM_ALIASES = {
        "DD": "DC",
        "KXIP": "PBKS"
    }
    
    def __init__(
        self,
        bid_df,
        player_df,
        initial_purse=8000,
        squad_size=25,
        overseas_limit=8
    ):
        
        self.bid_df = bid_df.copy()
        self.player_df = player_df.copy()

        self._normalize_inputs()

        self.initial_purse = initial_purse
        self.squad_size = squad_size
        self.overseas_limit = overseas_limit

        self.teams = sorted(pd.concat([
                self.bid_df["Team"],
                self.player_df["playsForTeam"]
            ]).dropna().unique())
        
    def _normalize_inputs(self):
        self.bid_df["Team"] = (
            self.bid_df["Team"]
            .replace(self.TEAM_ALIASES)
        )

        self.player_df["playsForTeam"] = (
            self.player_df["playsForTeam"]
            .replace(self.TEAM_ALIASES)
        )

        self.bid_df["BidAmount"] = (
            self.bid_df["BidAmount"]
            .apply(parse_price)
        )

        self.player_df["auctionPrice"] = (
            self.player_df["auctionPrice"]
            .apply(parse_price)
        )

    ###############################################################
    def _apply_preauction_events(self, team_state):
        retained = self.player_df[
            self.player_df["auctionStatus"]
            .str.lower()
            .eq("retained")
        ]

        for _, player in retained.iterrows():
            self._handle_retention(player, team_state)


    def replay(self):

        team_state = self._initialize_team_state()
        self._apply_preauction_events(team_state)

        auction_rows = []

        team_rows = []

        players = (
            self.player_df
            .iloc[::-1]
            .reset_index(drop=True)
        )

        total_players = len(players)

        for order, player in players.iterrows():

            ###################################################
            # Save state BEFORE auctioning this player
            ###################################################

            auction_rows.append(
                self._snapshot_auction_state(
                    players,
                    order,
                    player
                )
            )

            for team in self.teams:

                row = team_state[team].copy()

                row["auction_order"] = order

                row["playerId"] = player["playerId"]

                row["playerName"] = player["playerName"]

                team_rows.append(row)

            ###################################################
            # Replay this player's bidding
            ###################################################

            self._replay_player(
                player,
                team_state
            )

        auction_state_df = pd.DataFrame(
            auction_rows
        )

        team_state_df = pd.DataFrame(
            team_rows
        )

        return auction_state_df, team_state_df

    ###############################################################

    def _initialize_team_state(self):

        state = {}

        for team in self.teams:

            state[team] = {

                "team": team,

                "remaining_purse":
                    self.initial_purse,

                "players_bought": 0,

                "remaining_slots":
                    self.squad_size,

                "overseas_bought": 0

            }

        return state

    ###############################################################

    def _snapshot_auction_state(
        self,
        players,
        order,
        player
    ):

        remaining = players.iloc[order:]

        return {

            "auction_order":
                order,

            "playerId":
                player["playerId"],

            "playerName":
                player["playerName"],

            "players_completed":
                order,

            "players_remaining":
                len(players) - order,

            "remaining_batters":
                (remaining["role"] == "Batter").sum(),

            "remaining_bowlers":
                (remaining["role"] == "Bowler").sum(),

            "remaining_allrounders":
                (remaining["role"] == "Allrounder").sum(),

            "remaining_overseas":
                remaining["isPlayerOverseas"].sum()
        }

    ##############################################################

    def _replay_player(
        self,
        player,
        team_state
    ):

        if player["auctionStatus"] != "sold":
            return

        winner = player["playsForTeam"]

        price = player["auctionPrice"]

        team_state[winner]["remaining_purse"] -= price

        team_state[winner]["players_bought"] += 1

        team_state[winner]["remaining_slots"] -= 1

        if player["isPlayerOverseas"]:

            team_state[winner]["overseas_bought"] += 1

# def _replay_player(self, player, team_state):
#     status = player["auctionStatus"].lower()

#     if status == "sold":
#         self._handle_sold(player, team_state)

#     elif status == "retained":
#         self._handle_retention(player, team_state)

#     elif status == "rtm":
#         self._handle_rtm(player, team_state)

#     elif status == "unsold":
#         self._handle_unsold(player)

#     else:
#         self._handle_other(player)
