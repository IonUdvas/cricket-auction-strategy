import numpy as np
import pandas as pd
import re



class AuctionReplayEngine:
    """
    Replays an IPL auction exactly once.
    During replay, the engine simultaneously constructs

        • auction_state_df
        • team_state_df
        • bid_summary_df
        • training_df
    Every row corresponds to one
        Player × Team
    observed immediately BEFORE the player's auction outcome is applied.
    """

    STATUS_SOLD = "SOLD"
    STATUS_UNSOLD = "UNSOLD"
    STATUS_RETAINED = "RETAINED"
    STATUS_RTM = "RTM"
    STATUS_TRADED = "TRADED"

    ####################################################################
    # Team aliases
    ####################################################################

    TEAM_ALIASES = {
        "DD": "DC",
        "KXIP": "PBKS",
    }

    ####################################################################
    # Constructor
    ####################################################################

    def __init__(
        self,
        bid_df,
        player_df,
        auction_max_purse,
        squad_size=25,
        overseas_limit=8,
    ):

        self.bid_df = bid_df.copy()
        self.player_df = player_df.copy()

        self.auction_max_purse = auction_max_purse
        self.squad_size = squad_size
        self.overseas_limit = overseas_limit

        ############################################################
        # Internal state
        ############################################################

        self.teams = None
        self.team_state = None

        ############################################################
        # Output buffers
        ############################################################

        self.outputs = {
            "training": [],
            "auction_state": [],
            "team_state": [],
            "bid_summary": [],
        }

        ############################################################
        # Normalize all inputs
        ############################################################

        self._normalize_inputs()


    def _normalize_inputs(self):
        """
        Normalize all auction inputs into a consistent internal schema.

        Responsibilities
        ----------------
        1. Normalize team aliases
        2. Parse all monetary columns
        3. Normalize auction status
        4. Sort player and bid tables
        5. Discover participating teams
        """

        self.player_df = self.player_df.copy()
        self.bid_df = self.bid_df.copy()

        # ---------------------------------------------------------
        # Normalize team aliases
        # ---------------------------------------------------------

        self.player_df["playsForTeam"] = (
            self.player_df["playsForTeam"]
            .replace(self.TEAM_ALIASES)
        )

        self.bid_df["Team"] = (
            self.bid_df["Team"]
            .replace(self.TEAM_ALIASES)
        )

        self.bid_df["playsForTeam"] = (
            self.bid_df["playsForTeam"]
            .replace(self.TEAM_ALIASES)
        )

        # ---------------------------------------------------------
        # Parse monetary columns
        # ---------------------------------------------------------

        money_cols = [
            "basePrice",
            "auctionPrice",
        ]

        for col in money_cols:
            self.player_df[col] = (
                self.player_df[col]
                .apply(self._parse_price)
            )

        money_cols = [
            "BidAmount",
            "basePrice",
            "auctionPrice",
        ]

        for col in money_cols:
            self.bid_df[col] = (
                self.bid_df[col]
                .apply(self._parse_price)
            )

        # ---------------------------------------------------------
        # Normalize auction status
        # ---------------------------------------------------------

        self.player_df["auctionStatus"] = (
            self.player_df["auctionStatus"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        self.bid_df["auctionStatus"] = (
            self.bid_df["auctionStatus"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # ---------------------------------------------------------
        # Sort data
        # ---------------------------------------------------------

        self.bid_df = (
            self.bid_df
            .sort_values(
                ["playerId", "BidAmount"],
                ascending=[True, True]
            )
            .reset_index(drop=True)
        )
        
        # Recreate BidNumber within each player
        self.bid_df["BidNumber"] = (
            self.bid_df
            .groupby("playerId")
            .cumcount() + 1
        )
        
        VALID_STATUSES = {
            self.STATUS_SOLD,
            self.STATUS_UNSOLD,
            self.STATUS_RETAINED,
            self.STATUS_RTM,
        }

        self.player_df = (
            self.player_df[
                self.player_df["auctionStatus"].isin(VALID_STATUSES)
            ]
            .reset_index(drop=True)
        )

        self.bid_df = (
            self.bid_df
            .sort_values("BidNumber")
            .reset_index(drop=True)
        )

        valid_player_ids = set(self.player_df["playerId"])

        self.bid_df = (
            self.bid_df[
                self.bid_df["playerId"].isin(valid_player_ids)
            ]
            .reset_index(drop=True)
        )

        # player_df arrives in REVERSE bidding order (row 0 is the
        # last player who went under the hammer). There's no explicit
        # order column, so we recover chronological order by simply
        # reversing the frame rather than sorting on anything.
        self.player_df = self.player_df.iloc[::-1].reset_index(drop=True)

        # ---------------------------------------------------------
        # Participating teams
        # ---------------------------------------------------------

        self.teams = sorted(
            set(self.bid_df["Team"].dropna())
            |
            set(self.player_df["playsForTeam"].dropna())
        )

        ROLE_ALIASES = {
            "WK-BATTER": "WICKETKEEPER",
            "ALLROUNDER": "ALL-ROUNDER",
        }
        self.player_df["role"] = self.player_df["role"].str.upper().replace(ROLE_ALIASES)


    def _parse_price(self, value):
        """
        Convert auction prices into numeric values (Lakhs).
        Examples
        --------
        "75 L"     -> 75
        "2 Cr"     -> 200
        "1.25 Cr"  -> 125
        np.nan     -> np.nan
        """
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float)):
            return float(value)

        value = str(value).strip()

        if value == "":
            return np.nan

        value = value.replace(",", "")

        match = re.match(
            r"([\d.]+)\s*(CR|CRORE|L|LAKH)?",
            value.upper()
        )

        if match is None:
            return np.nan

        amount = float(match.group(1))

        unit = match.group(2)

        if unit in ["CR", "CRORE"]:
            return amount * 100

        return amount
    
    def _initialize_team_state(self):
        """
        Initialize the dynamic state maintained for each team during
        the auction replay.

        Returns
        -------
        dict
            Dictionary keyed by team name.
        """

        self.team_state = {
            team: {
                # Purse remaining
                "remaining_purse": self.auction_max_purse,
                # Number of players already acquired
                "players_bought": 0,
                # Squad slots remaining
                "remaining_slots": self.squad_size,
                # Overseas players already acquired
                "overseas_bought": 0,
                # Role composition
                "batters_bought": 0,
                "bowlers_bought": 0,
                "allrounders_bought": 0,
                "wicketkeepers_bought": 0,

            }
            for team in self.teams
        }

    def _apply_preauction_events(self):
        """
        Apply all auction events that occurred before the first player
        entered the auction.
        Currently this consists of retained players.
        """
        retained = self.player_df[
            self.player_df["auctionStatus"] == self.STATUS_RETAINED
        ]
        for _, player in retained.iterrows():
            # A retention is economically identical to a sale: one
            # player leaves the pool, one team's purse/slot/role
            # counters move. Reuse _apply_sale so this can never
            # drift out of sync with how a normal sale is applied.
            self._apply_sale(player)

        ############################################################
        # Remove retained players from the auction replay
        ############################################################

        self.player_df = (
            self.player_df[
                self.player_df["auctionStatus"] != self.STATUS_RETAINED
            ]
            .reset_index(drop=True)
        )
        

    def _increment_role_count(self, state, role):
        role = role.upper()

        mapping = {
            "BATTER": "batters_bought",
            "BOWLER": "bowlers_bought",
            "ALL-ROUNDER": "allrounders_bought",
            "WICKETKEEPER": "wicketkeepers_bought",
        }

        key = mapping.get(role)

        if key is not None:
            state[key] += 1

    def _snapshot_auction_state(
        self,
        auction_order,
    ):
        """
        Construct auction-level features immediately before the
        current player is auctioned.
        """

        remaining = self.player_df.iloc[auction_order:]

        return {
            "auction_order": auction_order + 1,
            "players_completed": auction_order,
            "players_remaining": len(self.player_df) - auction_order,
            "remaining_batters":
                (remaining["role"] == "BATTER").sum(),
            "remaining_bowlers":
                (remaining["role"] == "BOWLER").sum(),
            "remaining_allrounders":
                (remaining["role"] == "ALL-ROUNDER").sum(),
            "remaining_wicketkeepers":
                (remaining["role"] == "WICKETKEEPER").sum(),
            "remaining_overseas":
                remaining["isPlayerOverseas"].sum(),
        }
    
    def _build_bid_summary(
        self,
        player,
    ):
        """
        Construct valuation observations for every team for a single player.
        Parameters
        ----------
        player : pd.Series
            Row from self.player_df.
        Returns
        -------
        dict
            {
                team_name : {
                    "lower": ...,
                    "upper": ...,
                    "winner": ...,
                    "observation_type": ...
                }
            }

        Notes
        -----
        This function is PURE.
        It never modifies team state.
        """

        player_bid_df = (
            self.bid_df[
                self.bid_df["playerId"] == player["playerId"]
            ]
            .sort_values("BidNumber")
            .reset_index(drop=True)
        )       

        status = player["auctionStatus"]

        if status == self.STATUS_SOLD:
            return self._summary_sold(
                player,
                player_bid_df,
            )

        elif status == self.STATUS_RTM:
            return self._summary_rtm(
                player,
                player_bid_df,
            )

        elif status == self.STATUS_RETAINED:
            return self._summary_retained(
                player,
                player_bid_df,
            )
        elif status == self.STATUS_UNSOLD:
            return self._summary_unsold(
                player,
                player_bid_df,
            )

        else:
            raise ValueError(
                f"Unknown auction status: {status}"
            )
        

    def _empty_team_summary(self):
        """
        Create an empty valuation dictionary for every team.
        """

        return {
            team: {
                "lower": np.nan,
                "upper": np.nan,
                "winner": False,
                "observation_type": "unknown",
            }
            for team in self.teams

        }

    def _winner_upper_bound(
        self,
        winning_bid,
    ):
        """
        Returns the artificial upper bound used for
        right-censored winner observations.
        """
        return winning_bid * 2
    
    def _extract_team_bid_history(
        self,
        player_bid_df,
        base_price,
    ):

        history = {}
        # initialize
        for team in self.teams:
            history[team] = {   
                "entered": False, 
                "all_bids": [], 
                "last_bid": np.nan,  
                "previous_bid": np.nan,
                "next_bid": np.nan,
            }
    
        # collect bids
        for _, row in player_bid_df.iterrows():
            history[row["Team"]]["all_bids"].append(row["BidAmount"])
    
        # fill last/previous
        for team in self.teams:
            bids = history[team]["all_bids"]    
            if len(bids) == 0:
                continue    
            history[team]["entered"] = True    
            history[team]["last_bid"] = bids[-1]    
            history[team]["previous_bid"] = (
                bids[-2] if len(bids) >= 2 else base_price
            )
    
        # compute next_bid
        bids = player_bid_df.reset_index(drop=True)    
        for i in range(len(bids) - 1):    
            team = bids.loc[i, "Team"]    
            history[team]["next_bid"] = bids.loc[i + 1, "BidAmount"]
    
        return history
    
    def _summary_sold(
        self,
        player,
        player_bid_df,
    ):
        """
        Construct valuation observations for a sold player.
        """

        summary = self._empty_team_summary()

        history = self._extract_team_bid_history(
            player_bid_df,
            player["basePrice"]
        )

        winner = player["playsForTeam"]
        winning_bid = player["auctionPrice"]
        base_price = player["basePrice"]

        for team in self.teams:
            info = history[team]

            ####################################################
            # Never entered bidding
            ####################################################

            if not info["entered"]:
                summary[team].update({
                    "lower": 0.01,
                    "upper": base_price,
                    "winner": False,
                    "observation_type": "left",
                })

                continue

            ####################################################
            # Winner
            ####################################################

            if team == winner:
                summary[team].update({
                "previous_bid": info["previous_bid"],
                "last_bid": info["last_bid"],            
                "lower": winning_bid,            
                "upper": self._winner_upper_bound(
                    winning_bid
                ),            
                "winner": True,            
                "observation_type": "right",
            })

                continue

            ####################################################
            # Losing bidder
            ####################################################

            summary[team].update({
                "previous_bid": info["previous_bid"],
                "last_bid": info["last_bid"],
                "lower": info["last_bid"],
                "upper": info["next_bid"],
                "winner": False,
                "observation_type": "interval",
            })

        
        return summary
    
    def _summary_unsold(
        self,
        player,
        player_bid_df,
    ):
        """
        Construct valuation observations for an unsold player.
        """
        summary = self._empty_team_summary()
        history = self._extract_team_bid_history(
            player_bid_df,
            player["basePrice"]
        )
        base_price = player["basePrice"]

        for team in self.teams:
            info = history[team]

            ###############################################
            # Never entered
            ###############################################

            if not info["entered"]:
                summary[team].update({
                    "lower": 0.01,
                    "upper": base_price,
                    "winner": False,
                    "observation_type": "left",
                })

                continue

        return summary
    
    def _summary_retained(
        self,
        player,
        player_bid_df=None,
    ):
        """
        Construct valuation observations for retained players.
        """

        summary = self._empty_team_summary()
        retained_team = player["playsForTeam"]
        retained_price = player["auctionPrice"]

        for team in self.teams:
            if team == retained_team:
                summary[team].update({
                    "lower": retained_price,
                    "upper": self._winner_upper_bound(
                        retained_price
                    ),
                    "winner": True,
                    "observation_type": "right",
                })

            else:

                summary[team].update({
                    "lower": np.nan,
                    "upper": np.nan,
                    "winner": False,
                    "observation_type": "unknown",
                })

        return summary
    
    def _summary_rtm(self, player, player_bid_df):
        """
        RTM purchases are treated identically to normal sales for
        valuation-interval purposes, for now.
        """
        return self._summary_sold(player, player_bid_df)

    def _player_metadata(
        self,
        player,
        team,
    ):
        """
        Static metadata for a Player × Team training row.
        """
        return {
            "playerId": player["playerId"],
            "playerName": player["playerName"],
            "team": team,
            "role": player["role"],
            "country": player["country"],
            "countryId": player["countryId"],
            "cappedStatus": player["cappedStatus"],
            "isPlayerOverseas": player["isPlayerOverseas"],
            "basePrice": player["basePrice"],
            "auctionPrice": player["auctionPrice"],
            "auctionStatus": player["auctionStatus"],
            "playsForTeam": player["playsForTeam"],
        }
    
    def _build_player_training_rows(
        self,
        player,
        auction_order,
    ):
        """
        Construct all training rows for a single player.
        One row is produced for every team.
        This function is PURE.
        It snapshots the current auction state but DOES NOT modify it.
        """

        ############################################################
        # Auction-wide state
        ############################################################

        auction_state = self._snapshot_auction_state(
            auction_order
        )

        ############################################################
        # Bid summary
        ############################################################

        bid_summary = self._build_bid_summary(
            player
        )

        ############################################################
        # One row per team
        ############################################################

        for team in self.teams:
            ########################################################
            # Components
            ########################################################
            player_info = self._player_metadata(
                player,
                team,
            )

            team_state = self.team_state[team].copy()
            interval = bid_summary[team].copy()

            ########################################################
            # Individual outputs
            ########################################################

            auction_state_row = {
                **player_info,
                **auction_state,
            }

            team_state_row = {
                **player_info,            
                **auction_state,            
                **team_state,            
            }

            bid_summary_row = {
                **player_info,
                **interval,
            }

            training_row = {
                **player_info,
                **auction_state,
                **team_state,
                **interval,
            }

            ########################################################
            # Save
            ########################################################

            self.outputs["auction_state"].append(
                auction_state_row
            )

            self.outputs["team_state"].append(
                team_state_row
            )

            self.outputs["bid_summary"].append(
                bid_summary_row
            )

            self.outputs["training"].append(
                training_row
            )

    def _apply_sale(
        self,
        player,
    ):
        """
        Apply a successful auction sale to the team state.
        """

        team = player["playsForTeam"]

        if pd.isna(team):
            return

        state = self.team_state[team]

        ############################################################
        # Purse
        ############################################################

        state["remaining_purse"] -= player["auctionPrice"]

        ############################################################
        # Squad
        ############################################################

        state["players_bought"] += 1
        state["remaining_slots"] -= 1

        ############################################################
        # Overseas
        ############################################################

        if player["isPlayerOverseas"]:
            state["overseas_bought"] += 1

        ############################################################
        # Role
        ############################################################

        self._increment_role_count(
            state,
            player["role"],
        )

    def _apply_retention(
        self,
        player,
    ):
        """
        Retentions are applied before the auction begins.

        Therefore this method intentionally does nothing.
        """
        return
    
    def _apply_unsold(
        self,
        player,
    ):
        """
        Unsold players do not modify team state.
        """
        return

    def _apply_rtm(
        self,
        player,
    ):
        """
        Apply an RTM purchase.

        For now, an RTM buy is treated identically to a normal sale
        for team-state purposes (purse/slot/overseas/role counters).
        RTM-specific valuation modelling (bid_summary bounds) is a
        separate TODO — see _summary_rtm.
        """

        self._apply_sale(player)

    def _apply_player_result(
        self,
        player,
    ):
        """
        Advance the auction state by applying the current player's
        auction outcome.
        """
        status = player["auctionStatus"]
        if status == self.STATUS_SOLD:
            self._apply_sale(player)

        elif status == self.STATUS_RETAINED:
            self._apply_retention(player)

        elif status == self.STATUS_UNSOLD:
            self._apply_unsold(player)

        elif status == self.STATUS_RTM:
            self._apply_rtm(player)

        else:
            raise ValueError(
                f"Unknown auction status: {status}"
            )
        
    def replay(
        self,
    ):
        """
        Replay the auction exactly once.

        Returns
        -------
        dict

            {
                "training": DataFrame,
                "auction_state": DataFrame,
                "team_state": DataFrame,
                "bid_summary": DataFrame,
            }
        """

        ############################################################
        # Reset outputs
        ############################################################

        self.outputs = {
            "training": [],
            "auction_state": [],
            "team_state": [],
            "bid_summary": [],
        }

        ############################################################
        # Initialize auction state
        ############################################################

        self._initialize_team_state()
        self._apply_preauction_events()

        ############################################################
        # Replay
        ############################################################

        for auction_order, (_, player) in enumerate(
            self.player_df.iterrows()
        ):
            self._build_player_training_rows(
                player,
                auction_order,
            )

            self._apply_player_result(
                player,
            )

        ############################################################
        # Convert to DataFrames
        ############################################################

        return {
            key: pd.DataFrame(value)
            for key, value in self.outputs.items()
        }