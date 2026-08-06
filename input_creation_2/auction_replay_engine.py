import numpy as np
import pandas as pd

from input_creation_2.archetypes import (
    ARCHETYPES,
    apply_purchase,
    archetype_demand,
    auction_archetype_features,
    build_archetype_tags,
    empty_team_archetype_counts,
    focus_features,
    pool_archetype_counts,
    scarcity,
    tags_for,
    team_archetype_features,
)
import re

from input_creation_2.auction_order import resolve_auction_order


####################################################################
# The artificial upper bound on a winning observation.
#
# A winner tells us the buyer valued the player at AT LEAST the
# hammer price. It says nothing about the ceiling, so the row is
# right-censored and the interval needs some finite upper end for the
# likelihood to be computable. This constant is that end, as a
# multiple of the price: [P, WINNER_UPPER_MULTIPLE * P).
#
# It is a MODELLING CHOICE, not a fact about the auction, and it is
# the leading suspect for top-of-market compression: the likelihood
# is indifferent between any prediction inside the band and penalises
# predictions above it, so a larger multiple is what lets the model
# chase a marquee price. That makes it something to sweep.
#
# It lives at module scope, and _winner_upper_bound reads it through
# the module namespace at CALL time rather than binding it in
# __init__, precisely so that
#
#     import input_creation_2.auction_replay_engine as eng
#     eng.WINNER_UPPER_MULTIPLE = 3.0
#
# actually changes the next build. Before this existed the value was
# the literal 2 inside _winner_upper_bound, and that assignment
# created a new module attribute nobody read -- a sweep over it
# returned four identical rows and looked like a null result rather
# than like a broken experiment.
#
# If you set this, remember the data cache: src.sweep keys builds on
# it (see _build_key), but any cache of your own must be cleared.
####################################################################
WINNER_UPPER_MULTIPLE = 2.0


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
    STATUS_DRAFTED = "DRAFTED"

    # Acquisitions that happened before the hammer fell. All three move
    # a team's purse and squad exactly like a sale, and none of them is
    # a market-clearing valuation, so all three are applied to team
    # state and then removed from the bidding pool.
    PREAUCTION_STATUSES = (STATUS_RETAINED, STATUS_TRADED, STATUS_DRAFTED)

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
        archetype_df=None,
    ):

        self.bid_df = bid_df.copy()
        self.player_df = player_df.copy()

        # Archetype tags turn three coarse role counters into eleven that
        # match how a squad is actually built. Optional: without the table
        # the engine keeps emitting the legacy BATTER/BOWLER/ALL-ROUNDER
        # counters and nothing downstream changes.
        self.archetype_tags = (
            build_archetype_tags(archetype_df)
            if archetype_df is not None else None
        )

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
        # Data-quality diagnostics
        #
        # Every one of these was previously a silent recovery that
        # produced a plausible-looking training row.  They are counted
        # instead, and surfaced by `quality_report()` so a bad roster
        # or bid file shows up as a number rather than as a model that
        # mysteriously underprices RTM players.
        ############################################################

        self.diagnostics = {
            "buyer_absent_from_ladder": [],
            "unresolvable_buyer": [],
            "next_bid_backfilled": [],
            "matched_at_top": [],
            "dropped_bad_interval": [],
            "sale_price_below_base": [],
            "sale_price_missing": [],
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

        # ---------------------------------------------------------
        # Bid ordering
        #
        # This used to sort on (playerId, BidAmount) and then OVERWRITE
        # BidNumber with a cumcount, i.e. it threw away the recorded
        # bid sequence and re-derived it from the amounts.  Two things
        # went wrong with that.
        #
        # First, sort_values defaults to quicksort, which is unstable,
        # so equal BidAmounts came out in arbitrary order and the
        # last_bid / next_bid ladder for those teams was arbitrary too.
        #
        # Second and worse: when the recorded auctionPrice is not the
        # maximum BidAmount in the ladder (RTM, an unlogged final bid,
        # a rounding difference), re-sorting by amount puts a LOSING
        # team last.  The last team in the ladder is the one team that
        # never receives a next_bid, so that team ended up with
        # upper = NaN -- which auction_dataset.py then filled with 0
        # and the loss turned into a width-0.001 interval asserting
        # that team's valuation to three decimal places.
        #
        # So: trust the recorded BidNumber when the file has one, and
        # fall back to a STABLE sort on amount when it doesn't.
        # ---------------------------------------------------------

        has_recorded_order = (
            "BidNumber" in self.bid_df.columns
            and self.bid_df["BidNumber"].notna().all()
        )

        if has_recorded_order:
            self.bid_df = (
                self.bid_df
                .sort_values(["playerId", "BidNumber"], kind="mergesort")
                .reset_index(drop=True)
            )
        else:
            self.bid_df = (
                self.bid_df
                .sort_values(["playerId", "BidAmount"], kind="mergesort")
                .reset_index(drop=True)
            )

            self.bid_df["BidNumber"] = (
                self.bid_df
                .groupby("playerId")
                .cumcount() + 1
            )

        self.bid_order_source = (
            "recorded" if has_recorded_order else "reconstructed"
        )

        # A ladder whose amounts fall as BidNumber rises means the
        # recorded order and the recorded amounts disagree; one of them
        # is wrong and the interval bounds built from them will be too.
        self.non_monotone_ladders = sorted(
            pid
            for pid, grp in self.bid_df.groupby("playerId")
            if not grp["BidAmount"].is_monotonic_increasing
        )
        
        # TRADED and DRAFTED used to be absent here, so every traded or
        # drafted player was discarded before the replay began -- and
        # with them, the purse they cost. 49 players across seven
        # seasons, including the entire 2022 GT/LSG draft (KL Rahul at
        # 17 Cr, Hardik Pandya and Rashid Khan at 15 Cr each), Cameron
        # Green's 17.5 Cr trade to RCB in 2024 and Sanju Samson's 18 Cr
        # move to CSK in 2026.
        #
        # The rows themselves are no great loss: a trade fee and a draft
        # pick are negotiated, not bid, so they are not valuations. The
        # purse is the loss. GT began the 2022 auction 3800 lakh down
        # and LSG 3020 lakh down -- 42% and 34% of the purse -- and the
        # engine had both starting at the full 9000. Every team_state
        # feature for those franchises was wrong for the whole auction.
        VALID_STATUSES = {
            self.STATUS_SOLD,
            self.STATUS_UNSOLD,
            self.STATUS_RETAINED,
            self.STATUS_RTM,
            self.STATUS_TRADED,
            self.STATUS_DRAFTED,
        }

        self.player_df = (
            self.player_df[
                self.player_df["auctionStatus"].isin(VALID_STATUSES)
            ]
            .reset_index(drop=True)
        )

        # NOTE: no global sort_values("BidNumber") here any more.
        # BidNumber is a WITHIN-player counter, so sorting the whole
        # frame by it interleaved every player's first bid, then every
        # player's second bid, and so on.  Nothing downstream wanted
        # that -- _build_bid_summary re-sorts per player anyway -- and
        # it destroyed the only global ordering the frame had.

        valid_player_ids = set(self.player_df["playerId"])

        self.bid_df = (
            self.bid_df[
                self.bid_df["playerId"].isin(valid_player_ids)
            ]
            .reset_index(drop=True)
        )

        # Auction order.
        #
        # This used to be an unconditional `self.player_df.iloc[::-1]`
        # on the assumption the file arrives newest-first. The
        # assumption is load-bearing -- auction_order,
        # players_remaining and every purse trajectory come from it --
        # and nothing checked it, so a forward-ordered file produced a
        # well-formed frame describing an auction that ran backwards.
        #
        # resolve_auction_order prefers an explicit lot/timestamp
        # column when the file has one, and otherwise picks a direction
        # by testing which one keeps every team's running spend inside
        # its purse. See input_creation_2/auction_order.py.
        self.player_df, self.order_decision = resolve_auction_order(
            self.player_df,
            self.auction_max_purse,
        )

        # ---------------------------------------------------------
        # Participating teams
        # ---------------------------------------------------------

        self.teams = sorted(
            set(self.bid_df["Team"].dropna())
            |
            set(self.player_df["playsForTeam"].dropna())
        )

        CANONICAL_ROLES = {"BATTER", "BOWLER", "ALL-ROUNDER", "WICKETKEEPER"}

        def _normalize_role(role):
            """
            Collapse any raw role string onto one of the four
            canonical roles. Exact aliases are checked first;
            anything else falls back to keyword matching so
            compound labels (e.g. "Bowler Allrounder",
            "Batting Allrounder", "WK-Batsman") still land on the
            right bucket instead of silently becoming their own
            unrecognized category -- which previously caused
            _increment_role_count and _snapshot_auction_state's
            remaining_bowlers/remaining_allrounders to undercount
            any player whose role wasn't an exact-match string.
            """
            role = str(role).upper().strip()

            exact_aliases = {
                "WK-BATTER": "WICKETKEEPER",
                "ALLROUNDER": "ALL-ROUNDER",
            }

            if role in exact_aliases:
                return exact_aliases[role]

            if role in CANONICAL_ROLES:
                return role

            compact = role.replace("-", "").replace(" ", "")

            if "ALLROUNDER" in compact:
                return "ALL-ROUNDER"
            if "WK" in compact or "WICKETKEEPER" in compact:
                return "WICKETKEEPER"
            if "BOWL" in compact:
                return "BOWLER"
            if "BAT" in compact:
                return "BATTER"

            # Nothing matched -- fail loudly rather than silently
            # dropping this player out of every role-based count.
            raise ValueError(
                f"Unrecognized player role: {role!r}. Add it to "
                f"_normalize_role's alias/keyword handling."
            )

        self.player_df["role"] = self.player_df["role"].apply(_normalize_role)


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

                # Archetype counters live alongside the legacy role
                # counters rather than replacing them, so an ablation is a
                # column selection rather than a rebuild.
                **(empty_team_archetype_counts()
                   if self.archetype_tags is not None else {}),
            }
            for team in self.teams
        }

    def _apply_preauction_events(self):
        """
        Apply all auction events that occurred before the first player
        entered the auction: retentions, trades and draft picks.
        """
        preauction = self.player_df[
            self.player_df["auctionStatus"].isin(self.PREAUCTION_STATUSES)
        ]
        for _, player in preauction.iterrows():
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
                ~self.player_df["auctionStatus"].isin(
                    self.PREAUCTION_STATUSES
                )
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

    def _archetypes_of(self, player_id):
        if self.archetype_tags is None:
            return ()
        return tags_for(self.archetype_tags, player_id)

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
            **self._archetype_auction_state(remaining),
        }

    def _archetype_auction_state(self, remaining):
        """
        Per-archetype supply, demand and scarcity over the un-auctioned pool.

        Supply is a count of players still to come; demand is a count of teams
        that still want the archetype AND have a slot to put him in. The ratio
        is what makes a fourth right-arm seamer cheap in a year with thirty of
        them and expensive in a year with four.
        """
        if self.archetype_tags is None:
            return {}

        pool = pool_archetype_counts(
            remaining["playerId"].to_numpy(),
            self.archetype_tags,
        )
        demand = archetype_demand(self.team_state)
        scarce = scarcity(pool, demand)

        out = auction_archetype_features(pool)
        out.update({f"{a}_demand": int(demand[a]) for a in ARCHETYPES})
        out.update({f"{a}_scarcity": float(scarce[a]) for a in ARCHETYPES})
        return out
    
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

        The multiple is read from the module global on every call, so
        monkeypatching WINNER_UPPER_MULTIPLE takes effect on the next
        build. Do NOT cache it on self: an engine constructed before
        the assignment would keep the old value and the sweep would
        silently measure nothing.
        """
        multiple = globals().get("WINNER_UPPER_MULTIPLE", 2.0)

        if not multiple or multiple <= 1.0:
            raise ValueError(
                f"WINNER_UPPER_MULTIPLE must be > 1 (got {multiple!r}). "
                f"A multiple of 1 collapses the winner interval to a "
                f"point, and <1 inverts it; the loss clamps both to "
                f"lower + 1e-3 and the row becomes a near-certain "
                f"assertion carrying a huge NLL."
            )

        return winning_bid * multiple
    
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
    
        ############################################################
        # next_bid = the first later bid STRICTLY GREATER than this
        # team's own last bid, i.e. the amount it declined to match.
        #
        # This used to take literally the next row, which breaks on the
        # tie-in the source records whenever a new team joins at the
        # current standing amount: two consecutive rows carry the same
        # BidAmount. On Rishabh Pant in 2025 that is
        #
        #     ... DC 20.75 Cr | LSG 20.75 Cr | LSG 27.00 Cr
        #
        # so DC's upper bound came out as DC's own 20.75 -- a
        # zero-width interval, which then got discarded as unusable.
        # DC was not outbid by 20.75; it was outbid by LSG's next
        # advance to 27 Cr. Six such rows in 2025 alone, and they are
        # top-of-market bids on the most expensive players in the file.
        #
        # Scanning forward for a strictly larger amount also makes the
        # calculation independent of how many teams tie in at a level,
        # which is the only thing that varied here.
        ############################################################

        bids = player_bid_df.reset_index(drop=True)

        for i in range(len(bids)):
            team = bids.loc[i, "Team"]
            amount = bids.loc[i, "BidAmount"]

            later = bids.loc[i + 1:, "BidAmount"]
            above = later[later > amount]

            if len(above):
                history[team]["next_bid"] = above.iloc[0]
    
        return history
    
    def _resolve_purchase(self, player, history):
        """
        Work out who actually paid for this player, and who (if anyone)
        held the top bid but did not get him.

        Returns (winning_team, winning_bid, displaced_team).

        The old code took `playsForTeam` as the winner and left it at
        that.  On a normal sale that is right.  On an RTM it is not:
        the RTM-exercising team matches the top bid without ever
        appearing in the bid ladder, so `playsForTeam` had
        entered == False and fell into the "never entered" branch --
        the team that just paid 9 crore was labelled as valuing the
        player below his base price.  Meanwhile the genuine top bidder
        was scored as an ordinary losing bidder, and being last in the
        ladder had no next_bid, so his upper bound was NaN.

        One player, two corrupted rows, no error raised.
        """

        declared = player["playsForTeam"]
        winning_bid = player["auctionPrice"]

        entered = [t for t in self.teams if history[t]["entered"]]

        top_team = None
        if entered:
            top_team = max(entered, key=lambda t: history[t]["last_bid"])

        # Normal sale: the buyer is in the ladder.
        if declared in entered:
            return declared, winning_bid, None

        # RTM: the buyer never bid, but did pay.  The top bidder is
        # displaced rather than outbid -- nobody ever went above him,
        # so he is right-censored at his own last bid, not bounded
        # above by a bid that does not exist.
        if pd.notna(declared) and declared in self.teams:
            self.diagnostics["buyer_absent_from_ladder"].append(
                (player["playerId"], player["playerName"],
                 player["auctionStatus"], declared, top_team)
            )
            return declared, winning_bid, top_team

        # No usable buyer at all.  Emitting a winner row here would be
        # inventing one, so the caller gets None and every bidding team
        # is recorded as an interval against the sale price.
        self.diagnostics["unresolvable_buyer"].append(
            (player["playerId"], player["playerName"], declared)
        )
        return None, winning_bid, None

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

        base_price = player["basePrice"]

        winner, winning_bid, displaced = self._resolve_purchase(
            player, history
        )

        for team in self.teams:
            info = history[team]

            ####################################################
            # Never entered bidding
            ####################################################

            if not info["entered"]:

                # The RTM buyer never bids, so "did not enter" must not
                # be read as "did not want him".
                if team == winner:
                    summary[team].update({
                        "lower": winning_bid,
                        "upper": self._winner_upper_bound(winning_bid),
                        "winner": True,
                        "observation_type": "right",
                    })
                    continue

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
                    "upper": self._winner_upper_bound(winning_bid),
                    "winner": True,
                    "observation_type": "right",
                })

                continue

            ####################################################
            # Top bidder displaced by RTM
            #
            # Never outbid, so there is no upper bound on what he was
            # willing to pay -- same censoring shape as a winner, minus
            # the winner flag.
            ####################################################

            if team == displaced:
                summary[team].update({
                    "previous_bid": info["previous_bid"],
                    "last_bid": info["last_bid"],
                    "lower": info["last_bid"],
                    "upper": self._winner_upper_bound(info["last_bid"]),
                    "winner": False,
                    "observation_type": "right",
                })

                continue

            ####################################################
            # Losing bidder
            #
            # upper is the bid this team declined to match.  When the
            # ladder does not record one (this team placed the final
            # logged bid), the sale price is the correct bound: they
            # were outbid by it.  Falling through with NaN is what
            # produced the width-0.001 intervals.
            ####################################################

            upper = info["next_bid"]
            lower = info["last_bid"]

            if pd.isna(upper):
                upper = winning_bid
                self.diagnostics["next_bid_backfilled"].append(
                    (player["playerId"], player["playerName"], team)
                )

            ################################################
            # Matched at the top, not outbid.
            #
            # Under RTM the bidding stops and the original team
            # MATCHES the standing bid, so the top bidder's last
            # bid equals the sale price. That leaves upper == lower
            # and these rows were being discarded as degenerate --
            # 19 of them in 2018, 14 in 2025, and they are the most
            # informative rows in the file. Arshdeep Singh's 18 Cr
            # in 2025 was one: SRH bid it, PBKS matched it, and
            # SRH's row said only "unusable" instead of "SRH valued
            # Arshdeep at 18 Cr or more".
            #
            # A team that was matched rather than outbid was never
            # asked to go higher, so there is no upper bound on
            # what it would have paid: right-censored, exactly like
            # the winner and like the ladder-absent RTM case that
            # `displaced` already covers.
            ################################################

            matched_at_top = (
                pd.notna(lower)
                and pd.notna(winning_bid)
                and upper <= lower
                and lower >= winning_bid - 1e-6
            )

            if matched_at_top:
                summary[team].update({
                    "previous_bid": info["previous_bid"],
                    "last_bid": lower,
                    "lower": lower,
                    "upper": self._winner_upper_bound(lower),
                    "winner": False,
                    "observation_type": "right",
                })
                self.diagnostics["matched_at_top"].append(
                    (player["playerId"], player["playerName"], team, lower)
                )
                continue

            # Still unusable -> record it as unknown rather than emit a
            # degenerate interval.  A team cannot have been outbid by a
            # number at or below its own bid.
            if pd.isna(upper) or pd.isna(lower) or upper <= lower:
                self.diagnostics["dropped_bad_interval"].append(
                    (player["playerId"], player["playerName"], team,
                     lower, upper)
                )
                continue

            summary[team].update({
                "previous_bid": info["previous_bid"],
                "last_bid": lower,
                "lower": lower,
                "upper": upper,
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
        # Archetype focus
        ############################################################
        #
        # Sending every archetype's state on every row asks the model
        # to learn that wrist-spin supply is irrelevant while an
        # opener is being sold. The focus block sends only the
        # archetypes THIS player carries, reduced to fixed width
        # because the number of them varies from one to four.
        # It is team-specific, so it is built inside the team loop.
        ############################################################

        own_archetypes = self._archetypes_of(player["playerId"])
        remaining_ids = (
            self.player_df.iloc[auction_order:]["playerId"].to_numpy()
            if self.archetype_tags is not None else []
        )
        pool_counts = (
            pool_archetype_counts(remaining_ids, self.archetype_tags)
            if self.archetype_tags is not None else {}
        )
        demand_counts = (
            archetype_demand(self.team_state)
            if self.archetype_tags is not None else {}
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

            if self.archetype_tags is not None:
                team_state.update(team_archetype_features(team_state))
                team_state.update(focus_features(
                    own_archetypes,
                    team_state,
                    pool_counts,
                    demand_counts,
                    remaining_ids,
                    self.archetype_tags,
                ))

            ########################################################
            # Individual outputs
            ########################################################

            auction_state_row = {
                **player_info,
                **auction_state,
            }

            # auction_state is deliberately NOT spread in here.
            # team_state_df.columns is what becomes
            # attrs["team_state_columns"], so including auction_state
            # made all eight auction-level features members of BOTH
            # column groups -- and AuctionAdjustmentNetwork
            # concatenates both groups, so auction_order,
            # players_remaining and friends were fed to the model
            # twice, at double weight against remaining_purse.
            team_state_row = {
                **player_info,
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

        price = player["auctionPrice"]

        # Subtracting NaN turns a team's purse into NaN for the rest of
        # the auction, and every team_state feature after that point is
        # NaN too -- which the dataset then filled with 0, i.e. "this
        # team has no money left", for every subsequent player.
        if pd.isna(price):
            self.diagnostics["sale_price_missing"].append(
                (player["playerId"], player["playerName"], team)
            )
            price = 0.0

        elif (
            pd.notna(player["basePrice"])
            and price < player["basePrice"]
        ):
            self.diagnostics["sale_price_below_base"].append(
                (player["playerId"], player["playerName"],
                 price, player["basePrice"])
            )

        state["remaining_purse"] -= price

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

        ############################################################
        # Archetypes
        ############################################################
        #
        # A player increments EVERY archetype he carries, so these
        # counters do not sum to players_bought. That is deliberate:
        # they count role coverage, not bodies. Jadeja fills a
        # middle-order slot, a finisher slot, a finger-spin slot and a
        # bowling-allrounder slot at once, which is exactly why a team
        # pays for him.
        ############################################################

        if self.archetype_tags is not None:
            apply_purchase(
                state,
                self._archetypes_of(player["playerId"]),
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
        # Post-replay invariants
        #
        # A replay that ends with a team overdrawn, or over the squad
        # or overseas limit, did not describe a real auction -- most
        # often because the roster arrived in an order the engine
        # guessed wrong (see the iloc[::-1] in _normalize_inputs).
        ############################################################

        self._check_final_state()

        ############################################################
        # Convert to DataFrames
        ############################################################

        return {
            key: pd.DataFrame(value)
            for key, value in self.outputs.items()
        }

    def _check_final_state(self):

        self.final_state_violations = []

        for team, state in self.team_state.items():

            if state["remaining_purse"] < 0:
                self.final_state_violations.append(
                    (team, "negative purse", state["remaining_purse"])
                )

            if state["remaining_slots"] < 0:
                self.final_state_violations.append(
                    (team, "over squad size", state["remaining_slots"])
                )

            if state["overseas_bought"] > self.overseas_limit:
                self.final_state_violations.append(
                    (team, "over overseas limit", state["overseas_bought"])
                )

    def quality_report(self):
        """
        Everything the replay had to recover from, as counts.

        Call this after replay().  A healthy auction is all zeros with
        bid_order_source == "recorded".
        """

        report = {
            key: len(value)
            for key, value in self.diagnostics.items()
        }

        report["bid_order_source"] = self.bid_order_source
        report["auction_order_method"] = self.order_decision["method"]
        report["auction_order_column"] = self.order_decision["column"]
        report["auction_order_warning"] = self.order_decision["warning"]
        report["non_monotone_ladders"] = len(self.non_monotone_ladders)
        report["final_state_violations"] = len(
            getattr(self, "final_state_violations", [])
        )

        return report