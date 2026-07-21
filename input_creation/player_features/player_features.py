import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

class PlayerStatsAggregator:
    def __init__(self, ball_df):

        self.ball_df = ball_df.copy()

        self.ball_df["match_date"] = pd.to_datetime(
            self.ball_df["match_date"]
        )

        self.ball_df.sort_values(
            "match_date",
            inplace=True
        )

        self.ball_df.reset_index(
            drop=True,
            inplace=True
        )

        ###############################################
        # NEW
        ###############################################

        self.stats_cache = {}

        self.empty_df = self.ball_df.iloc[:0].copy()

        self.batting_groups = {
            p: g.sort_values("match_date").reset_index(drop=True)
            for p, g in self.ball_df.groupby("batsman")
        }

        self.bowling_groups = {
            p: g.sort_values("match_date").reset_index(drop=True)
            for p, g in self.ball_df.groupby("bowler")
        }

    def get_player_stats(self, player, end_date):

        end_date = pd.to_datetime(end_date)

        ##################################################
        # Cache
        ##################################################

        key = (player, end_date)

        if key in self.stats_cache:
            return self.stats_cache[key]

        ##################################################
        # Batting dataframe
        ##################################################

        bat_df = self.batting_groups.get(
            player,
            self.empty_df
        )

        if len(bat_df):

            bat_idx = bat_df["match_date"].searchsorted(
                end_date,
                side="left"
            )

            bat_df = bat_df.iloc[:bat_idx]

        ##################################################
        # Bowling dataframe
        ##################################################

        bowl_df = self.bowling_groups.get(
            player,
            self.empty_df
        )

        if len(bowl_df):

            bowl_idx = bowl_df["match_date"].searchsorted(
                end_date,
                side="left"
            )

            bowl_df = bowl_df.iloc[:bowl_idx]

        ##################################################
        # Compute statistics
        ##################################################

        batting_raw = self._get_batting_raw_stats(
            bat_df,
            player
        )

        bowling_raw = self._get_bowling_raw_stats(
            bowl_df
        )

        experience = self._get_experience(
            bat_df,
            bowl_df
        )

        # No fielding data available yet
        fielding = self._get_fielding_stats(
            None,
            player
        )

        batting = self._compute_batting_metrics(
            batting_raw
        )

        bowling = self._compute_bowling_metrics(
            bowling_raw
        )

        result = {

            "playerName": player,

            "end_date": end_date,

            "experience": experience,

            "batting": {
                "raw": batting_raw,
                "metrics": batting
            },

            "bowling": {
                "raw": bowling_raw,
                "metrics": bowling
            },

            "fielding": fielding
        }

        ##################################################
        # Save in cache
        ##################################################

        self.stats_cache[key] = result

        return result

    def _get_batting_raw_stats(self, bat_df, player):

        if len(bat_df) == 0:

            return {

                "runs": 0,
                "balls": 0,
                "outs": 0,
                "fours": 0,
                "sixes": 0,
                "dots": 0,
                "boundaries": 0,
                "matches": 0,
                "innings": 0
            }

        ##################################################
        # Convert once to NumPy
        ##################################################

        runs = bat_df["runs_off_bat"].to_numpy()
        balls = bat_df["ball_faced"].to_numpy()
        fours = bat_df["is_four"].to_numpy()
        sixes = bat_df["is_six"].to_numpy()
        dots = bat_df["is_dot"].to_numpy()
        boundaries = bat_df["is_boundary"].to_numpy()
        dismissed = bat_df["dismissed_player"].to_numpy()
        matches = bat_df["match_uid"].to_numpy()

        return {

            "runs":
                runs.sum(),

            "balls":
                balls.sum(),

            "outs":
                np.sum(dismissed == player),

            "fours":
                fours.sum(),

            "sixes":
                sixes.sum(),

            "dots":
                dots.sum(),

            "boundaries":
                boundaries.sum(),

            "matches":
                np.unique(matches).size,

            "innings":
                np.unique(matches).size
        }

    def _get_bowling_raw_stats(self, bowl_df):

        if len(bowl_df) == 0:

            return {

                "balls": 0,
                "deliveries": 0,
                "runs": 0,
                "wickets": 0,
                "wides": 0,
                "noballs": 0,
                "matches": 0,
                "innings": 0
            }

        ##################################################
        # Convert once to NumPy
        ##################################################

        runs_off_bat = bowl_df["runs_off_bat"].to_numpy()
        wides = bowl_df["wide"].to_numpy()
        noballs = bowl_df["noball"].to_numpy()
        legal = bowl_df["is_legal_ball"].to_numpy()

        is_wicket = bowl_df["is_wicket"].to_numpy()
        dismissal = bowl_df["dismissal_type"].to_numpy()

        matches = bowl_df["match_uid"].to_numpy()

        ##################################################
        # Bowling wickets
        ##################################################

        invalid_dismissals = np.isin(
            dismissal,
            [
                "run out",
                "retired hurt",
                "retired out",
                "obstructing the field"
            ]
        )

        wickets = np.sum(
            is_wicket & (~invalid_dismissals)
        )

        ##################################################
        # Runs conceded
        ##################################################

        runs = (
            runs_off_bat.sum()
            + wides.sum()
            + noballs.sum()
        )

        return {

            "balls":
                legal.sum(),

            "deliveries":
                len(bowl_df),

            "runs":
                runs,

            "wickets":
                wickets,

            "wides":
                wides.sum(),

            "noballs":
                noballs.sum(),

            "matches":
                np.unique(matches).size,

            "innings":
                np.unique(matches).size
        }


    def _get_experience(self, bat_df, bowl_df):

        matches = pd.concat(
            [
                bat_df["match_uid"],
                bowl_df["match_uid"]
            ]
        ).nunique()

        return {

            "matches": matches,

            "batting_matches":
                bat_df["match_uid"].nunique(),

            "bowling_matches":
                bowl_df["match_uid"].nunique(),

            "batting_innings":
                bat_df.groupby("match_uid").ngroups,

            "bowling_innings":
                bowl_df.groupby("match_uid").ngroups
        }

    def _get_fielding_stats(self, df, player):

        # Current parquet has no fielder column.

        return {

            "catches": None,

            "run_outs": None,

            "stumpings": None
        }


    def _compute_batting_metrics(self, raw):

        balls = raw["balls"]
        outs = raw["outs"]

        return {

            "average":
                raw["runs"] / outs
                if outs else None,

            "strike_rate":
                100 * raw["runs"] / balls
                if balls else None,

            "boundary_percentage":
                raw["boundaries"] / balls
                if balls else None,

            "dot_ball_percentage":
                raw["dots"] / balls
                if balls else None
        }


    def _compute_bowling_metrics(self, raw):

        balls = raw["balls"]
        wickets = raw["wickets"]

        return {

            "economy":
                6 * raw["runs"] / balls
                if balls else None,

            "average":
                raw["runs"] / wickets
                if wickets else None,

            "strike_rate":
                balls / wickets
                if wickets else None
        }

class PlayerFeatureBuilder:
    def __init__(self, stats_aggregator):
        self.stats_aggregator = stats_aggregator

    def build_feature_table(self, players, auction_date):
        """
        Build a feature table for all players available in an auction.

        Parameters
        ----------
        players : list[str]
            List of player names.

        auction_date : str or datetime

        Returns
        ----------
        feature_df : pd.DataFrame
            One row per player.

        player_to_idx : dict
            Mapping from player name to row index.
        """

        rows = []

        for player in tqdm(players):

            stats = self.stats_aggregator.get_player_stats(
                player,
                auction_date
            )

            row = self._flatten_player_stats(stats)

            rows.append(row)

        feature_df = pd.DataFrame(rows)

        feature_df.fillna(0.0, inplace=True)

        feature_df.sort_values(
            "playerName",
            inplace=True
        )

        feature_df.reset_index(
            drop=True,
            inplace=True
        )

        return feature_df

    def _flatten_player_stats(self, stats):

        return {

            ##################################################
            # Identity
            ##################################################

            "playerName":
                stats["playerName"],

            ##################################################
            # Experience
            ##################################################

            "matches":
                stats["experience"]["matches"],

            "batting_matches":
                stats["experience"]["batting_matches"],

            "bowling_matches":
                stats["experience"]["bowling_matches"],

            "batting_innings":
                stats["experience"]["batting_innings"],

            "bowling_innings":
                stats["experience"]["bowling_innings"],

            ##################################################
            # Batting Raw
            ##################################################

            "bat_runs":
                stats["batting"]["raw"]["runs"],

            "bat_balls":
                stats["batting"]["raw"]["balls"],

            "bat_outs":
                stats["batting"]["raw"]["outs"],

            "bat_fours":
                stats["batting"]["raw"]["fours"],

            "bat_sixes":
                stats["batting"]["raw"]["sixes"],

            "bat_dots":
                stats["batting"]["raw"]["dots"],

            ##################################################
            # Batting Metrics
            ##################################################

            "bat_average":
                stats["batting"]["metrics"]["average"],

            "bat_strike_rate":
                stats["batting"]["metrics"]["strike_rate"],

            "boundary_percentage":
                stats["batting"]["metrics"]["boundary_percentage"],

            "dot_ball_percentage":
                stats["batting"]["metrics"]["dot_ball_percentage"],

            ##################################################
            # Bowling Raw
            ##################################################

            "bowl_balls":
                stats["bowling"]["raw"]["balls"],

            "bowl_runs":
                stats["bowling"]["raw"]["runs"],

            "bowl_wickets":
                stats["bowling"]["raw"]["wickets"],

            "bowl_wides":
                stats["bowling"]["raw"]["wides"],

            "bowl_noballs":
                stats["bowling"]["raw"]["noballs"],

            ##################################################
            # Bowling Metrics
            ##################################################

            "bowl_average":
                stats["bowling"]["metrics"]["average"],

            "economy":
                stats["bowling"]["metrics"]["economy"],

            "bowl_strike_rate":
                stats["bowling"]["metrics"]["strike_rate"]
        }

    def build_player_features(self, player, auction_date):

        stats = self.stats_aggregator.get_player_stats(
            player,
            auction_date
        )

        return {

            # Experience
            "matches":
                stats["experience"]["matches"],
            "batting_matches":
                stats["experience"]["batting_matches"],
            "bowling_matches":
                stats["experience"]["bowling_matches"],

            # Batting metrics
            "bat_average":
                stats["batting"]["metrics"]["average"],
            "bat_strike_rate":
                stats["batting"]["metrics"]["strike_rate"],
            "boundary_percentage":
                stats["batting"]["metrics"]["boundary_percentage"],
            "dot_ball_percentage":
                stats["batting"]["metrics"]["dot_ball_percentage"],

            # Bowling metrics
            "bowl_average":
                stats["bowling"]["metrics"]["average"],
            "economy":
                stats["bowling"]["metrics"]["economy"],
            "bowl_strike_rate":
                stats["bowling"]["metrics"]["strike_rate"],

            # Keep the raw stats as well
            "bat_runs":
                stats["batting"]["raw"]["runs"],
            "bat_balls":
                stats["batting"]["raw"]["balls"],
            "bat_outs":
                stats["batting"]["raw"]["outs"],
            "bowl_balls":
                stats["bowling"]["raw"]["balls"],
            "bowl_runs":
                stats["bowling"]["raw"]["runs"],
            "bowl_wickets":
                stats["bowling"]["raw"]["wickets"]
        }