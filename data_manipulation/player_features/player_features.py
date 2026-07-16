import numpy as np
import pandas as pd

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

    def get_player_stats(self, player, end_date):
        end_date = pd.to_datetime(end_date)

        df = self.ball_df[
            self.ball_df["match_date"] < end_date
        ]

        bat_df = df[df["batsman"] == player]

        bowl_df = df[df["bowler"] == player]


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

        fielding = self._get_fielding_stats(
            df,
            player
        )


        batting = self._compute_batting_metrics(
            batting_raw
        )

        bowling = self._compute_bowling_metrics(
            bowling_raw
        )

        return {

            "player": player,

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


    def _get_batting_raw_stats(self, bat_df, player):

        return {

            "runs":
                bat_df["runs_off_bat"].sum(),

            "balls":
                bat_df["ball_faced"].sum(),

            "outs":
                (bat_df["dismissed_player"] == player).sum(),

            "fours":
                bat_df["is_four"].sum(),

            "sixes":
                bat_df["is_six"].sum(),

            "dots":
                bat_df["is_dot"].sum(),

            "boundaries":
                bat_df["is_boundary"].sum(),

            "matches":
                bat_df["match_uid"].nunique(),

            "innings":
                bat_df.groupby("match_uid").ngroups
        }

    def _get_bowling_raw_stats(self, bowl_df):

        wickets = (
            bowl_df["is_wicket"]
            &
            ~bowl_df["dismissal_type"].isin(
                [
                    "run out",
                    "retired hurt",
                    "retired out",
                    "obstructing the field"
                ]
            )
        ).sum()

        runs = (
            bowl_df["runs_off_bat"].sum()
            + bowl_df["wide"].sum()
            + bowl_df["noball"].sum()
        )

        return {

            "balls":
                bowl_df["is_legal_ball"].sum(),

            "deliveries":
                len(bowl_df),

            "runs":
                runs,

            "wickets":
                wickets,

            "wides":
                bowl_df["wide"].sum(),

            "noballs":
                bowl_df["noball"].sum(),

            "matches":
                bowl_df["match_uid"].nunique(),

            "innings":
                bowl_df.groupby("match_uid").ngroups
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