"""
Data collection and processing module for player injury prevention system.
Handles loading and preprocessing of player statistics and injury data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from db_config import engine
from sqlalchemy import text
import os

class PlayerDataCollector:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_path = os.path.join(self.base_path, 'data', 'raw')
        self.processed_data_path = os.path.join(self.base_path, 'data', 'processed')
        
    def load_player_stats(self):
        """Load player statistics from Kaggle dataset"""
        file_path = os.path.join(self.raw_data_path, 'player_stats_2023.csv')
        print(f"Loading data from: {file_path}")

        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=';')

                # # Map columns to match database schema
                # df = df.rename(columns={
                #     'Player': 'player_name',
                #     'Age': 'age',
                #     'Position': 'position',
                #     'Squad': 'team'
                # })

                print(f"Successfully loaded data using {encoding} encoding")
                print(f"Loaded {len(df)} players")
                print(f"Columns: {df.columns.tolist()}")
                return df
            except UnicodeDecodeError:
                continue
          
        raise ValueError("Could not read the file with any of the attempted encodings")
    
    def process_data(self, df):
        """Process and split data into players and performance tables"""

        # Basic player information
        players_df = df[['Player', 'Age', 'Pos', 'Squad', 'Comp']].copy()
        players_df = players_df.rename(columns={
            'Player': 'player_name',
            'Age': 'age',
            'Pos': 'position',
            'Squad': 'team',
            'Comp': 'league'
        })

        # Performance data
        performance_df = df.copy()
        performance_df = performance_df.rename(columns={
            'Min': 'minutes_played',
            'MP': 'games_played',
            'Starts': 'games_started',
            'Carries': 'carries',
            'CarTotDist': 'total_carry_distance',
            'CarPrgDist': 'progressive_carry_distance',
            'CarProg': 'progressive_carries',
            'CPA': 'carries_into_18',
            'CarDis': 'dispossessed_carries',
            'CarMis': 'missed_carries',
            'AerWon': 'aerial_duels_won',
            'AerLost': 'aerial_duels_lost',
            'AerWon%': 'aerial_duels_won_percentage',
            'PasTotCmp': 'passes_completed',
            'PasTotAtt': 'passes_attempted',
            'PasTotCmp%': 'pass_completion_percentage',
            'PasTotDist': 'total_passes_distance',
            'PasTotPrgDist': 'progressive_pass_distance',
            'PasShoCmp': 'short_passes_completed',  # short pass is between 5 to 15 yards
            'PasShoAtt': 'short_passes_attempted',
            'PasShoCmp%': 'short_pass_completion_percentage',
            'PasMedCmp': 'medium_passes_completed',  # medium pass is between 15 to 30 yards
            'PasMedAtt': 'medium_passes_attempted',
            'PasMedCmp%': 'medium_pass_completion_percentage',
            'PasLonCmp': 'long_passes_completed',  # long pass is more than 30 yards
            'PasLonAtt': 'long_passes_attempted',
            'PasLonCmp%': 'long_pass_completion_percentage',
            'Shots': 'shots',
            'SoT': 'shots_on_target',
            'SoT%': 'shots_on_target_percentage',
            'Goals': 'goals',
            'G/Sh': 'goals_per_shot',
            'G/SoT': 'goals_per_shot_on_target',
            'Assists': 'assists',
            'PasAss': 'assisted_shots',  # passes that directly lead to a shot
            'PPA': 'completed_passes_into_18',
            'CrsPA': 'completed_crosses_into_18',
            'PasProg': 'progressive_passes_count',
            'PasLive': 'live_ball_passes',
            'PasDead': 'dead_ball_passes',
            'PasFK': 'passes_attempted_from_free_kicks',
            'PasOff': 'passes_offside',
            'Off': 'offsides',
            'Crs': 'crosses',
            'PasBlocks': 'passes_blocked_by_opponent',  # Blocked by the opponent who was standing it the path of the pass
            'SCA': 'shot_creating_actions',  # passes that directly lead to a shot
            'ScaPassLive': 'shot_creating_actions_from_live_ball',  # passes that directly lead to a shot from a live ball
            'ScaPassDead': 'shot_creating_actions_from_dead_ball',  # passes that directly lead to a shot from a dead ball
            'ScaDrib': 'successful_dribbles_leading_to_shot',  # dribbles that directly lead to a shot
            'ScaSh': 'shots_leading_to_another_shot',  # shots that lead to another shot
            'ScaFld': 'fouls_drawn_leading_to_shot',  # fouls that directly lead to a shot
            'ScaDef': 'defensive_actions_leading_to_shot',  # defensive actions that directly lead to a shot
            'GCA': 'goal_creating_actions',  # actions that directly lead to a goal
            'GcaPassLive': 'live_ball_passes_leading_to_goal',  # passes that directly lead to a shot from a live ball
            'GcaPassDead': 'dead_ball_passes_leading_to_goal',  # passes that directly lead to a shot from a dead ball
            'GcaDrib': 'successful_dribbles_leading_to_goal',  # dribbles that directly lead to a shot
            'GcaSh': 'shots_leading_to_goal_scoring_shot',  # shots that lead to another shot
            'GcaFld': 'fouls_drawn_leading_to_goal',  # fouls that directly lead to a shot
            'GcaDef': 'defensive_actions_leading_to_goal',  # defensive actions that directly lead to a shot
            'Tkl': 'tackles',
            'TklWon': 'tackles_won',
            'TklDef3rd': 'tackles_in_defensive_third',
            'TklMid3rd': 'tackles_in_middle_third',
            'TklAtt3rd': 'tackles_in_attacking_third',
            'TklDri': 'number_of_dribblers_tackled',
            'TklDri%': 'percentage_of_dribblers_tackled',
            'TklDriPast': 'number_of_times_dribbled_past_by_opponent',            
            'Int': 'interceptions',
            'Tkl+Int': 'number_of_tackles_and_interceptions',
            'Blocks': 'blocks',
            'BlkSh': 'shots_blocked',
            'BlkPass': 'passes_blocked',
            'Clr': 'clearances',
            'Err': 'errors_leading_to_opponent_shot',
            'Touches': 'touches',
            'ToAtt': 'number_attempts_take_on_defender',
            'ToSuc': 'number_defenders_taken_on_successfully',
            'ToSuc%': 'percentage_of_take_on_success',
            'ToTkl': 'number_times_tackled_by_defender_during_take_on',
            'ToTkl%': 'percentage_tackled_by_defender_during_take_on',
            'Fls': 'fouls_committed',
            'Fld': 'fouls_drawn',
            'CrdY': 'yellow_cards',
            'CrdR': 'red_cards',
            '2CrdY': 'second_yellow_card',
            'Rec': 'pass_received',
            'RecProg': 'progressive_pass_received',
            'PKwon': 'penalty_kicks_won',
            'PKcon': 'penalty_kicks_conceded',
            'PKatt': 'penalty_kicks_attempted',
            'OG': 'own_goals',
            'Recov': 'number_of_loose_balls_recovered'
        })

        # Add season column
        performance_df['season'] = '2023-2024'

        # Get list of columns that match our schema
        schema_columns = [
            'minutes_played', 'games_played', 'games_started',
            'carries', 'total_carry_distance', 'progressive_carry_distance',
            'progressive_carries', 'carries_into_18', 'dispossessed_carries',
            'missed_carries', 'aerial_duels_won', 'aerial_duels_lost',
            'aerial_duels_won_percentage', 'passes_completed', 'passes_attempted',
            'pass_completion_percentage', 'total_passes_distance',
            'progressive_pass_distance', 'progressive_passes_count',
            'progressive_pass_received', 'short_passes_completed',
            'short_passes_attempted', 'short_pass_completion_percentage',
            'medium_passes_completed', 'medium_passes_attempted',
            'medium_pass_completion_percentage', 'long_passes_completed',
            'long_passes_attempted', 'long_pass_completion_percentage',
            'shots', 'shots_on_target', 'shots_on_target_percentage',
            'goals', 'goals_per_shot', 'goals_per_shot_on_target',
            'assists', 'assisted_shots', 'completed_passes_into_18',
            'completed_crosses_into_18', 'live_ball_passes',
            'dead_ball_passes', 'passes_attempted_from_free_kicks',
            'passes_offside', 'crosses', 'offsides',
            'passes_blocked_by_opponent', 'shot_creating_actions',
            'shot_creating_actions_from_live_ball',
            'shot_creating_actions_from_dead_ball',
            'successful_dribbles_leading_to_shot',
            'shots_leading_to_another_shot', 'fouls_drawn_leading_to_shot',
            'defensive_actions_leading_to_shot', 'goal_creating_actions',
            'live_ball_passes_leading_to_goal',
            'dead_ball_passes_leading_to_goal',
            'successful_dribbles_leading_to_goal',
            'shots_leading_to_goal_scoring_shot',
            'fouls_drawn_leading_to_goal',
            'defensive_actions_leading_to_goal', 'tackles', 'tackles_won',
            'tackles_in_defensive_third', 'tackles_in_middle_third',
            'tackles_in_attacking_third', 'number_of_dribblers_tackled',
            'percentage_of_dribblers_tackled',
            'number_of_times_dribbled_past_by_opponent', 'interceptions',
            'number_of_tackles_and_interceptions', 'blocks', 'shots_blocked',
            'passes_blocked', 'clearances', 'errors_leading_to_opponent_shot',
            'fouls_committed', 'fouls_drawn', 'yellow_cards', 'red_cards', 
            'second_yellow_card', 'touches', 'number_attempts_take_on_defender',
            'number_defenders_taken_on_successfully', 'percentage_of_take_on_success',
            'number_times_tackled_by_defender_during_take_on',
            'percentage_tackled_by_defender_during_take_on', 'pass_received', 
            'penalty_kicks_won', 'penalty_kicks_conceded', 'penalty_kicks_attempted', 
            'own_goals', 'number_of_loose_balls_recovered', 'season'
        ]

        # Keep only the columns that match our schema
        performance_df = performance_df[schema_columns]

        return players_df, performance_df
    

    def load_to_database(self, players_df, performance_df):
        """Load dataframe to PostgreSQL database"""
        try:
            # First, clear existing data to prevent duplicates
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM player_performance"))
                conn.execute(text("DELETE FROM players"))
                conn.commit()
            
            # Load the players
            players_df.to_sql('players', engine, if_exists='append', index=False)
            print(f"Loaded {len(players_df)} records to players table")

            # Get the player_ids for the performance data
            with engine.connect() as conn:
                player_ids = pd.read_sql(
                    "SELECT player_id, player_name FROM players WHERE player_name IN %s",
                    conn,
                    params=(tuple(players_df['player_name']),)
                )
            
            # Add player_name back to performance_df for merging
            performance_df['player_name'] = players_df['player_name']
            
            # Merge player_ids with performance data
            performance_df = performance_df.merge(
                player_ids,
                on='player_name',
                how='left'
            )

            # Verify we have player_ids for all records
            if performance_df['player_id'].isnull().any():
                print("Warning: Some performance records could not be matched to players")
                print(performance_df[performance_df['player_id'].isnull()]['player_name'].unique())

            # Convert numeric columns to appropriate types
            numeric_columns = performance_df.select_dtypes(include=['float64']).columns
            for col in numeric_columns:
                if 'percentage' in col.lower():
                    performance_df[col] = performance_df[col].round(2)
                elif 'distance' in col.lower():
                    performance_df[col] = performance_df[col].round(2)
                else:
                    performance_df[col] = performance_df[col].round(0).astype('Int64')

            performance_columns = ['player_id'] + [col for col in performance_df.columns if col not in ['player_id', 'player_name']]
            performance_df = performance_df[performance_columns]

            # Load the performance data
            performance_df.to_sql('player_performance', engine, if_exists='append', index=False)
            print(f"Loaded {len(performance_df)} records to player_performance table")

            return True
        except Exception as e:
            print(f"Error loading to database: {e}")
            return False
    
    # def load_to_database(self, players_df, performance_df):
    #     """Load dataframe to PostgreSQL database"""
    #     try:
    #         # Load the players
    #         players_df.to_sql('players', engine, if_exists='append', index=False)
    #         print(f"Loaded {len(players_df)} records to players table")

    #         # Get the player_ids for the performance data
    #         with engine.connect() as conn:
    #             conn.execute("DELETE FROM player_performance")
    #             conn.execute("DELETE FROM players")
    #             conn.commit()
                
    #             player_ids = pd.read_sql(
    #                 "SELECT player_id, player_name FROM players WHERE player_name IN %s",
    #                 conn,
    #                 params=(tuple(players_df['player_name']),)
    #             )
            
    #         # Add player_name back to performance_df for merging
    #         performance_df['player_name'] = players_df['player_name']
            
    #         # Merge player_ids with performance data
    #         performance_df = performance_df.merge(
    #             player_ids,
    #             on='player_name',
    #             how='left'
    #         )

    #         # Verify we have player_ids for all records
    #         if performance_df['player_id'].isnull().any():
    #             print("Warning: Some performance records could not be matched to players")
    #             print(performance_df[performance_df['player_id'].isnull()]['player_name'].unique())

    #         # Convert numeric columns to appropriate types
    #         numeric_columns = performance_df.select_dtypes(include=['float64']).columns
    #         for col in numeric_columns:
    #             if 'percentage' in col.lower():
    #                 performance_df[col] = performance_df[col].round(2)
    #             elif 'distance' in col.lower():
    #                 performance_df[col] = performance_df[col].round(2)
    #             else:
    #                 performance_df[col] = performance_df[col].round(0).astype('Int64')


    #         performance_columns = ['player_id'] + [col for col in performance_df.columns if col not in ['player_id', 'player_name']]

    #         performance_df = performance_df[performance_columns]

    #         # Load the performance data
    #         performance_df.to_sql('player_performance', engine, if_exists='append', index=False)
    #         print(f"Loaded {len(performance_df)} records to player_performance table")

    #         return True
    #     except Exception as e:
    #         print(f"Error loading to database: {e}")
    #         return False

if __name__ == "__main__":
    collector = PlayerDataCollector()
    df = collector.load_player_stats()
    players_df, performance_df = collector.process_data(df)
    collector.load_to_database(players_df, performance_df)