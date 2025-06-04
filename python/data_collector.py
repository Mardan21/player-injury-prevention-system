"""
Data collection and processing module for player injury prevention system.
FIXED VERSION - Aggregates duplicate players before database insertion.
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
                print(f"Successfully loaded data using {encoding} encoding")
                print(f"Loaded {len(df)} player records")
                print(f"Columns: {df.columns.tolist()}")
                
                # Check for duplicates in raw data
                duplicate_count = df['Player'].duplicated().sum()
                unique_players = df['Player'].nunique()
                print(f"Raw data: {unique_players} unique players, {duplicate_count} duplicates")
                
                return df
            except UnicodeDecodeError:
                continue
          
        raise ValueError("Could not read the file with any of the attempted encodings")
    
    def aggregate_duplicate_players(self, df):
        """
        Aggregate duplicate players before database insertion.
        This is the KEY FIX - prevents multiple player_id values for same player.
        """
        print("\n=== AGGREGATING DUPLICATE PLAYERS ===")
        
        # Check for duplicates
        duplicate_players = df['Player'].value_counts()
        duplicates = duplicate_players[duplicate_players > 1]
        
        if len(duplicates) == 0:
            print("No duplicate players found.")
            return df
        
        print(f"Found {len(duplicates)} players with multiple records:")
        for player, count in duplicates.head(10).items():
            teams = df[df['Player'] == player]['Squad'].unique()
            print(f"  {player}: {count} records ({', '.join(teams)})")
        
        print(f"\nAggregating {len(duplicates)} duplicate players...")
        
        # Define aggregation rules
        aggregation_rules = {
            # Keep first value (these should be same across records)
            # NOTE: 'Player' is the grouping key, so don't include it here
            'Nation': 'first',  # Nationality  
            'Age': 'first',     # Age
            'Born': 'first',    # Birth year
            'Pos': 'first',     # Position
            'Comp': 'first',    # Competition (keep first, could be combined)
            
            # Combine team names with separator
            'Squad': lambda x: ' + '.join(sorted(x.unique())),
            
            # Sum total stats (what we want - combined performance)
            'MP': 'sum',        # Match appearances
            'Starts': 'sum',    # Games started
            'Min': 'sum',       # Minutes played
            '90s': 'sum',       # 90s played
            'Goals': 'sum',     # Goals
            'Shots': 'sum',     # Shots
            'SoT': 'sum',       # Shots on target
            'Assists': 'sum',   # Assists
            
            # For percentages and ratios, use weighted average by minutes
            'SoT%': lambda x: self._weighted_average(x, df.loc[x.index, 'Min']),
            'G/Sh': lambda x: self._weighted_average(x, df.loc[x.index, 'Min']),
            'G/SoT': lambda x: self._weighted_average(x, df.loc[x.index, 'Min']),
            
            # Sum all other counting stats
            'PasTotCmp': 'sum',     # Passes completed
            'PasTotAtt': 'sum',     # Passes attempted
            'PasTotDist': 'sum',    # Pass distance
            'PasTotPrgDist': 'sum', # Progressive pass distance
            'PasShoCmp': 'sum',     # Short passes completed
            'PasShoAtt': 'sum',     # Short passes attempted
            'PasMedCmp': 'sum',     # Medium passes completed
            'PasMedAtt': 'sum',     # Medium passes attempted
            'PasLonCmp': 'sum',     # Long passes completed
            'PasLonAtt': 'sum',     # Long passes attempted
            'PasAss': 'sum',        # Assisted shots
            'PPA': 'sum',           # Passes into penalty area
            'CrsPA': 'sum',         # Crosses into penalty area
            'PasProg': 'sum',       # Progressive passes
            'PasAtt': 'sum',        # Pass attempts
            'PasLive': 'sum',       # Live ball passes
            'PasDead': 'sum',       # Dead ball passes
            'PasFK': 'sum',         # Free kick passes
            'TB': 'sum',            # Through balls
            'Sw': 'sum',            # Switches
            'PasCrs': 'sum',        # Crosses
            'TI': 'sum',            # Throw-ins
            'CK': 'sum',            # Corner kicks
            'PasCmp': 'sum',        # Passes completed
            'PasOff': 'sum',        # Offside passes
            'PasBlocks': 'sum',     # Passes blocked
            'SCA': 'sum',           # Shot creating actions
            'GCA': 'sum',           # Goal creating actions
            'Tkl': 'sum',           # Tackles
            'TklWon': 'sum',        # Tackles won
            'TklDef3rd': 'sum',     # Tackles defensive third
            'TklMid3rd': 'sum',     # Tackles middle third
            'TklAtt3rd': 'sum',     # Tackles attacking third
            'TklDri': 'sum',        # Dribblers tackled
            'TklDriAtt': 'sum',     # Dribbler tackle attempts
            'TklDriPast': 'sum',    # Times dribbled past
            'Blocks': 'sum',        # Blocks
            'BlkSh': 'sum',         # Shots blocked
            'BlkPass': 'sum',       # Passes blocked
            'Int': 'sum',           # Interceptions
            'Tkl+Int': 'sum',       # Tackles + Interceptions
            'Clr': 'sum',           # Clearances
            'Err': 'sum',           # Errors
            'Touches': 'sum',       # Touches
            'ToAtt': 'sum',         # Take-on attempts
            'ToSuc': 'sum',         # Successful take-ons
            'ToTkl': 'sum',         # Times tackled during take-on
            'Carries': 'sum',       # Carries
            'CarTotDist': 'sum',    # Total carry distance
            'CarPrgDist': 'sum',    # Progressive carry distance
            'CarProg': 'sum',       # Progressive carries
            'Car3rd': 'sum',        # Carries into final third
            'CPA': 'sum',           # Carries into penalty area
            'CarMis': 'sum',        # Miscontrols
            'CarDis': 'sum',        # Dispossessed
            'Rec': 'sum',           # Passes received
            'RecProg': 'sum',       # Progressive passes received
            'CrdY': 'sum',          # Yellow cards
            'CrdR': 'sum',          # Red cards
            '2CrdY': 'sum',         # Second yellow cards
            'Fls': 'sum',           # Fouls committed  
            'Fld': 'sum',           # Fouls drawn
            'Off': 'sum',           # Offsides
            'Crs': 'sum',           # Crosses
            'TklW': 'sum',          # Tackles won (different column?)
            'PKwon': 'sum',         # Penalties won
            'PKcon': 'sum',         # Penalties conceded
            'OG': 'sum',            # Own goals
            'Recov': 'sum',         # Recoveries
            'AerWon': 'sum',        # Aerial duels won
            'AerLost': 'sum',       # Aerial duels lost
        }
        
        # Add default 'sum' for any numeric columns not specified
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in aggregation_rules and col not in ['Rk']:  # Skip rank column
                aggregation_rules[col] = 'sum'
        
        # Add default 'first' for any other columns not specified
        for col in df.columns:
            if col not in aggregation_rules and col != 'Player':  # Skip Player (grouping key)
                aggregation_rules[col] = 'first'
        
        try:
            # Aggregate duplicate players
            print("Performing aggregation...")
            aggregated_df = df.groupby('Player').agg(aggregation_rules).reset_index()
            
            # Recalculate percentage stats after aggregation
            aggregated_df = self._recalculate_percentage_stats(aggregated_df)
            
            print(f"✅ Aggregation complete:")
            print(f"   Original records: {len(df)}")
            print(f"   After aggregation: {len(aggregated_df)}")
            print(f"   Reduced by: {len(df) - len(aggregated_df)} records")
            print(f"   Unique players: {aggregated_df['Player'].nunique()}")
            
            # Show some aggregated examples
            combined_teams = aggregated_df[aggregated_df['Squad'].str.contains(' + ', na=False)]
            if len(combined_teams) > 0:
                print(f"\nPlayers with combined teams: {len(combined_teams)}")
                for _, player in combined_teams.head(5).iterrows():
                    print(f"  {player['Player']}: {player['Squad']}")
            
            return aggregated_df
            
        except Exception as e:
            print(f"❌ Error during aggregation: {e}")
            print("Returning original data without aggregation")
            return df
    
    def _weighted_average(self, values, weights):
        """Calculate weighted average for percentage stats"""
        try:
            # Handle NaN values
            mask = ~(pd.isna(values) | pd.isna(weights))
            if mask.sum() == 0:
                return 0
            
            clean_values = values[mask]
            clean_weights = weights[mask]
            
            total_weight = clean_weights.sum()
            if total_weight == 0:
                return clean_values.mean()
            
            return (clean_values * clean_weights).sum() / total_weight
        except:
            return values.mean() if len(values) > 0 else 0
    
    def _recalculate_percentage_stats(self, df):
        """Recalculate percentage stats after aggregation"""
        print("Recalculating percentage statistics...")
        
        # Pass completion percentages
        if 'PasTotCmp' in df.columns and 'PasTotAtt' in df.columns:
            df['PasTotCmp%'] = (df['PasTotCmp'] / df['PasTotAtt'].replace(0, np.nan) * 100).fillna(0)
        
        if 'PasShoCmp' in df.columns and 'PasShoAtt' in df.columns:
            df['PasShoCmp%'] = (df['PasShoCmp'] / df['PasShoAtt'].replace(0, np.nan) * 100).fillna(0)
        
        if 'PasMedCmp' in df.columns and 'PasMedAtt' in df.columns:
            df['PasMedCmp%'] = (df['PasMedCmp'] / df['PasMedAtt'].replace(0, np.nan) * 100).fillna(0)
        
        if 'PasLonCmp' in df.columns and 'PasLonAtt' in df.columns:
            df['PasLonCmp%'] = (df['PasLonCmp'] / df['PasLonAtt'].replace(0, np.nan) * 100).fillna(0)
        
        # Shot percentages
        if 'SoT' in df.columns and 'Shots' in df.columns:
            df['SoT%'] = (df['SoT'] / df['Shots'].replace(0, np.nan) * 100).fillna(0)
        
        # Goals per shot
        if 'Goals' in df.columns and 'Shots' in df.columns:
            df['G/Sh'] = (df['Goals'] / df['Shots'].replace(0, np.nan)).fillna(0)
        
        if 'Goals' in df.columns and 'SoT' in df.columns:
            df['G/SoT'] = (df['Goals'] / df['SoT'].replace(0, np.nan)).fillna(0)
        
        # Take-on success rate
        if 'ToSuc' in df.columns and 'ToAtt' in df.columns:
            df['ToSuc%'] = (df['ToSuc'] / df['ToAtt'].replace(0, np.nan) * 100).fillna(0)
        
        # Tackle success rate  
        if 'TklDri' in df.columns and 'TklDriAtt' in df.columns:
            df['TklDri%'] = (df['TklDri'] / df['TklDriAtt'].replace(0, np.nan) * 100).fillna(0)
        
        # Aerial duel win rate
        if 'AerWon' in df.columns and 'AerLost' in df.columns:
            total_duels = df['AerWon'] + df['AerLost']
            df['AerWon%'] = (df['AerWon'] / total_duels.replace(0, np.nan) * 100).fillna(0)
        
        return df
    
    def process_data(self, df):
        """Process and split data into players and performance tables"""
        
        # CRITICAL FIX: Aggregate duplicate players FIRST
        df_aggregated = self.aggregate_duplicate_players(df)
        
        # Now process the aggregated data
        print(f"\nProcessing {len(df_aggregated)} aggregated player records...")
        
        # Basic player information
        players_df = df_aggregated[['Player', 'Age', 'Pos', 'Squad', 'Comp']].copy()
        players_df = players_df.rename(columns={
            'Player': 'player_name',
            'Age': 'age',
            'Pos': 'position', 
            'Squad': 'team',
            'Comp': 'league'
        })
        
        # Add missing columns for schema compatibility
        if 'Nation' in df_aggregated.columns:
            players_df['nation'] = df_aggregated['Nation']
        else:
            players_df['nation'] = None
            
        if 'Born' in df_aggregated.columns:
            players_df['year_born'] = df_aggregated['Born']
        else:
            players_df['year_born'] = None

        # Performance data (rename all columns to match schema)
        performance_df = df_aggregated.copy()
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
            'PasShoCmp': 'short_passes_completed',
            'PasShoAtt': 'short_passes_attempted',
            'PasShoCmp%': 'short_pass_completion_percentage',
            'PasMedCmp': 'medium_passes_completed',
            'PasMedAtt': 'medium_passes_attempted',
            'PasMedCmp%': 'medium_pass_completion_percentage',
            'PasLonCmp': 'long_passes_completed',
            'PasLonAtt': 'long_passes_attempted',
            'PasLonCmp%': 'long_pass_completion_percentage',
            'Shots': 'shots',
            'SoT': 'shots_on_target',
            'SoT%': 'shots_on_target_percentage',
            'Goals': 'goals',
            'G/Sh': 'goals_per_shot',
            'G/SoT': 'goals_per_shot_on_target',
            'Assists': 'assists',
            'PasAss': 'assisted_shots',
            'PPA': 'completed_passes_into_18',
            'CrsPA': 'completed_crosses_into_18',
            'PasProg': 'progressive_passes_count',
            'RecProg': 'progressive_pass_received',
            'PasLive': 'live_ball_passes',
            'PasDead': 'dead_ball_passes',
            'PasFK': 'passes_attempted_from_free_kicks',
            'PasOff': 'passes_offside',
            'Crs': 'crosses',
            'Off': 'offsides',
            'PasBlocks': 'passes_blocked_by_opponent',
            'SCA': 'shot_creating_actions',
            'ScaPassLive': 'shot_creating_actions_from_live_ball',
            'ScaPassDead': 'shot_creating_actions_from_dead_ball',
            'ScaDrib': 'successful_dribbles_leading_to_shot',
            'ScaSh': 'shots_leading_to_another_shot',
            'ScaFld': 'fouls_drawn_leading_to_shot',
            'ScaDef': 'defensive_actions_leading_to_shot',
            'GCA': 'goal_creating_actions',
            'GcaPassLive': 'live_ball_passes_leading_to_goal',
            'GcaPassDead': 'dead_ball_passes_leading_to_goal',
            'GcaDrib': 'successful_dribbles_leading_to_goal',
            'GcaSh': 'shots_leading_to_goal_scoring_shot',
            'GcaFld': 'fouls_drawn_leading_to_goal',
            'GcaDef': 'defensive_actions_leading_to_goal',
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
            'PKwon': 'penalty_kicks_won',
            'PKcon': 'penalty_kicks_conceded',
            'PKatt': 'penalty_kicks_attempted',
            'OG': 'own_goals',
            'Recov': 'number_of_loose_balls_recovered'
        })

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
            'own_goals', 'number_of_loose_balls_recovered'
        ]

        # Keep only the columns that exist and match our schema
        available_columns = [col for col in schema_columns if col in performance_df.columns]
        performance_df = performance_df[available_columns]
        
        print(f"✅ Data processing complete:")
        print(f"   Players: {len(players_df)} unique players")
        print(f"   Performance columns: {len(available_columns)}")

        return players_df, performance_df

    def load_to_database(self, players_df, performance_df):
        """Load dataframe to PostgreSQL database"""
        try:
            print("\n=== LOADING TO DATABASE ===")
            
            # First, clear existing data to prevent duplicates
            # IMPORTANT: Delete in correct order due to foreign key constraints
            with engine.connect() as conn:
                # Delete child tables first (foreign key references)
                conn.execute(text("DELETE FROM injury_labels"))
                conn.execute(text("DELETE FROM player_performance"))
                conn.execute(text("DELETE FROM players"))
                conn.commit()
                print("Cleared existing data from database (including injury_labels)")
            
            # Load the players (now deduplicated!)
            players_df.to_sql('players', engine, if_exists='append', index=False)
            print(f"✅ Loaded {len(players_df)} unique players to players table")

            # Get the player_ids for the performance data
            with engine.connect() as conn:
                player_ids = pd.read_sql(
                    "SELECT player_id, player_name FROM players", 
                    conn
                )
            
            # Add player_name to performance_df for merging
            performance_df['player_name'] = players_df['player_name']
            
            # Merge player_ids with performance data
            performance_df = performance_df.merge(
                player_ids,
                on='player_name',
                how='left'
            )

            # Verify we have player_ids for all records
            missing_ids = performance_df['player_id'].isnull().sum()
            if missing_ids > 0:
                print(f"⚠️  Warning: {missing_ids} performance records could not be matched to players")
                print("Unmatched players:", performance_df[performance_df['player_id'].isnull()]['player_name'].unique())
            else:
                print("✅ All performance records matched to player IDs")

            # Convert numeric columns to appropriate types
            numeric_columns = performance_df.select_dtypes(include=['float64']).columns
            for col in numeric_columns:
                if 'percentage' in col.lower():
                    performance_df[col] = performance_df[col].round(2)
                elif 'distance' in col.lower():
                    performance_df[col] = performance_df[col].round(2)
                else:
                    performance_df[col] = performance_df[col].round(0).astype('Int64')

            # Select columns for database
            performance_columns = ['player_id'] + [col for col in performance_df.columns if col not in ['player_id', 'player_name']]
            performance_df = performance_df[performance_columns]

            # Load the performance data
            performance_df.to_sql('player_performance', engine, if_exists='append', index=False)
            print(f"✅ Loaded {len(performance_df)} performance records to player_performance table")
            
            # Verify the fix worked
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT player_name, COUNT(DISTINCT player_id) as id_count 
                    FROM players 
                    GROUP BY player_name 
                    HAVING COUNT(DISTINCT player_id) > 1
                """)).fetchall()
                
                if len(result) == 0:
                    print("✅ SUCCESS: No duplicate player_ids found!")
                else:
                    print(f"⚠️  WARNING: {len(result)} players still have multiple IDs:")
                    for row in result[:5]:
                        print(f"   {row[0]}: {row[1]} IDs")

            return True
            
        except Exception as e:
            print(f"❌ Error loading to database: {e}")
            return False

if __name__ == "__main__":
    collector = PlayerDataCollector()
    
    print("=== LOADING PLAYER STATS WITH AGGREGATION ===")
    df = collector.load_player_stats()
    
    print("\n=== PROCESSING DATA ===")
    players_df, performance_df = collector.process_data(df)
    
    print("\n=== LOADING TO DATABASE ===")
    success = collector.load_to_database(players_df, performance_df)
    
    if success:
        print("\n🎉 DATA LOADING COMPLETED SUCCESSFULLY!")
        print("The duplicate player issue has been fixed at the source!")
        print("\n⚠️  IMPORTANT: You need to regenerate injury labels after this fix!")
        print("\nNext steps:")
        print("1. Run: python python/simple_injury_generator.py")
        print("2. Run: python models/injury_predictor.py")
        print("3. Check your new predictions - no more duplicates!")
    else:
        print("\n❌ Data loading failed. Check the error messages above.")