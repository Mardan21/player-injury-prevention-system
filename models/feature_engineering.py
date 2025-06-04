"""
Feature engineering module for injury risk prediction
FIXED VERSION - Aligns with SQL analytics and fixes target variable issues
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


class PlayerDataAggregator:
    """Handle aggregation of duplicate players due to transfers"""
    
    def __init__(self):
        pass
    
    def aggregate_duplicate_players(self, df):
        """
        Aggregate stats for players who appear multiple times (due to transfers)
        Combines their performance across different clubs in the same season
        """
        print("Checking for duplicate players...")
        
        # Check if we have duplicates
        duplicate_players = df['player_name'].value_counts()
        duplicates = duplicate_players[duplicate_players > 1]
        
        if len(duplicates) == 0:
            print("No duplicate players found.")
            return df
        
        print(f"Found {len(duplicates)} players with multiple records:")
        for player, count in duplicates.head(10).items():
            teams = df[df['player_name'] == player]['team'].unique()
            print(f"  {player}: {count} records ({', '.join(teams)})")
        
        print(f"\nAggregating {len(duplicates)} duplicate players...")
        
        # Separate duplicates from unique players
        unique_players = df[~df['player_name'].isin(duplicates.index)].copy()
        duplicate_data = df[df['player_name'].isin(duplicates.index)].copy()
        
        # Define aggregation rules
        aggregation_rules = {
            # Keep first value (these should be same across records)
            'player_id': 'first',
            'age': 'first', 
            'position': 'first',
            'league': 'first',
            'nationality': 'first',
            
            # Combine team names
            'team': lambda x: ' + '.join(sorted(x.unique())),
            
            # Sum total stats
            'minutes_played': 'sum',
            'games_played': 'sum', 
            'games_started': 'sum',
            'goals': 'sum',
            'assists': 'sum',
            'shots': 'sum',
            'shots_on_target': 'sum',
            'passes_completed': 'sum',
            'passes_attempted': 'sum',
            'tackles': 'sum',
            'interceptions': 'sum',
            'yellow_cards': 'sum',
            'red_cards': 'sum',
            'fouls_committed': 'sum',
            'fouls_drawn': 'sum',
            'touches': 'sum',
            'carries': 'sum',
            'progressive_carries': 'sum',
            'aerial_duels_won': 'sum',
            'aerial_duels_lost': 'sum',
            
            # For distance stats, sum them
            'total_passes_distance': 'sum',
            'progressive_pass_distance': 'sum',
            'total_carry_distance': 'sum',
            'progressive_carry_distance': 'sum',
            
            # For other counting stats, sum them
            'blocks': 'sum',
            'clearances': 'sum',
            'progressive_passes_count': 'sum',
            'shot_creating_actions': 'sum',
            'goal_creating_actions': 'sum',
            
            # For injury/risk features, take weighted average by minutes played
            'injury_target': lambda x: (x * duplicate_data.loc[x.index, 'minutes_played']).sum() / duplicate_data.loc[x.index, 'minutes_played'].sum() if 'minutes_played' in duplicate_data.columns else x.mean(),
            'injury_prone_score': lambda x: (x * duplicate_data.loc[x.index, 'minutes_played']).sum() / duplicate_data.loc[x.index, 'minutes_played'].sum() if 'minutes_played' in duplicate_data.columns else x.mean(),
            'fatigue_score': lambda x: (x * duplicate_data.loc[x.index, 'minutes_played']).sum() / duplicate_data.loc[x.index, 'minutes_played'].sum() if 'minutes_played' in duplicate_data.columns else x.mean(),
        }
        
        # Add default 'sum' for any numeric columns not specified
        numeric_columns = duplicate_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in aggregation_rules and col not in ['player_id']:
                aggregation_rules[col] = 'sum'
        
        # Add default 'first' for any other columns
        for col in duplicate_data.columns:
            if col not in aggregation_rules and col != 'player_name':
                aggregation_rules[col] = 'first'
        
        # Aggregate duplicate players
        try:
            aggregated_duplicates = duplicate_data.groupby('player_name').agg(aggregation_rules).reset_index()
            
            # Recalculate per-game and percentage stats after aggregation
            aggregated_duplicates = self._recalculate_per_game_stats(aggregated_duplicates)
            
            # Combine with unique players
            final_df = pd.concat([unique_players, aggregated_duplicates], ignore_index=True)
            
            print(f"✅ Aggregation complete:")
            print(f"   Original records: {len(df)}")
            print(f"   After aggregation: {len(final_df)}")
            print(f"   Reduced by: {len(df) - len(final_df)} records")
            
            return final_df
            
        except Exception as e:
            print(f"❌ Error during aggregation: {e}")
            print("Returning original data without aggregation")
            return df
    
    def _recalculate_per_game_stats(self, df):
        """Recalculate per-game and percentage stats after aggregation"""
        
        # Recalculate percentage stats
        if 'passes_completed' in df.columns and 'passes_attempted' in df.columns:
            df['pass_completion_percentage'] = (df['passes_completed'] / df['passes_attempted'].replace(0, 1) * 100).fillna(0)
        
        if 'shots_on_target' in df.columns and 'shots' in df.columns:
            df['shots_on_target_percentage'] = (df['shots_on_target'] / df['shots'].replace(0, 1) * 100).fillna(0)
        
        if 'aerial_duels_won' in df.columns and 'aerial_duels_lost' in df.columns:
            total_duels = df['aerial_duels_won'] + df['aerial_duels_lost']
            df['aerial_duels_won_percentage'] = (df['aerial_duels_won'] / total_duels.replace(0, 1) * 100).fillna(0)
        
        # Recalculate per-shot stats
        if 'goals' in df.columns and 'shots' in df.columns:
            df['goals_per_shot'] = (df['goals'] / df['shots'].replace(0, 1)).fillna(0)
        
        if 'goals' in df.columns and 'shots_on_target' in df.columns:
            df['goals_per_shot_on_target'] = (df['goals'] / df['shots_on_target'].replace(0, 1)).fillna(0)
        
        return df


class InjuryFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        
    def create_base_features(self, df):
        """Create base features from player performance data"""
        
        # Create a copy to avoid modifying original
        features_df = df.copy()
        
        # 1. Workload Features (aligned with SQL analytics)
        features_df['minutes_per_game'] = features_df['minutes_played'] / features_df['games_played'].replace(0, 1)
        features_df['high_minutes_flag'] = (features_df['minutes_per_game'] > 75).astype(int)
        features_df['workload_consistency'] = features_df['games_started'] / features_df['games_played'].replace(0, 1)
        
        # 2. Physical Load Features (matching SQL DAP90 calculation)
        features_df['total_physical_actions'] = (
            features_df['tackles'] + 
            features_df['aerial_duels_won'] + 
            features_df['aerial_duels_lost'] +
            features_df['blocks'] +
            features_df['clearances']
        )
        
        features_df['physical_actions_per_game'] = (
            features_df['total_physical_actions'] / features_df['games_played'].replace(0, 1)
        )
        
        # 3. Defensive Actions Per 90 (DAP90) - from SQL analytics
        features_df['dap90'] = (
            features_df['tackles_won'] * 0.5 +
            features_df['interceptions'] * 0.5 +
            features_df['clearances'] * 0.1 + 
            features_df['number_of_dribblers_tackled'] * 0.5 +
            features_df['shots_blocked'] * 0.1 +
            features_df['passes_blocked'] * 0.1 +
            features_df['aerial_duels_won'] * 0.25 + 
            features_df['number_of_loose_balls_recovered'] * 0.1
        )
        
        # 4. Creative Actions Per 90 (CAP90) - from SQL analytics
        features_df['cap90'] = (
            features_df['assists'] * 0.7 +
            features_df['assisted_shots'] * 0.5 +
            features_df['progressive_passes_count'] * 0.1 +
            features_df['progressive_carries'] * 0.05 +
            features_df['pass_completion_percentage'] * 0.1 +
            features_df['completed_crosses_into_18'] * 0.01 +
            features_df['shot_creating_actions'] * 0.25 +
            features_df['successful_dribbles_leading_to_shot'] * 0.25
        )
        
        # 5. Offensive Actions Per 90 (OAP90) - from SQL analytics
        features_df['oap90'] = (
            features_df['assists'] * 1.0 + 
            features_df['shots_on_target'] * 0.5 + 
            features_df['goals_per_shot_on_target'] * 0.8 +
            features_df['shots_leading_to_another_shot'] * 0.1 +
            features_df['successful_dribbles_leading_to_goal'] * 0.2 +
            features_df['shots_leading_to_goal_scoring_shot'] * 0.5
        )
        
        # 6. Composite Workload Score - from SQL analytics
        features_df['composite_workload_score'] = (
            features_df['minutes_per_game'] * 0.4 +
            features_df['dap90'] * 0.4 + 
            features_df['oap90'] * 0.3 +
            features_df['cap90'] * 0.25
        )
        
        # 7. High Intensity Features
        features_df['high_intensity_actions'] = (
            features_df['progressive_carries'] +
            features_df['progressive_passes_count'] +
            features_df['tackles_won'] +
            features_df['number_of_dribblers_tackled']
        )
        
        features_df['intensity_per_minute'] = (
            features_df['high_intensity_actions'] / features_df['minutes_played'].replace(0, 1)
        )
        
        # 8. High Intensity Score (from SQL analytics)
        features_df['high_intensity_score'] = (
            features_df['progressive_carries'] * 0.25 +
            features_df['progressive_carry_distance'] * 0.25 +
            features_df['successful_dribbles_leading_to_shot'] * 0.2 +
            features_df['number_attempts_take_on_defender'] * 0.15 +
            (features_df['aerial_duels_won'] + features_df['aerial_duels_lost']) * 0.15 
        )
        
        # 9. Fatigue Indicators (enhanced)
        features_df['error_rate'] = (
            features_df['errors_leading_to_opponent_shot'] +
            features_df['dispossessed_carries'] +
            features_df['missed_carries']
        ) / features_df['minutes_played'].replace(0, 1) * 90
        
        features_df['technical_failure_rate'] = (
            features_df['dispossessed_carries'] + features_df['missed_carries']
        ) / features_df['carries'].replace(0, 1)
        
        # 10. Fatigue Score (from SQL analytics)
        features_df['fatigue_score'] = (
            (features_df['minutes_played'] * 1.0 / features_df['games_played'].replace(0, 1)) * 0.4 +
            (features_df['games_played'] * 1.0 / 295) * (features_df['minutes_played'] * 1.0 / features_df['games_played'].replace(0, 1)) * 0.2 +
            features_df['high_intensity_score'] * 0.25 +
            (features_df['tackles'] +
             features_df['interceptions'] +
             features_df['clearances'] +
             features_df['blocks']) * 0.15
        )
        
        # 11. Contact Exposure (enhanced)
        features_df['contact_events'] = (
            features_df['fouls_committed'] +
            features_df['fouls_drawn'] +
            features_df['tackles'] +
            features_df['aerial_duels_won'] +
            features_df['aerial_duels_lost']
        )
        
        features_df['contact_per_game'] = (
            features_df['contact_events'] / features_df['games_played'].replace(0, 1)
        )
        
        # 12. Recovery Demand Score (from SQL analytics)
        features_df['recovery_demand'] = (
            features_df['tackles'] * 2.0 +
            features_df['fouls_committed'] * 1.5 +
            features_df['aerial_duels_won'] * 1.2 +
            features_df['aerial_duels_lost'] * 1.2 +
            features_df['progressive_carries'] * 1.5 +
            features_df['number_attempts_take_on_defender'] * 1.3 +
            features_df['shots_blocked'] * 1.5
        )
        
        # 13. Age-related risk factors (aligned with SQL analytics)
        features_df['age_risk'] = features_df['age'].apply(self._calculate_age_risk)
        features_df['age_workload_interaction'] = (
            features_df['age_risk'] * features_df['minutes_per_game']
        )
        
        # 14. Position-specific risk (enhanced for hybrid positions)
        features_df['is_defender'] = features_df['position'].str.contains('DF|CB|LB|RB', na=False).astype(int)
        features_df['is_midfielder'] = features_df['position'].str.contains('MF|CM|DM|AM', na=False).astype(int)
        features_df['is_forward'] = features_df['position'].str.contains('FW|CF|LW|RW', na=False).astype(int)
        features_df['is_hybrid'] = ((features_df['is_defender'] + features_df['is_midfielder'] + features_df['is_forward']) > 1).astype(int)
        
        # 15. Performance efficiency
        features_df['defensive_efficiency'] = (
            features_df['tackles_won'] / features_df['tackles'].replace(0, 1)
        )
        
        features_df['passing_efficiency'] = (
            features_df['pass_completion_percentage'] / 100
        )
        
        features_df['dribble_success_rate'] = (
            features_df['number_defenders_taken_on_successfully'] / 
            features_df['number_attempts_take_on_defender'].replace(0, 1)
        )
        
        # 16. Disciplinary risk (enhanced from SQL analytics)
        features_df['discipline_score'] = (
            features_df['fouls_committed'] * 0.5 + 
            features_df['yellow_cards'] * 1.0 + 
            features_df['red_cards'] * 3.0
        )
        
        features_df['card_risk'] = (
            features_df['yellow_cards'] + 
            features_df['red_cards'] * 3 +
            features_df.get('second_yellow_card', 0) * 2
        ) / features_df['games_played'].replace(0, 1)
        
        features_df['aggressive_play_index'] = (
            features_df['fouls_committed'] / features_df['minutes_played'].replace(0, 1) * 90
        )
        
        return features_df
    
    def create_advanced_features(self, df):
        """Create advanced interaction and polynomial features"""
        
        features_df = df.copy()
        
        # 1. Interaction features (enhanced)
        features_df['workload_age_risk'] = (
            features_df['minutes_per_game'] * features_df['age_risk']
        )
        
        features_df['intensity_fatigue'] = (
            features_df['intensity_per_minute'] * features_df['error_rate']
        )
        
        features_df['contact_age_risk'] = (
            features_df['contact_per_game'] * features_df['age_risk']
        )
        
        # 2. Composite risk scores from SQL analytics
        features_df['physical_risk_score'] = (
            features_df['physical_actions_per_game'] * 0.3 +
            features_df['contact_per_game'] * 0.3 +
            features_df['recovery_demand'] * 0.4
        ) / 100
        
        features_df['technical_risk_score'] = (
            features_df['error_rate'] * 0.4 +
            features_df['technical_failure_rate'] * 0.3 +
            features_df['aggressive_play_index'] * 0.3
        )
        
        # 3. Injury Risk Index (from SQL analytics)
        features_df['injury_risk_index'] = (
            features_df['fatigue_score'] * 0.4 +
            features_df['discipline_score'] * 0.2 +
            features_df['age'] * 0.1 
        )
        
        # 4. Ratio features
        features_df['offensive_defensive_ratio'] = (
            (features_df['carries'] + features_df['passes_attempted']) /
            (features_df['tackles'] + features_df['interceptions']).replace(0, 1)
        )
        
        features_df['creation_finishing_ratio'] = (
            features_df['shot_creating_actions'] /
            (features_df['shots'] + features_df['goals']).replace(0, 1)
        )
        
        # 5. Progressive actions (SQL analytics inspired)
        features_df['progressive_actions_per_90'] = (
            features_df['progressive_carries'] + features_df['progressive_passes_count']
        )
        
        features_df['high_intensity_defensive_actions'] = (
            features_df['tackles_won'] + features_df['number_of_dribblers_tackled']
        )
        
        # 6. Team and position aggregations
        try:
            features_df['minutes_variance'] = features_df.groupby('team')['minutes_played'].transform('std').fillna(0)
            features_df['team_avg_fouls'] = features_df.groupby('team')['fouls_committed'].transform('mean')
            features_df['position_avg_minutes'] = features_df.groupby('position')['minutes_played'].transform('mean')
        except Exception:
            # Fallback if groupby fails
            features_df['minutes_variance'] = 0
            features_df['team_avg_fouls'] = features_df['fouls_committed'].mean()
            features_df['position_avg_minutes'] = features_df['minutes_played'].mean()
        
        # 7. Outlier indicators
        for col in ['minutes_played', 'tackles', 'fouls_committed']:
            if col in features_df.columns:
                mean = features_df[col].mean()
                std = features_df[col].std()
                if std > 0:
                    features_df[f'{col}_zscore'] = (features_df[col] - mean) / std
                    features_df[f'{col}_outlier'] = (np.abs(features_df[f'{col}_zscore']) > 2).astype(int)
                else:
                    features_df[f'{col}_zscore'] = 0
                    features_df[f'{col}_outlier'] = 0
        
        return features_df
    
    def create_injury_features(self, df):
        """Create features from injury data (real or synthetic)"""
        
        # Ensure all injury features exist with defaults
        injury_features = [
            'career_injuries',
            'total_days_injured', 
            'injury_prone_score',
            'injuries_per_year',
            'muscle_injury_count',
            'knee_injury_count',
            'recent_injury_count',
            'days_since_injury'
        ]
        
        for feature in injury_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Create additional injury-based features
        df['has_injury_history'] = (df['career_injuries'] > 0).astype(int)
        df['chronic_injury_risk'] = (df['career_injuries'] > 3).astype(int)
        df['recent_injury_flag'] = (df['days_since_injury'] < 90).astype(int)
        
        # Injury type risk scores
        df['muscle_injury_risk'] = df['muscle_injury_count'] / (df['career_injuries'] + 1)
        df['joint_injury_risk'] = (
            df['knee_injury_count'] + df.get('ankle_injury_count', 0)
        ) / (df['career_injuries'] + 1)
        
        # Recovery pattern features
        df['avg_recovery_time'] = df['total_days_injured'] / (df['career_injuries'] + 1)
        df['quick_recovery'] = (df['avg_recovery_time'] < 21).astype(int)
        
        # Workload-injury interaction
        df['workload_injury_risk'] = df['minutes_per_game'] * df['injury_prone_score']
        df['age_injury_risk'] = df['age_risk'] * df['injury_prone_score']
        
        return df
    
    def create_target_variable(self, df):
        """Create target variable from injury data or synthetic approach"""
        
        if 'injury_target' not in df.columns:
            print("Warning: Creating synthetic injury target based on risk factors.")
            
            # Create synthetic target based on multiple risk factors
            # This mimics the SQL query approach for synthetic data
            conditions = [
                (df['minutes_per_game'] > 85) & (df['age'] > 30),
                (df['fouls_committed'] > 2.0) & (df['yellow_cards'] > 0.3),
                (df['minutes_per_game'] > 75) & (df['physical_actions_per_game'] > 15),
                (df['error_rate'] > 1.0) & (df['age'] > 28),
                (df['recovery_demand'] > 100) & (df['age'] > 29)
            ]
            
            # Player is at risk if any condition is met
            df['injury_target'] = 0
            for condition in conditions:
                df.loc[condition, 'injury_target'] = 1
            
            # Add some randomness to make it more realistic
            np.random.seed(42)
            random_injuries = np.random.random(len(df)) < 0.05  # 5% baseline injury rate
            df.loc[random_injuries, 'injury_target'] = 1
        
        return df
    
    def select_features(self, X, y, k=30):
        """Select top k features using statistical tests"""
        
        if len(X.columns) <= k:
            print(f"Using all {len(X.columns)} features (less than k={k})")
            return X, X.columns.tolist()
        
        try:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            feature_scores = pd.DataFrame({
                'feature': X.columns,
                'score': selector.scores_,
                'selected': selector.get_support()
            }).sort_values('score', ascending=False)
            
            selected_features = feature_scores[feature_scores['selected']]['feature'].tolist()
            self.feature_importance = feature_scores
            
            print(f"Selected {len(selected_features)} features out of {len(X.columns)}")
            
            return X[selected_features], selected_features
            
        except Exception as e:
            print(f"Warning: Feature selection failed: {e}")
            print("Using all features.")
            return X, X.columns.tolist()
    
    def prepare_features(self, df, target_col='injury_target', scale=True):
        """Prepare final feature set for modeling"""
        
        # Select feature columns (exclude identifiers and target)
        exclude_cols = [
            'player_id', 'player_name', 'team', 'season', target_col,
            'performance_id', 'created_at', 'league', 'calculated_at'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables
        categorical_cols = ['position']
        for col in categorical_cols:
            if col in df.columns and col in feature_cols:
                encoded_col = f'{col}_encoded'

                # Only create if doesnt exist
                if encoded_col not in df.columns:
                    le = LabelEncoder()
                    df[encoded_col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Use existing encoder
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[encoded_col] = df[col].astype(str).map(
                        dict(zip(le.classes_, le.transform(le.classes_)))
                    ).fillna(-1)
                
                # Only modify feature_cols once
                if encoded_col not in feature_cols:
                    feature_cols.append(encoded_col)
                if col in feature_cols:
                    feature_cols.remove(col)

        # Remove original categorical columns
        for col in categorical_cols:
            if col in feature_cols:
                feature_cols.remove(col)
        
        # Select features that exist in the dataframe
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features]
        
        # Handle any remaining non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Fill any NaN values
        X = X.fillna(0)
        
        # Replace infinity values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features if requested
        if scale:
            try:
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
                return X_scaled
            except Exception as e:
                print(f"Warning: Scaling failed: {e}. Using unscaled features.")
                return X
        
        return X
    
    def _calculate_age_risk(self, age):
        """Calculate age-based risk factor (aligned with SQL analytics)"""
        if age < 20:
            return 0.35
        elif age <= 23:
            return 0.32
        elif age <= 27:
            return 0.29
        elif age <= 30:
            return 0.32
        elif age <= 33:
            return 0.35
        else:
            return 0.38
    
    def get_feature_importance_report(self):
        """Generate feature importance report"""
        if not hasattr(self, 'feature_importance') or self.feature_importance.empty:
            return "No feature importance data available. Run select_features() first."
        
        report = "Top 20 Most Important Features:\n"
        report += "=" * 50 + "\n"
        
        for idx, row in self.feature_importance.head(20).iterrows():
            report += f"{row['feature']:<40} Score: {row['score']:>10.2f}\n"
        
        return report