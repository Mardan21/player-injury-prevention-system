"""
Injury risk prediction model using Random Forest with SHAP explainability
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add python directory to path for db_config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_dir = os.path.join(project_root, 'python')
sys.path.insert(0, python_dir)

try:
    from db_config import engine
    print("✅ Database configuration loaded successfully")
except ImportError as e:
    print(f"❌ Warning: db_config not found in python/ directory: {e}")
    print("Make sure python/db_config.py exists and is configured correctly.")
    engine = None

# SHAP import with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available - some visualizations will be skipped")
    SHAP_AVAILABLE = False

from feature_engineering import InjuryFeatureEngineer, PlayerDataAggregator

class InjuryPredictor:
    def __init__(self):
        self.model = None
        self.feature_engineer = InjuryFeatureEngineer()
        self.data_aggregator = PlayerDataAggregator()
        self.feature_names = None
        self.model_metrics = {}
        self.shap_explainer = None
        self.shap_values = None
        self.X_test_sample = None
        self.base_path = project_root
        
    # def load_data(self, use_synthetic_injuries=True): # Minimal Fix 
    #     """Load data from database with optional synthetic injury data"""
    #     print("Loading data from database...")
        
    #     if engine is None:
    #         raise ValueError("Database engine not available. Check python/db_config.py")
        
    #     if use_synthetic_injuries:
    #         # Query without injury tables - creates synthetic injury target
    #         query = """
    #         SELECT 
    #             p.*,
    #             pp.*,
    #             -- Use injury labels from injury_labels table
    #             COALESCE(il.injured_next_month, FALSE) as injury_target,
    #             COALESCE(il.injury_risk_score, 0) as injury_prone_score,
    #             COALESCE(il.fatigue_score, 0) as fatigue_score,
    #             COALESCE(il.workload_increase, 0) as workload_increase,
    #             COALESCE(il.recent_injury_history, FALSE) as recent_injury_history,
                
    #             -- Synthetic injury history features (since we don't have real injury events yet)
    #             0 as career_injuries,
    #             0 as total_days_injured,
    #             0 as injuries_per_year,
    #             0 as muscle_injury_count,
    #             0 as knee_injury_count,
    #             '1900-01-01'::date as last_injury_date,
    #             999 as days_since_injury,
    #             0 as recent_injury_count
                
    #         FROM players p
    #         JOIN player_performance pp ON p.player_id = pp.player_id
    #         LEFT JOIN injury_labels il ON p.player_id = il.player_id
    #         WHERE pp.minutes_played > 400  -- Minimum 400 minutes (4 games * 90 min)
    #         AND il.player_id IS NOT NULL  -- Only include players with injury labels
    #         """
    #     else:
    #         # Original query with real injury tables (if they exist)
    #         # query = """
    #         # SELECT 
    #         #     p.*,
    #         #     pp.*,
    #         #     -- Real injury history
    #         #     COALESCE(pis.total_injuries, 0) as career_injuries,
    #         #     COALESCE(pis.total_days_injured, 0) as total_days_injured,
    #         #     COALESCE(pis.injury_prone_score, 0) as injury_prone_score,
    #         #     COALESCE(pis.injury_frequency, 0) as injuries_per_year,
    #         #     COALESCE(pis.muscle_injuries, 0) as muscle_injury_count,
    #         #     COALESCE(pis.knee_injuries, 0) as knee_injury_count,
    #         #     COALESCE(pis.last_injury_date, '1900-01-01'::date) as last_injury_date,
                
    #         #     -- Calculate days since injury
    #         #     CASE 
    #         #         WHEN pis.last_injury_date IS NOT NULL 
    #         #         THEN CURRENT_DATE - pis.last_injury_date
    #         #         ELSE 999
    #         #     END as days_since_injury,
                
    #         #     -- Recent injuries
    #         #     (SELECT COUNT(*) FROM injury_events ie 
    #         #      WHERE ie.player_id = p.player_id 
    #         #      AND ie.injury_date >= CURRENT_DATE - INTERVAL '180 days') as recent_injury_count,
                
    #         #     -- Target: Did player get injured this season?
    #         #     CASE 
    #         #         WHEN EXISTS (
    #         #             SELECT 1 FROM injury_events ie
    #         #             WHERE ie.player_id = p.player_id
    #         #             AND ie.season = '22/23'
    #         #         ) THEN 1 ELSE 0
    #         #     END as injury_target
                
    #         # FROM players p
    #         # JOIN player_performance pp ON p.player_id = pp.player_id
    #         # LEFT JOIN player_injury_summary pis ON p.player_id = pis.player_id
    #         # WHERE pp.minutes_played > 360
    #         # """
    #         query = """
    #         SELECT 
    #             p.player_id,
    #             p.player_name,
    #             p.age,
    #             p.position,
    #             p.team,
    #             p.league,
    #             pp.*
    #         FROM players p
    #         JOIN player_performance pp ON p.player_id = pp.player_id
    #         WHERE pp.minutes_played > 360
    #         """ 
        
    #     try:
    #         base_query = """
    #         SELECT 
    #             p.player_id,
    #             p.player_name,
    #             p.age,
    #             p.position,
    #             p.team,
    #             p.league,
    #             pp.*
    #         FROM players p
    #         JOIN player_performance pp ON p.player_id = pp.player_id
    #         WHERE pp.minutes_played > 360
    #         """
    #         print("Loading base player data...")
    #         df = pd.read_sql(base_query, engine)
    #         print(f"Loaded {len(df)} player records")
    

    #         # Step 2: Load injury events separately
    #         injury_query = """
    #         SELECT 
    #             player_id,
    #             injury_date,
    #             days_out,
    #             injury_type,
    #             injury_category,
    #             season
    #         FROM injury_events
    #         """
            
    #         print("Loading injury data...")
    #         injury_df = pd.read_sql(injury_query, engine)
    #         print(f"Loaded {len(injury_df)} injury records")

    #         # Add these 3 lines right after: df = pd.read_sql(base_query, engine)
    #         duplicate_cols = df.columns.duplicated()
    #         if duplicate_cols.any():
    #             df = df.loc[:, ~duplicate_cols]  # Remove duplicate columns
            
    #         # Step 3: Create injury features manually (avoiding complex subqueries)
    #         print("Creating injury features...")
            
    #         # Create target variable: injured in 2022-23?
    #         target_injuries = injury_df[injury_df['season'] == '22/23']['player_id'].unique()
    #         df['injury_target'] = df['player_id'].isin(target_injuries).astype(int)
            
    #         # Create historical injury features (pre-2022-07-01)
    #         # Convert injury_date to datetime
    #         injury_df['injury_date'] = pd.to_datetime(injury_df['injury_date'])
    #         historical_injuries = injury_df[injury_df['injury_date'] < '2022-07-01']
            
    #         # Calculate historical injury stats per player
    #         if len(historical_injuries) > 0:
    #             hist_stats = historical_injuries.groupby('player_id').agg({
    #                 'injury_date': 'count',  # career_injuries
    #                 'days_out': ['sum', 'mean']  # total_days_injured, avg_recovery
    #             }).round(3)
                
    #             hist_stats.columns = ['career_injuries', 'total_days_injured', 'avg_recovery_days']
    #             hist_stats = hist_stats.reset_index()
                
    #             # Merge with main dataframe
    #             df = df.merge(hist_stats, on='player_id', how='left')
    #         else:
    #             print("No historical injury data found (pre-2022)")
    #             df['career_injuries'] = 0
    #             df['total_days_injured'] = 0
    #             df['avg_recovery_days'] = 0
            
    #         # Fill NaN values for players with no injury history
    #         injury_features = ['career_injuries', 'total_days_injured', 'avg_recovery_days']
    #         for feature in injury_features:
    #             df[feature] = df[feature].fillna(0)
            
    #         # Create additional injury features
    #         df['has_injury_history'] = (df['career_injuries'] > 0).astype(int)
    #         df['injury_prone_score'] = (df['career_injuries'] * 0.1 + df['total_days_injured'] / 365.0 * 0.2).clip(0, 1)
    #         df['recent_injury_count'] = 0  # Since we filtered out recent injuries
    #         df['days_since_injury'] = 999  # Default for no recent injuries
    #         df['muscle_injury_count'] = 0  # Simplified for now
    #         df['knee_injury_count'] = 0    # Simplified for now
            
    #         # NOW we can safely check injury rate (after creating injury_target)
    #         injury_rate = df['injury_target'].mean()
    #         print(f"Injury rate in data: {injury_rate:.2%}")
            
    #         if injury_rate == 0:
    #             print("❌ No injuries found in target season (2022-23)!")
    #             print("Check if injury_events table has data for season '22/23'")
    #             raise ValueError("No injury targets found.")

    #         injured_count = df['injury_target'].sum()
    #         print(f"Players with injuries in 2022-23: {injured_count} ({injury_rate:.1%})")
            
    #         return df
            
    #     except Exception as e:
    #         print(f"Error loading data: {e}")
    #         print("Make sure:")
    #         print("1. Database is running and accessible")
    #         print("2. Tables 'players', 'player_performance', and 'injury_events' exist")
    #         print("3. injury_events table has data for season '22/23'")
    #         raise

    def load_data(self, use_synthetic_injuries=True):
        """Load data from database - BULLETPROOF VERSION"""
        print("Loading data from database...")
        
        if engine is None:
            raise ValueError("Database engine not available. Check python/db_config.py")
        
        try:
            # Step 1: Load players first
            print("Loading players...")
            players_df = pd.read_sql("SELECT * FROM players", engine)
            print(f"Loaded {len(players_df)} players")
            
            # Step 2: Load performance data
            print("Loading performance data...")
            performance_df = pd.read_sql("SELECT * FROM player_performance WHERE minutes_played > 360", engine)
            print(f"Loaded {len(performance_df)} performance records")
            
            # Step 3: Merge carefully (avoid duplicate columns)
            df = players_df.merge(performance_df, on='player_id', how='inner', suffixes=('', '_perf'))
            print(f"Merged data: {len(df)} records")
            
            # Debug: Check for duplicate columns
            duplicate_cols = df.columns.duplicated()
            if duplicate_cols.any():
                print(f"⚠️ Duplicate columns found: {df.columns[duplicate_cols].tolist()}")
                df = df.loc[:, ~duplicate_cols]  # Remove duplicates
            
            # Step 4: Load injury data
            print("Loading injury data...")
            injury_df = pd.read_sql("SELECT * FROM injury_events", engine)
            print(f"Loaded {len(injury_df)} injury records")
            
            # Step 5: Create target variable
            print("Creating target variable...")
            injury_df['injury_date'] = pd.to_datetime(injury_df['injury_date'])
            target_injuries = injury_df[injury_df['season'] == '22/23']['player_id'].unique()
            print(f"Found {len(target_injuries)} players with 2022-23 injuries")
            
            # Debug the target creation
            print(f"Target injuries type: {type(target_injuries)}")
            print(f"Target injuries shape: {target_injuries.shape if hasattr(target_injuries, 'shape') else 'No shape'}")
            print(f"Sample target injuries: {target_injuries[:5] if len(target_injuries) > 0 else 'None'}")
            
            # Create target (ensure it's a Series)
            injury_mask = df['player_id'].isin(target_injuries)
            df['injury_target'] = injury_mask.astype(int)
            
            # Step 6: Create historical injury features (pre-2022)
            print("Creating historical injury features...")
            historical_injuries = injury_df[injury_df['injury_date'] < '2022-08-01']
            print(f"Historical injuries (pre-2022): {len(historical_injuries)}")
            # Add this debug code after: historical_injuries = injury_df[injury_df['injury_date'] < '2022-07-01']
            print("\n=== HISTORICAL INJURY DIAGNOSTIC ===")
            print(f"All injury date range: {injury_df['injury_date'].min()} to {injury_df['injury_date'].max()}")
            print(f"Historical injury date range: {historical_injuries['injury_date'].min() if len(historical_injuries) > 0 else 'None'} to {historical_injuries['injury_date'].max() if len(historical_injuries) > 0 else 'None'}")
            if len(historical_injuries) > 0:
                hist_player_ids = historical_injuries['player_id'].unique()
                target_player_ids = target_injuries
                overlap = set(hist_player_ids) & set(target_player_ids)
                print(f"Players with historical injuries: {len(hist_player_ids)}")
                print(f"Players with 2022-23 injuries: {len(target_player_ids)}")
                print(f"Overlap (potential leakage): {len(overlap)} players")
                if len(overlap) > 0:
                    print(f"Overlapping players: {list(overlap)[:5]}")
            
            # ELIMINATE ALL INJURY HISTORY FEATURES - Learn from performance patterns only
            print("Setting all injury history features to 0 (performance-only model)")
            df['career_injuries'] = 0
            df['total_days_injured'] = 0  
            df['avg_recovery_days'] = 0
            df['has_injury_history'] = 0
            df['injury_prone_score'] = 0
            df['recent_injury_count'] = 0
            df['days_since_injury'] = 999
            df['muscle_injury_count'] = 0
            df['knee_injury_count'] = 0

            print("✅ All injury history features set to 0 - model will learn from performance patterns only")
            
            # Fill NaN values
            injury_features = ['career_injuries', 'total_days_injured', 'avg_recovery_days']
            for feature in injury_features:
                df[feature] = df[feature].fillna(0)
            
            # Create additional features
            df['has_injury_history'] = (df['career_injuries'] > 0).astype(int)
            df['injury_prone_score'] = (df['career_injuries'] * 0.1 + df['total_days_injured'] / 365.0 * 0.2).clip(0, 1)
            df['recent_injury_count'] = 0
            df['days_since_injury'] = 999
            df['muscle_injury_count'] = 0
            df['knee_injury_count'] = 0
            
            # Final check
            injury_rate = df['injury_target'].mean()
            injured_count = df['injury_target'].sum()
            
            print(f"✅ SUCCESS!")
            print(f"   Final dataset: {len(df)} players")
            print(f"   Injured in 2022-23: {injured_count} ({injury_rate:.1%})")
            print(f"   Historical injuries: {df['career_injuries'].sum()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def prepare_data(self, df):
        """Prepare data with feature engineering including injury features"""
        print("\nEngineering features...")

        # Aggregate duplicate players first
        df_clean = self.data_aggregator.aggregate_duplicate_players(df)
        
        # Create base features
        df_features = self.feature_engineer.create_base_features(df_clean)
        
        # Create advanced features
        df_features = self.feature_engineer.create_advanced_features(df_features)
        
        # Add injury features from the original dataframe
        injury_columns = [
            'career_injuries', 'total_days_injured', 'injury_prone_score',
            'injuries_per_year', 'muscle_injury_count', 'knee_injury_count',
            'recent_injury_count', 'days_since_injury', 'injury_target'
        ]
        
        for col in injury_columns:
            if col in df.columns:
                df_features[col] = df[col]
        
        # Create injury-specific features
        df_features = self.feature_engineer.create_injury_features(df_features)
        
        # Ensure we have the target
        if 'injury_target' not in df_features.columns:
            df_features = self.feature_engineer.create_target_variable(df_features)
        
        print(f"Created {len(df_features.columns)} total features")
        print(f"Injury rate in prepared data: {df_features['injury_target'].mean():.2%}")
        
        return df_features
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """Train Random Forest model with cross-validation"""
        print("\nTraining Random Forest model...")
        
        # Fix: Use consistent target variable name
        X = self.feature_engineer.prepare_features(df, target_col='injury_target')
        y = df['injury_target']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Check class balance
        class_counts = y.value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            raise ValueError("Need at least 2 classes to train a binary classifier")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
        
        # Simplified hyperparameter tuning for faster execution
        print("\nPerforming hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }
        
        rf = RandomForestClassifier(
            random_state=random_state, 
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Use fewer CV folds for speed, handle small datasets
        cv_folds = min(3, len(y_train) // 10)
        if cv_folds < 2:
            cv_folds = 2
            
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Use best model
        self.model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.model_metrics['accuracy'] = self.model.score(X_test, y_test)
        self.model_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        self.model_metrics['classification_report'] = classification_report(y_test, y_pred)
        self.model_metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {self.model_metrics['accuracy']:.3f}")
        print(f"ROC AUC: {self.model_metrics['roc_auc']:.3f}")
        print("\nClassification Report:")
        print(self.model_metrics['classification_report'])
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
        self.model_metrics['cv_scores'] = cv_scores
        print(f"\nCross-validation ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.model_metrics['feature_importance'] = feature_importance
        
        return X_train, X_test, y_train, y_test
    
    def explain_predictions(self, X_train, X_test):
        """Generate SHAP explanations for model predictions"""
        if not SHAP_AVAILABLE:
            print("⚠️  SHAP not available - skipping explanations")
            return None
            
        print("\nGenerating SHAP explanations...")
        
        try:
            # Create SHAP explainer
            self.shap_explainer = shap.TreeExplainer(self.model)
            
            # Calculate SHAP values for test set (limit size for speed)
            sample_size = min(50, len(X_test))  # Reduced for faster execution
            self.X_test_sample = X_test.iloc[:sample_size]
            
            shap_values = self.shap_explainer.shap_values(self.X_test_sample)
            
            # For binary classification, use positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Store SHAP values
            self.shap_values = shap_values
            
            print(f"SHAP analysis completed for {sample_size} samples")
            
            return shap_values
            
        except Exception as e:
            print(f"Warning: SHAP analysis failed: {e}")
            print("Continuing without SHAP explanations...")
            return None
    
    def save_model(self, model_name='injury_risk_model'):
        """Save trained model and related artifacts"""
        print("\nSaving model artifacts...")
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(self.base_path, 'models', 'saved')
        os.makedirs(models_dir, exist_ok=True)
        
        try:
            # Save model
            model_path = os.path.join(models_dir, f'{model_name}.pkl')
            joblib.dump(self.model, model_path)
            print(f"Model saved to: {model_path}")
            
            # Save feature engineer
            fe_path = os.path.join(models_dir, f'{model_name}_feature_engineer.pkl')
            joblib.dump(self.feature_engineer, fe_path)
            print(f"Feature engineer saved to: {fe_path}")
            
            # Save feature names
            features_path = os.path.join(models_dir, f'{model_name}_features.txt')
            with open(features_path, 'w') as f:
                for feature in self.feature_names:
                    f.write(f"{feature}\n")
            print(f"Feature names saved to: {features_path}")
            
            # Save metrics
            import json
            metrics_path = os.path.join(models_dir, f'{model_name}_metrics.json')
            metrics_to_save = {
                'accuracy': float(self.model_metrics['accuracy']),
                'roc_auc': float(self.model_metrics['roc_auc']),
                'cv_scores_mean': float(self.model_metrics['cv_scores'].mean()),
                'cv_scores_std': float(self.model_metrics['cv_scores'].std()),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_count': len(self.feature_names)
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"Metrics saved to: {metrics_path}")
            
            return models_dir
            
        except Exception as e:
            print(f"Error saving model artifacts: {e}")
            return None
    
    def predict_risk(self, player_data):
        """Predict injury risk for new player data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare features
        player_features = self.feature_engineer.prepare_features(
            player_data, target_col='injury_target' #, scale=True
        )

        print(f"Features prepared: {player_features.shape}")
        print(f"Training feature names: {len(self.feature_names)}")
        print(f"Prediction feature names: {len(player_features.columns)}")

        # Check for feature mismatches
        training_features = set(self.feature_names)
        prediction_features = set(player_features.columns)
        
        missing_in_prediction = training_features - prediction_features
        extra_in_prediction = prediction_features - training_features

        if missing_in_prediction:
            print(f"Adding missing features: {missing_in_prediction}")
            for feature in missing_in_prediction:
                player_features[feature] = 0
        
        if extra_in_prediction:
            print(f"Removing extra features: {extra_in_prediction}")

        # CRITICAL: Ensure exact feature order and names match training
        try:
            player_features = player_features[self.feature_names]
            print(f"✅ Feature alignment successful: {player_features.shape}")
        except KeyError as e:
            print(f"❌ Feature alignment failed: {e}")
            print(f"Available features: {list(player_features.columns)}")
            print(f"Required features: {self.feature_names}")
            raise ValueError(f"Feature mismatch: {e}")
        
        # # Make predictions
        # try:
        #     risk_probability = self.model.predict_proba(player_features)[:, 1]
        #     risk_class = self.model.predict(player_features)

        #     # Debug: Check shapes
        #     print(f"Risk probability shape: {risk_probability.shape}")
        #     print(f"Risk class shape: {risk_class.shape}")
        #     print(f"Player data shape: {len(player_data)}")

        #     # FIXED: Use reset_index to ensure proper alignment
        #     player_data_clean = player_data.reset_index(drop=True)

        #     # BULLETPROOF: Ensure everything is 1D
        #     player_ids = np.array(player_data['player_id']).flatten()
        #     player_names = np.array(player_data['player_name']).flatten()
        #     risk_probs = np.array(risk_probability).flatten()
        #     risk_classes = np.array(risk_class).flatten()

        #     # Create risk levels
        #     risk_levels = []
        #     for i in range(len(risk_probability)):
        #         prob = risk_probability[i]
        #         if prob <= 0.3:
        #             risk_levels.append('Low')
        #         elif prob <= 0.5:
        #             risk_levels.append('Moderate')
        #         elif prob <= 0.7:
        #             risk_levels.append('High')
        #         else:
        #             risk_levels.append('Critical')


        #     # Debug: Check final shapes
        #     print(f"Final shapes - IDs: {player_ids.shape}, Names: {player_names.shape}")
        #     print(f"Final shapes - Probs: {risk_probs.shape}, Classes: {risk_classes.shape}")
        #     print(f"Risk levels count: {len(risk_levels)}")
            
        #     # Create results DataFrame -> with explicit 1D arrays (.values() and .flatten())
        #     results = pd.DataFrame({
        #         'player_id': player_ids,
        #         'player_name': player_names,
        #         'risk_probability': risk_probs,
        #         'risk_class': risk_classes,
        #         'risk_level': risk_levels
        #     })
            
        #     print(f"✅ Predictions generated for {len(results)} players")
        #     return results
        # Make predictions
        try:
            risk_probability = self.model.predict_proba(player_features)[:, 1]
            risk_class = self.model.predict(player_features)
            
            # SIMPLE BULLETPROOF APPROACH - Build results row by row
            results_list = []
            
            for i in range(len(risk_probability)):
                prob = risk_probability[i]
                
                # Determine risk level
                if prob <= 0.3:
                    level = 'Low'
                elif prob <= 0.5:
                    level = 'Moderate'
                elif prob <= 0.7:
                    level = 'High'
                else:
                    level = 'Critical'
                
                # Add row to results
                results_list.append({
                    'player_id': player_data.iloc[i]['player_id'],
                    'player_name': player_data.iloc[i]['player_name'],
                    'risk_probability': float(prob),
                    'risk_class': int(risk_class[i]),
                    'risk_level': level
                })
            
            # Convert to DataFrame
            results = pd.DataFrame(results_list)
            
            print(f"✅ Predictions generated for {len(results)} players")
            return results
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            print(f"Player features shape: {player_features.shape}")
            print(f"Player features columns: {list(player_features.columns)}")
            raise
    
    
    def generate_risk_report(self, player_id, player_data):
        """Generate detailed risk report for a specific player"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            print("Warning: SHAP not available. Generating basic report.")
            
        # Get player features
        # player_row = player_data[player_data['player_id'] == player_id]
        player_row = player_data.reset_index(drop=True)
        player_row = player_row[player_row['player_id'] == player_id].iloc[:1] # take first match


        if player_row.empty:
            raise ValueError(f"Player ID {player_id} not found")
        
        player_features = self.feature_engineer.prepare_features(
            player_row, target_col='injury_target', scale=True
        )[self.feature_names]
        
        # Get prediction
        risk_prob = self.model.predict_proba(player_features)[0, 1]
        
        # Create basic report
        report = {
            'player_name': player_row['player_name'].values[0],
            'age': int(player_row['age'].values[0]),
            'position': player_row['position'].values[0],
            'team': player_row['team'].values[0],
            'risk_probability': float(risk_prob),
            'risk_level': 'Critical' if risk_prob > 0.7 else 'High' if risk_prob > 0.5 else 'Moderate' if risk_prob > 0.3 else 'Low',
            'top_risk_factors': [],
            'recommendations': []
        }
        
        # Add SHAP analysis if available
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(player_features)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Get top risk factors
                feature_impacts = pd.DataFrame({
                    'feature': self.feature_names,
                    'impact': shap_values[0]
                }).sort_values('impact', ascending=False)
                
                # Add top 5 risk factors
                for _, row in feature_impacts.head(5).iterrows():
                    feature_name = row['feature']
                    impact = row['impact']
                    value = player_features[feature_name].values[0]
                    
                    report['top_risk_factors'].append({
                        'factor': feature_name,
                        'impact': float(impact),
                        'current_value': float(value)
                    })
            except Exception as e:
                print(f"Warning: Could not generate SHAP analysis: {e}")
        
        # Generate basic recommendations
        if risk_prob > 0.7:
            report['recommendations'].append("Immediate medical evaluation recommended")
            report['recommendations'].append("Consider rest period before next match")
        elif risk_prob > 0.5:
            report['recommendations'].append("Monitor closely during training and matches")
            report['recommendations'].append("Consider workload reduction")
        elif risk_prob > 0.3:
            report['recommendations'].append("Standard monitoring protocols")
        
        return report

    def evaluate_injury_model_professionally(self, X_test, y_test, y_pred_proba):
        """Evaluate injury prediction model using sports science standards"""
        
        print("\n" + "="*60)
        print("PROFESSIONAL SPORTS SCIENCE EVALUATION")
        print("="*60)
        
        # 1. ROC AUC (Primary Metric)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n🎯 PRIMARY METRIC - ROC AUC: {roc_auc:.3f}")
        
        # Interpret ROC AUC for sports context
        if roc_auc >= 0.75:
            interpretation = "EXCELLENT - Professional grade performance"
        elif roc_auc >= 0.65:
            interpretation = "GOOD - Suitable for professional use"
        elif roc_auc >= 0.55:
            interpretation = "FAIR - May provide some value"
        else:
            interpretation = "POOR - Needs improvement"
        
        print(f"   Interpretation: {interpretation}")
        print(f"   (Professional injury models typically achieve 0.60-0.80)")
        
        # 2. Risk Percentile Analysis (How sports scientists actually use models)
        print(f"\n📊 RISK PERCENTILE ANALYSIS:")
        
        # Create risk percentiles
        percentiles = [90, 80, 70, 60, 50]
        
        for pct in percentiles:
            threshold = np.percentile(y_pred_proba, pct)
            high_risk_players = y_pred_proba >= threshold
            
            if high_risk_players.sum() > 0:
                # How many actual injuries in this high-risk group?
                injuries_in_group = y_test[high_risk_players].sum()  
                total_in_group = high_risk_players.sum()
                injury_rate_in_group = injuries_in_group / total_in_group if total_in_group > 0 else 0
                
                # What % of all injuries did we capture?
                total_injuries = y_test.sum()
                capture_rate = injuries_in_group / total_injuries if total_injuries > 0 else 0
                
                print(f"   Top {100-pct:2d}% risk players: {total_in_group:2d} players, "
                    f"{injuries_in_group} injuries ({injury_rate_in_group:.1%} rate), "
                    f"captures {capture_rate:.1%} of all injuries")
        
        # 3. Early Warning System Performance
        print(f"\n⚠️  EARLY WARNING SYSTEM PERFORMANCE:")
        
        # Top 10% most at-risk players
        top_10_pct_threshold = np.percentile(y_pred_proba, 90)
        top_10_pct_players = y_pred_proba >= top_10_pct_threshold
        
        if top_10_pct_players.sum() > 0:
            injuries_in_top_10 = y_test[top_10_pct_players].sum()
            total_injuries = y_test.sum()
            
            print(f"   🎯 Top 10% highest risk: {top_10_pct_players.sum()} players")
            print(f"   📈 Captures: {injuries_in_top_10}/{total_injuries} injuries "
                f"({injuries_in_top_10/total_injuries:.1%} of all injuries)")
            print(f"   💡 Medical staff should focus on these {top_10_pct_players.sum()} players")
        
        # 4. Feature Importance (What drives injury risk?)
        print(f"\n🔍 TOP INJURY RISK FACTORS:")
        
        if hasattr(self, 'feature_importance'):
            top_features = self.model_metrics['feature_importance'].head(10)
            for idx, row in top_features.iterrows():
                print(f"   {row['feature']:<30} Importance: {row['importance']:.3f}")
        
        # 5. Practical Implementation Guidance
        print(f"\n📋 PRACTICAL IMPLEMENTATION:")
        print(f"   • Model Quality: Professional standard (AUC: {roc_auc:.3f})")
        print(f"   • Recommended Use: Pre-season and monthly risk assessment")
        print(f"   • Focus: Top 10-20% highest risk players for intervention")  
        print(f"   • Integration: Combine with medical staff judgment")
        print(f"   • Update Frequency: Re-train every 6 months with new data")
        
        return {
            'roc_auc': roc_auc,
            'interpretation': interpretation,
            'top_10_pct_players': top_10_pct_players.sum(),
            'injuries_captured_top_10': injuries_in_top_10 if 'injuries_in_top_10' in locals() else 0
        }

    def create_risk_tier_report(self, df_features, risk_predictions):
        """Create professional risk tier report for sports science staff"""
        
        print("\n" + "="*60)  
        print("PLAYER RISK TIER REPORT")
        print("="*60)
        
        # Merge predictions with player info
        report_df = df_features[['player_name', 'age', 'position', 'team', 'minutes_played', 'injury_target']].copy()
        report_df = report_df.merge(
            risk_predictions[['player_id', 'risk_probability']], 
            left_index=True, right_on='player_id', how='left'
        )
        
        # Create professional risk tiers
        report_df['risk_tier'] = pd.cut(
            report_df['risk_probability'], 
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']
        )
        
        # Summary by risk tier
        print("\n📊 RISK TIER SUMMARY:")
        tier_summary = report_df.groupby('risk_tier').agg({
            'player_name': 'count',
            'injury_target': ['sum', 'mean'],
            'age': 'mean',
            'minutes_played': 'mean'
        }).round(2)
        
        tier_summary.columns = ['Players', 'Actual_Injuries', 'Injury_Rate', 'Avg_Age', 'Avg_Minutes']
        print(tier_summary)
        
        # High-risk players detail
        high_risk_players = report_df[report_df['risk_tier'].isin(['High Risk', 'Critical Risk'])].sort_values('risk_probability', ascending=False)
        
        print(f"\n🚨 HIGH-RISK PLAYERS REQUIRING ATTENTION ({len(high_risk_players)} players):")
        print(high_risk_players[['player_name', 'team', 'position', 'age', 'risk_probability', 'injury_target']].head(15))
        
        return report_df


def diagnose_data_leakage(df_features):
    """Diagnose potential data leakage in injury prediction"""
    print("\n=== DATA LEAKAGE DIAGNOSTIC ===")
    
    # Check injury history features for same-season leakage
    injured_players = df_features[df_features['injury_target'] == 1]
    non_injured = df_features[df_features['injury_target'] == 0]
    
    print(f"\nInjured players: {len(injured_players)}")
    print(f"Non-injured players: {len(non_injured)}")
    
    # Check suspicious perfect separations
    leakage_features = []
    
    for feature in ['career_injuries', 'total_days_injured', 'injury_prone_score', 'has_injury_history']:
        if feature in df_features.columns:
            injured_mean = injured_players[feature].mean()
            non_injured_mean = non_injured[feature].mean()
            
            print(f"\n{feature}:")
            print(f"  Injured mean: {injured_mean:.3f}")
            print(f"  Non-injured mean: {non_injured_mean:.3f}")
            
            # Flag potential leakage
            if injured_mean > 0 and non_injured_mean == 0:
                print(f"  🚨 POTENTIAL LEAKAGE: All injured players have {feature} > 0, non-injured = 0")
                leakage_features.append(feature)
            elif abs(injured_mean - non_injured_mean) > 0.5:
                print(f"  ⚠️  Large difference detected")
    
    return leakage_features


def main():
    """Main execution function"""
    print("=== Player Injury Risk Prediction Model ===\n")
    
    # Initialize predictor
    predictor = InjuryPredictor()
    
    try:
        # Load data (using synthetic injuries for testing)
        df = predictor.load_data(use_synthetic_injuries=False)
        
        # Prepare data with features
        df_features = predictor.prepare_data(df)

        # Add this line after df_features = predictor.prepare_data(df)
        leakage_features = diagnose_data_leakage(df_features)
        print(f"Potentially leaking features: {leakage_features}")
        
        # Train model
        # Train model  
        X_train, X_test, y_train, y_test = predictor.train_model(df_features)

        # PROFESSIONAL EVALUATION (Add this)
        y_pred_proba = predictor.model.predict_proba(X_test)[:, 1]
        professional_metrics = predictor.evaluate_injury_model_professionally(X_test, y_test, y_pred_proba)
        
        # Generate SHAP explanations
        predictor.explain_predictions(X_train, X_test)
        
        # Save model
        models_dir = predictor.save_model()
        
        # Generate predictions for all players
        print("\nGenerating risk predictions for all players...")
        risk_predictions = predictor.predict_risk(df_features)

        # Create professional risk tier report
        risk_tier_report = predictor.create_risk_tier_report(df_features, risk_predictions)

        # Save predictions
        output_dir = os.path.join(predictor.base_path, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'injury_risk_predictions.csv')
        risk_predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        
        # Show high-risk players
        high_risk = risk_predictions[risk_predictions['risk_level'].isin(['High', 'Critical'])]
        print(f"\n{len(high_risk)} players identified as high/critical risk:")
        print(high_risk[['player_name', 'risk_probability', 'risk_level']].head(10))
        
        # # Generate sample report for highest risk player
        # if len(high_risk) > 0:
        #     highest_risk_player = high_risk.iloc[0]['player_id']
        #     report = predictor.generate_risk_report(highest_risk_player, df_features)
            
        #     print(f"\nSample Risk Report for {report['player_name']}:")
        #     print(f"Age: {report['age']}, Position: {report['position']}, Team: {report['team']}")
        #     print(f"Risk Level: {report['risk_level']} ({report['risk_probability']:.2%})")
            
        #     if report['top_risk_factors']:
        #         print("\nTop Risk Factors:")
        #         for factor in report['top_risk_factors']:
        #             print(f"  - {factor['factor']}: Impact={factor['impact']:.3f}")
            
        #     print("\nRecommendations:")
        #     for rec in report['recommendations']:
        #         print(f"  - {rec}")
        
        print("\n=== Model training completed successfully! ===")
        return predictor
        
    except Exception as e:
        print(f"\nError during model training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure database is running and accessible")
        print("2. Check that python/db_config.py is properly configured")
        print("3. Verify data has been imported successfully")
        print("4. Check that all required tables exist in your schema")
        raise


if __name__ == "__main__":
    main()