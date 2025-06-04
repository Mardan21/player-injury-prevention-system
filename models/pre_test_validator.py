"""
Pre-test validation script for ML files - COMPLETE FIXED VERSION
Run this before testing your ML pipeline to catch potential issues
"""

import sys
import os
import pandas as pd
import numpy as np

def check_imports():
    """Check that all required packages can be imported"""
    print("Checking imports...")
    issues = []
    
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        issues.append(f"❌ scikit-learn not available: {e}")
    
    try:
        import matplotlib
        print(f"✅ matplotlib: {matplotlib.__version__}")
        
        # Test seaborn style
        import matplotlib.pyplot as plt
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            print("✅ seaborn-v0_8-darkgrid style available")
            plt.style.use('default')  # Reset
        except:
            try:
                plt.style.use('seaborn-darkgrid')
                print("⚠️  Using fallback seaborn-darkgrid style")
                plt.style.use('default')  # Reset
            except:
                print("⚠️  Seaborn styles not available, will use default")
                
    except ImportError as e:
        issues.append(f"❌ matplotlib not available: {e}")
    
    try:
        import seaborn
        print(f"✅ seaborn: {seaborn.__version__}")
    except ImportError as e:
        issues.append(f"❌ seaborn not available: {e}")
    
    try:
        import shap
        print(f"✅ SHAP: {shap.__version__}")
        
        # Test SHAP API compatibility
        try:
            from shap import Explainer
            print("✅ SHAP Explainer API available")
        except Exception as e:
            print(f"⚠️  SHAP issue (non-critical): {e}")
            
    except ImportError as e:
        print(f"⚠️  SHAP not available (will skip SHAP visualizations): {e}")
    
    try:
        import joblib
        print(f"✅ joblib: {joblib.__version__}")
    except ImportError as e:
        issues.append(f"❌ joblib not available: {e}")
    
    return issues

def check_database():
    """Check database connection and required tables"""
    print("\nChecking database...")
    issues = []
    
    try:
        # Add python directory to path for db_config import
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        python_dir = os.path.join(project_root, 'python')
        sys.path.insert(0, python_dir)
        
        import db_config
        print("✅ db_config.py found in python/ directory")
        
        if hasattr(db_config, 'test_connection') and db_config.test_connection():
            print("✅ Database connection successful")
            
            # Check for required tables
            try:
                from sqlalchemy import text
                with db_config.engine.connect() as conn:
                    # Check players table
                    try:
                        result = conn.execute(text("SELECT COUNT(*) FROM players"))
                        count = result.scalar()
                        print(f"✅ players table: {count} records")
                    except Exception as e:
                        issues.append(f"❌ players table issue: {e}")
                    
                    # Check player_performance table  
                    try:
                        result = conn.execute(text("SELECT COUNT(*) FROM player_performance"))
                        count = result.scalar()
                        print(f"✅ player_performance table: {count} records")
                    except Exception as e:
                        issues.append(f"❌ player_performance table issue: {e}")
                    
                    # Check for key columns (simplified check)
                    try:
                        result = conn.execute(text("SELECT * FROM player_performance LIMIT 1"))
                        row = result.fetchone()
                        if row:
                            columns = list(row._mapping.keys())
                            
                            required_cols = ['minutes_played', 'games_played', 'tackles', 'interceptions']
                            missing_cols = [col for col in required_cols if col not in columns]
                            
                            if missing_cols:
                                issues.append(f"❌ Missing required columns: {missing_cols}")
                            else:
                                print("✅ Required columns present")
                                
                            # Check for advanced columns that might be missing
                            advanced_cols = ['number_of_dribblers_tackled', 'shots_blocked', 'passes_blocked']
                            missing_advanced = [col for col in advanced_cols if col not in columns]
                            if missing_advanced:
                                print(f"⚠️  Missing advanced columns (will use defaults): {missing_advanced}")
                        else:
                            print("⚠️  No data in player_performance table")
                        
                    except Exception as e:
                        print(f"⚠️  Column check failed (non-critical): {e}")
                        
            except Exception as e:
                issues.append(f"❌ Database query test failed: {e}")
        else:
            issues.append("❌ Database connection failed")
            
    except ImportError as e:
        issues.append(f"❌ db_config.py not found in python/ directory: {e}")
    except Exception as e:
        issues.append(f"❌ Database check failed: {e}")
    
    return issues

def check_file_structure():
    """Check that files are in the expected structure"""
    print("\nChecking file structure...")
    issues = []
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    required_files = [
        'feature_engineering.py',
        'injury_predictor.py', 
        'model_evaluation.py',
        'run_evaluation.py'
    ]
    
    for file in required_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} found")
        else:
            issues.append(f"❌ {file} not found")
    
    # Check if db_config is accessible
    try:
        python_dir = os.path.join(project_root, 'python')
        sys.path.insert(0, python_dir)
        import db_config
        print("✅ db_config.py accessible")
    except ImportError:
        issues.append("❌ db_config.py not accessible")
    
    # Check outputs directory
    outputs_dir = os.path.join(project_root, 'outputs')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)
        print("✅ outputs directory created")
    else:
        print("✅ outputs directory exists")
    
    return issues

def test_feature_engineering():
    """Test feature engineering with complete test data"""
    print("\nTesting feature engineering...")
    issues = []
    
    try:
        from feature_engineering import InjuryFeatureEngineer
        
        # Create COMPLETE test data with ALL expected columns including fouls_drawn!
        test_data = pd.DataFrame({
            'player_id': [1, 2, 3],
            'player_name': ['Test A', 'Test B', 'Test C'],
            'age': [25, 30, 28],
            'position': ['FW', 'MF', 'DF'],
            'team': ['Team A', 'Team B', 'Team C'],
            
            # Basic game metrics
            'minutes_played': [2000, 2500, 1800],
            'games_played': [25, 30, 22],
            'games_started': [20, 25, 18],
            
            # Goal/Assist metrics
            'goals': [10, 5, 1],
            'assists': [5, 8, 2],
            'shots': [35, 25, 10],
            'shots_on_target': [20, 15, 5],
            'goals_per_shot_on_target': [0.5, 0.3, 0.2],
            
            # Defensive metrics
            'tackles': [20, 45, 80],
            'tackles_won': [15, 30, 60],
            'interceptions': [10, 25, 40],
            'clearances': [5, 15, 60],
            'blocks': [5, 10, 20],
            'shots_blocked': [2, 5, 15],
            'passes_blocked': [3, 8, 12],
            'number_of_dribblers_tackled': [8, 12, 20],
            'number_of_times_dribbled_past_by_opponent': [3, 8, 15],
            'number_of_loose_balls_recovered': [10, 15, 25],
            
            # Aerial duels
            'aerial_duels_won': [15, 20, 35],
            'aerial_duels_lost': [10, 15, 20],
            
            # Discipline - INCLUDING THE MISSING fouls_drawn!
            'fouls_committed': [25, 35, 45],
            'fouls_drawn': [20, 30, 25],  # ← This was missing!
            'yellow_cards': [3, 4, 6],
            'red_cards': [0, 1, 0],
            'second_yellow_card': [0, 0, 1],
            
            # Passing metrics
            'passes_attempted': [400, 800, 300],
            'pass_completion_percentage': [85, 90, 88],
            'progressive_passes_count': [20, 60, 15],
            
            # Carrying metrics
            'carries': [100, 150, 80],
            'progressive_carries': [30, 50, 25],
            'progressive_carry_distance': [300, 500, 200],
            'dispossessed_carries': [8, 12, 6],
            'missed_carries': [4, 6, 3],
            'carries_into_18': [5, 8, 2],
            'total_carry_distance': [800, 1200, 600],
            
            # Creative metrics
            'assisted_shots': [3, 6, 1],
            'completed_crosses_into_18': [2, 4, 1],
            'shot_creating_actions': [15, 20, 8],
            'goal_creating_actions': [3, 5, 1],
            'successful_dribbles_leading_to_shot': [5, 8, 3],
            'successful_dribbles_leading_to_goal': [2, 3, 1],
            'shots_leading_to_another_shot': [3, 5, 2],
            'shots_leading_to_goal_scoring_shot': [2, 3, 1],
            
            # Take-on metrics
            'number_attempts_take_on_defender': [15, 20, 8],
            'number_defenders_taken_on_successfully': [10, 15, 5],
            
            # Error tracking
            'errors_leading_to_opponent_shot': [1, 2, 3],
            
            # Additional metrics that might be referenced
            'crosses': [10, 15, 5],
            'penalty_kicks_attempted': [1, 0, 0],
            'penalty_kicks_won': [1, 0, 0],
            'own_goals': [0, 0, 0],
            'touches': [120, 180, 100]
        })
        
        fe = InjuryFeatureEngineer()
        
        # Test each stage
        print("   Testing base features...")
        base_features = fe.create_base_features(test_data)
        print(f"   ✅ Base features: {len(base_features.columns)} columns")
        
        print("   Testing advanced features...")
        advanced_features = fe.create_advanced_features(base_features)
        print(f"   ✅ Advanced features: {len(advanced_features.columns)} columns")
        
        print("   Testing injury features...")
        injury_features = fe.create_injury_features(advanced_features)
        print(f"   ✅ Injury features: {len(injury_features.columns)} columns")
        
        print("   Testing target creation...")
        target_features = fe.create_target_variable(injury_features)
        print(f"   ✅ Target created: {'injury_target' in target_features.columns}")
        
        print("   Testing feature preparation...")
        X = fe.prepare_features(target_features, target_col='injury_target')
        print(f"   ✅ Features prepared: {X.shape}")
        
        print("✅ Feature engineering test completed successfully")
        
    except Exception as e:
        issues.append(f"❌ Feature engineering test failed: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
    
    return issues

def main():
    """Run all validation checks"""
    print("=" * 60)
    print("ML FILES PRE-TEST VALIDATION")
    print("=" * 60)
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_imports())
    all_issues.extend(check_database())
    all_issues.extend(check_file_structure())
    all_issues.extend(test_feature_engineering())
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if not all_issues:
        print("🎉 ALL CHECKS PASSED!")
        print("\nYour ML files are ready for testing.")
        print("\nNext steps:")
        print("1. Run: python models/injury_predictor.py")
        print("2. Then: python models/run_evaluation.py")
        return True
    else:
        # Separate critical from non-critical issues
        critical_issues = [issue for issue in all_issues if issue.startswith("❌")]
        warning_issues = [issue for issue in all_issues if issue.startswith("⚠️")]
        
        if critical_issues:
            print(f"❌ Found {len(critical_issues)} critical issues:")
            for issue in critical_issues:
                print(f"   {issue}")
        
        if warning_issues:
            print(f"\n⚠️  Found {len(warning_issues)} warnings (non-critical):")
            for issue in warning_issues:
                print(f"   {issue}")
        
        if not critical_issues:
            print("\n✅ No critical issues found - you can proceed with testing!")
            print("⚠️  Warnings may cause some features to not work but won't prevent testing")
            return True
        else:
            print("\n🔧 Fix critical issues before proceeding:")
            print("1. Install missing packages: pip install [package-name]")
            print("2. Ensure db_config.py is in project root directory")
            print("3. Check database connection and schema")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)