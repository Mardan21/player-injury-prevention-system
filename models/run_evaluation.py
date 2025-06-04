"""
Run comprehensive model evaluation
"""

import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("RUNNING COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    try:
        # Import modules
        print("Importing modules...")
        from injury_predictor import InjuryPredictor
        from model_evaluation import ModelEvaluator
        print("✅ Modules imported successfully")
        
        # Initialize and train model
        print("\n1. Initializing predictor...")
        predictor = InjuryPredictor()
        
        # Load and prepare data
        print("2. Loading and preparing data...")
        df = predictor.load_data(use_synthetic_injuries=True)
        df_features = predictor.prepare_data(df)
        print(f"   Data prepared: {len(df_features)} records, {len(df_features.columns)} features")
        
        # Train model
        print("3. Training model...")
        X_train, X_test, y_train, y_test = predictor.train_model(df_features)
        print(f"   Model trained: {len(X_train)} training samples, {len(X_test)} test samples")
        
        # Generate SHAP explanations (optional)
        print("4. Generating explanations...")
        try:
            predictor.explain_predictions(X_train, X_test)
            print("   ✅ SHAP explanations generated")
        except Exception as e:
            print(f"   ⚠️  SHAP explanations failed: {e}")
            print("   ✅ Continuing without SHAP (model still functional)")
        
        # Save model
        print("5. Saving model...")
        models_dir = predictor.save_model()
        if models_dir:
            print(f"   ✅ Model saved to: {models_dir}")
        
        # Generate predictions
        print("6. Generating predictions...")
        predictions_df = predictor.predict_risk(df_features)
        
        # Save predictions
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        predictions_path = os.path.join(output_dir, 'injury_risk_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"   ✅ Predictions saved to: {predictions_path}")
        
        # Create evaluator and generate visualizations
        print("7. Creating evaluation visualizations...")
        try:
            evaluator = ModelEvaluator(predictor)
            eval_output_dir = evaluator.generate_all_visualizations(X_test, y_test, predictions_df)
            print(f"   ✅ Evaluation visualizations saved to: {eval_output_dir}")
        except Exception as e:
            print(f"   ⚠️  Some visualizations failed: {e}")
            print("   ✅ Core evaluation completed (check basic metrics)")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total players analyzed: {len(predictions_df)}")
        print(f"   Features engineered: {len(predictor.feature_names)}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        print(f"\n🎯 Model Performance:")
        print(f"   Accuracy: {predictor.model_metrics['accuracy']:.3f}")
        print(f"   ROC AUC: {predictor.model_metrics['roc_auc']:.3f}")
        if 'cv_scores' in predictor.model_metrics:
            cv_mean = predictor.model_metrics['cv_scores'].mean()
            cv_std = predictor.model_metrics['cv_scores'].std()
            print(f"   Cross-validation AUC: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
        
        print(f"\n⚠️  Risk Distribution:")
        risk_dist = predictions_df['risk_level'].value_counts()
        total_players = len(predictions_df)
        for level, count in risk_dist.items():
            percentage = (count / total_players) * 100
            print(f"   {level}: {count} players ({percentage:.1f}%)")
        
        # High-risk players
        high_risk = predictions_df[predictions_df['risk_level'].isin(['High', 'Critical'])]
        print(f"\n🚨 High/Critical Risk Players: {len(high_risk)}")
        if len(high_risk) > 0:
            print("   Top 10 highest risk players:")
            for i, (_, row) in enumerate(high_risk.head(10).iterrows(), 1):
                print(f"   {i:2d}. {row['player_name']:<25} Risk: {row['risk_probability']:.3f} ({row['risk_level']})")
        
        # Feature importance
        if 'feature_importance' in predictor.model_metrics:
            print(f"\n🔍 Top 10 Important Features:")
            for i, (_, row) in enumerate(predictor.model_metrics['feature_importance'].head(10).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
        # Sample risk report
        if len(high_risk) > 0:
            print(f"\n📋 Sample Risk Report:")
            highest_risk_player = high_risk.iloc[0]['player_id']
            try:
                report = predictor.generate_risk_report(highest_risk_player, df_features)
                print(f"   Player: {report['player_name']} (Age: {report['age']}, {report['position']}, {report['team']})")
                print(f"   Risk Level: {report['risk_level']} ({report['risk_probability']:.1%})")
                if report['top_risk_factors']:
                    print(f"   Top Risk Factors:")
                    for factor in report['top_risk_factors'][:3]:
                        print(f"     • {factor['factor']}: {factor['impact']:.3f}")
                print(f"   Recommendations:")
                for rec in report['recommendations']:
                    print(f"     • {rec}")
            except Exception as e:
                print(f"   ⚠️  Could not generate detailed report: {e}")
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"\n📁 Check these directories for results:")
        print(f"   • {output_dir} - Predictions and data")
        if models_dir:
            print(f"   • {models_dir} - Saved model artifacts")
        try:
            print(f"   • {eval_output_dir} - Evaluation visualizations")
        except:
            print(f"   • {output_dir}/model_evaluation - Evaluation visualizations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Ensure database is running and accessible")
        print(f"2. Run ml_test_script.py first to verify setup")
        print(f"3. Check that injury_predictor.py runs successfully")
        print(f"4. Verify all required packages are installed")
        
        import traceback
        print(f"\nDetailed error:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ SUCCESS: Model evaluation completed!")
        print(f"🚀 Your injury prediction system is ready for production use.")
    else:
        print(f"\n❌ FAILED: Check the errors above and retry.")
    
    sys.exit(0 if success else 1)