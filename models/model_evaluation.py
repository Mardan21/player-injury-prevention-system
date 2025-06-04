"""
Model evaluation and visualization module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import shap
import os
from datetime import datetime

class ModelEvaluator:
    def __init__(self, predictor, output_dir='outputs'):
        self.predictor = predictor
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_path, output_dir, 'model_evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance from Random Forest"""
        plt.figure(figsize=(10, 8))
        
        feature_imp = self.predictor.model_metrics['feature_importance'].head(top_n)
        
        # Create horizontal bar plot
        plt.barh(range(len(feature_imp)), feature_imp['importance'])
        plt.yticks(range(len(feature_imp)), feature_imp['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features for Injury Risk Prediction')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to {self.output_dir}")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        
        cm = self.predictor.model_metrics['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Low Risk', 'High Risk'],
                    yticklabels=['Low Risk', 'High Risk'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix - Injury Risk Prediction')
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved to {self.output_dir}")
    
    def plot_roc_curve(self, X_test, y_test):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        
        # Get predictions
        y_pred_proba = self.predictor.model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve plot saved to {self.output_dir}")
    
    def plot_precision_recall_curve(self, X_test, y_test):
        """Plot precision-recall curve"""
        plt.figure(figsize=(8, 6))
        
        # Get predictions
        y_pred_proba = self.predictor.model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        # Plot
        plt.plot(recall, precision, color='darkblue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Precision-recall curve saved to {self.output_dir}")
    
    def plot_shap_summary(self):
        """Create SHAP summary plot"""
        if self.predictor.shap_values is None:
            print("SHAP values not available. Run explain_predictions() first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create summary plot
        shap.summary_plot(
            self.predictor.shap_values, 
            self.predictor.X_test_sample,
            plot_type="bar",
            show=False
        )
        
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP summary plot saved to {self.output_dir}")
    
    def plot_shap_waterfall(self, player_index=0):
        """Create SHAP waterfall plot for a specific player"""
        if self.predictor.shap_values is None:
            print("SHAP values not available. Run explain_predictions() first.")
            return
        
        # Create explanation object
        explainer = shap.Explainer(self.predictor.model, self.predictor.X_test_sample)
        shap_values = explainer(self.predictor.X_test_sample)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[player_index], show=False)
        plt.title(f'SHAP Explanation for Player {player_index}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'shap_waterfall_player_{player_index}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP waterfall plot saved to {self.output_dir}")
    
    def plot_risk_distribution(self, predictions_df):
        """Plot distribution of risk scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of risk probabilities
        ax1.hist(predictions_df['risk_probability'], bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(0.5, color='red', linestyle='--', label='High Risk Threshold')
        ax1.axvline(0.7, color='darkred', linestyle='--', label='Critical Risk Threshold')
        ax1.set_xlabel('Risk Probability')
        ax1.set_ylabel('Number of Players')
        ax1.set_title('Distribution of Injury Risk Probabilities')
        ax1.legend()
        
        # Pie chart of risk levels
        risk_counts = predictions_df['risk_level'].value_counts()
        colors = ['green', 'yellow', 'orange', 'red']
        ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=colors[:len(risk_counts)])
        ax2.set_title('Distribution of Risk Levels')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'risk_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Risk distribution plot saved to {self.output_dir}")
    
    def create_evaluation_report(self, X_test, y_test, predictions_df):
        """Create comprehensive evaluation report"""
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("INJURY RISK PREDICTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Performance
            f.write("MODEL PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {self.predictor.model_metrics['accuracy']:.3f}\n")
            f.write(f"ROC AUC: {self.predictor.model_metrics['roc_auc']:.3f}\n")
            f.write(f"Cross-Validation ROC AUC: {self.predictor.model_metrics['cv_scores'].mean():.3f} "
                   f"(+/- {self.predictor.model_metrics['cv_scores'].std() * 2:.3f})\n\n")
            
            # Classification Report
            f.write("CLASSIFICATION REPORT:\n")
            f.write("-" * 30 + "\n")
            f.write(self.predictor.model_metrics['classification_report'])
            f.write("\n\n")
            
            # Feature Importance
            f.write("TOP 15 IMPORTANT FEATURES:\n")
            f.write("-" * 30 + "\n")
            for idx, row in self.predictor.model_metrics['feature_importance'].head(15).iterrows():
                f.write(f"{row['feature']:<40} {row['importance']:.4f}\n")
            f.write("\n")
            
            # Risk Distribution
            f.write("RISK LEVEL DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            risk_dist = predictions_df['risk_level'].value_counts()
            total_players = len(predictions_df)
            for level, count in risk_dist.items():
                percentage = (count / total_players) * 100
                f.write(f"{level:<15} {count:>5} ({percentage:>5.1f}%)\n")
            f.write("\n")
            
            # High Risk Players Summary
            high_risk = predictions_df[predictions_df['risk_level'].isin(['High', 'Critical'])]
            f.write(f"HIGH/CRITICAL RISK PLAYERS: {len(high_risk)}\n")
            f.write("-" * 30 + "\n")
            f.write("Top 10 Highest Risk Players:\n")
            for idx, row in high_risk.head(10).iterrows():
                f.write(f"{row['player_name']:<30} Risk: {row['risk_probability']:.3f} ({row['risk_level']})\n")
            
        print(f"Evaluation report saved to {report_path}")
        
        return report_path
    
    def generate_all_visualizations(self, X_test, y_test, predictions_df):
        """Generate all evaluation visualizations"""
        print("\nGenerating model evaluation visualizations...")
        
        # Create all plots
        self.plot_feature_importance()
        self.plot_confusion_matrix()
        self.plot_roc_curve(X_test, y_test)
        self.plot_precision_recall_curve(X_test, y_test)
        self.plot_shap_summary()
        self.plot_shap_waterfall()
        self.plot_risk_distribution(predictions_df)
        
        # Create evaluation report
        self.create_evaluation_report(X_test, y_test, predictions_df)
        
        print(f"\nAll visualizations saved to: {self.output_dir}")
        
        return self.output_dir


if __name__ == "__main__":
    # This module is meant to be imported and used with a trained predictor
    print("Model evaluation module. Import and use with a trained InjuryPredictor instance.")