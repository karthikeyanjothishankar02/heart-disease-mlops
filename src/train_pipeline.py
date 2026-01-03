"""
Complete training pipeline for Heart Disease Prediction
Includes data loading, preprocessing, training, evaluation, and model saving
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, f1_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, evaluation, and saving"""
    
    def __init__(self, experiment_name="heart_disease_prediction"):
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filepath="data/processed/heart_disease_clean.csv"):
        """Load processed data"""
        logger.info(f"Loading data from {filepath}")
        
        if not Path(filepath).exists():
            logger.warning("Processed data not found. Running preprocessing...")
            self._preprocess_data()
        
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded: {df.shape}")
        
        return df
    
    def _preprocess_data(self):
        """Preprocess raw data if needed"""
        # This would call your EDA and cleaning functions
        from eda import load_data, clean_data
        
        raw_data = load_data("data/raw/heart_disease.csv")
        clean_data(raw_data)
    
    def prepare_features(self, df, test_size=0.2, random_state=42):
        """Prepare features and split data"""
        logger.info("Preparing features...")
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Transform features
        X, y = self.feature_engineer.fit_transform(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")
        
        with mlflow.start_run(run_name="Logistic_Regression") as run:
            # Define hyperparameters
            params = {
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42
            }
            
            # Train model
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test)
            
            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Save locally
            self.models['logistic_regression'] = model
            self.results['logistic_regression'] = metrics
            
            logger.info(f"Logistic Regression - Test Accuracy: {metrics['test_accuracy']:.4f}, ROC-AUC: {metrics['test_roc_auc']:.4f}")
            
            return model, metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        with mlflow.start_run(run_name="Random_Forest") as run:
            # Define hyperparameters
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            }
            
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test)
            
            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance
            self._log_feature_importance(model, X_train.columns)
            
            # Save locally
            self.models['random_forest'] = model
            self.results['random_forest'] = metrics
            
            logger.info(f"Random Forest - Test Accuracy: {metrics['test_accuracy']:.4f}, ROC-AUC: {metrics['test_roc_auc']:.4f}")
            
            return model, metrics
    
    def _evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Evaluate model performance"""
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Calculate overfitting metric
        metrics['overfit_gap'] = metrics['train_accuracy'] - metrics['test_accuracy']
        
        return metrics
    
    def _log_feature_importance(self, model, feature_names):
        """Log feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig('reports/feature_importance.png')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact('reports/feature_importance.png')
    
    def select_best_model(self):
        """Select best model based on test ROC-AUC"""
        logger.info("Selecting best model...")
        
        best_model_name = max(
            self.results.keys(), 
            key=lambda k: self.results[k]['test_roc_auc']
        )
        
        self.best_model = self.models[best_model_name]
        self.best_score = self.results[best_model_name]['test_roc_auc']
        
        logger.info(f"Best model: {best_model_name} with ROC-AUC: {self.best_score:.4f}")
        
        return best_model_name, self.best_model
    
    def save_production_model(self, output_dir="models"):
        """Save best model and preprocessor for production"""
        logger.info("Saving production model...")
        
        # Save model
        model_path = f"{output_dir}/production_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save preprocessor
        preprocessor_path = f"{output_dir}/production_preprocessor.pkl"
        self.feature_engineer.save(preprocessor_path)
        
        # Save with MLflow
        mlflow.sklearn.save_model(self.best_model, f"{output_dir}/production_mlflow")
        
        logger.info(f"Production model saved to {output_dir}")
        
        return model_path, preprocessor_path
    
    def generate_report(self):
        """Generate training report"""
        logger.info("Generating training report...")
        
        report = []
        report.append("# Heart Disease Prediction - Training Report\n")
        report.append(f"Experiment: {self.experiment_name}\n\n")
        
        report.append("## Model Results\n\n")
        
        for model_name, metrics in self.results.items():
            report.append(f"### {model_name}\n")
            report.append(f"- Test Accuracy: {metrics['test_accuracy']:.4f}\n")
            report.append(f"- Test Precision: {metrics['test_precision']:.4f}\n")
            report.append(f"- Test Recall: {metrics['test_recall']:.4f}\n")
            report.append(f"- Test F1-Score: {metrics['test_f1']:.4f}\n")
            report.append(f"- Test ROC-AUC: {metrics['test_roc_auc']:.4f}\n")
            report.append(f"- CV Mean Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})\n")
            report.append(f"- Overfit Gap: {metrics['overfit_gap']:.4f}\n\n")
        
        # Save report
        with open("reports/training_report.md", 'w') as f:
            f.writelines(report)
        
        logger.info("Report saved to reports/training_report.md")


def main():
    """Main training pipeline"""
    logger.info("Starting training pipeline...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    df = trainer.load_data()
    
    # Prepare features
    X_train, X_test, y_train, y_test = trainer.prepare_features(df)
    
    # Train models
    trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_model_name, best_model = trainer.select_best_model()
    
    # Save production model
    trainer.save_production_model()
    
    # Generate report
    trainer.generate_report()
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Production model saved to models/")
    
    return trainer


if __name__ == "__main__":
    main()