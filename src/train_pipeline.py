"""
Complete training pipeline for Heart Disease Prediction
Includes data loading, preprocessing, training, evaluation, and model saving
"""

import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

# Add src to path for internal imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from feature_engineering import FeatureEngineer  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, evaluation, and saving."""

    def __init__(self, experiment_name="heart_disease_prediction"):
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0

        # --------------------------
        # MLflow setup (CI/CD safe)
        # --------------------------
        mlruns_dir = os.path.join(os.getcwd(), "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlruns_dir}")
        mlflow.set_experiment(experiment_name)

        # --------------------------
        # Folders for outputs
        # --------------------------
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)

    def load_data(self, filepath="data/processed/heart_disease_clean.csv"):
        """Load processed data."""
        logger.info("Loading data from %s", filepath)

        if not Path(filepath).exists():
            logger.warning("Processed data not found. Running preprocessing...")
            self._preprocess_data()

        df = pd.read_csv(filepath)
        logger.info("Data loaded: %s", df.shape)
        return df

    def _preprocess_data(self):
        """Preprocess raw data if needed."""
        from eda import clean_data, load_data

        raw_data = load_data("data/raw/heart_disease.csv")
        clean_data(raw_data)

    def prepare_features(self, df, test_size=0.2, random_state=42):
        """Prepare features and split data safely, even for tiny datasets."""
        logger.info("Preparing features...")

        self.feature_engineer = FeatureEngineer()
        X, y = self.feature_engineer.fit_transform(df)

        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Convert fractional test_size to absolute number
        if isinstance(test_size, float):
            test_size_abs = max(int(n_samples * test_size), n_classes)
        else:
            test_size_abs = max(test_size, n_classes)

        # If dataset too small, skip stratified split
        if n_samples <= n_classes:
            logger.warning(
                """Too few samples (%d) for %d classes.
                Using full dataset as train/test.""",
                n_samples,
                n_classes,
            )
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            logger.info("Splitting dataset with test_size=%d", test_size_abs)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size_abs,
                random_state=random_state,
                stratify=y if test_size_abs >= n_classes else None,
            )

        logger.info("Training set: %s, Test set: %s", X_train.shape, X_test.shape)
        logger.info(
            "Class distribution - Train: %s, Test: %s",
            np.bincount(y_train),
            np.bincount(y_test),
        )

        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")

        with mlflow.start_run(run_name="Logistic_Regression"):
            params = {
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
            }

            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            self.models["logistic_regression"] = model
            self.results["logistic_regression"] = metrics

            logger.info(
                "Logistic Regression - Test Accuracy: %.4f, ROC-AUC: %.4f",
                metrics["test_accuracy"],
                metrics["test_roc_auc"],
            )

            return model, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model."""
        logger.info("Training Random Forest...")

        with mlflow.start_run(run_name="Random_Forest"):
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            }

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            self._log_feature_importance(model, X_train.columns)

            self.models["random_forest"] = model
            self.results["random_forest"] = metrics

            logger.info(
                "Random Forest - Test Accuracy: %.4f, ROC-AUC: %.4f",
                metrics["test_accuracy"],
                metrics["test_roc_auc"],
            )

            return model, metrics

    def _evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Evaluate model performance."""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_roc_auc": roc_auc_score(y_test, y_test_proba),
        }

        if len(X_train) >= 5:  # Only do CV if enough samples
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="accuracy"
            )
            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()
        else:
            metrics["cv_mean"] = np.nan
            metrics["cv_std"] = np.nan

        metrics["overfit_gap"] = metrics["train_accuracy"] - metrics["test_accuracy"]

        return metrics

    def _log_feature_importance(self, model, feature_names):
        """Log feature importance for tree-based models."""
        if not hasattr(model, "feature_importances_"):
            return

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["feature"][:10], importance_df["importance"][:10])
        plt.xlabel("Importance")
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("reports/feature_importance.png")
        plt.close()

        mlflow.log_artifact("reports/feature_importance.png")

    def select_best_model(self):
        """Select best model based on test ROC-AUC."""
        logger.info("Selecting best model...")

        best_model_name = max(
            self.results, key=lambda k: self.results[k]["test_roc_auc"]
        )
        self.best_model = self.models[best_model_name]
        self.best_score = self.results[best_model_name]["test_roc_auc"]

        logger.info(
            "Best model: %s with ROC-AUC: %.4f", best_model_name, self.best_score
        )

        return best_model_name, self.best_model

    def save_production_model(self, output_dir="models"):
        """Save best model and preprocessor for production."""
        logger.info("Saving production model...")

        model_path = f"{output_dir}/production_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.best_model, f)

        preprocessor_path = f"{output_dir}/production_preprocessor.pkl"
        self.feature_engineer.save(preprocessor_path)

        mlflow.sklearn.save_model(self.best_model, f"{output_dir}/production_mlflow")

        logger.info("Production model saved to %s", output_dir)
        return model_path, preprocessor_path

    def generate_report(self):
        """Generate training report."""
        logger.info("Generating training report...")

        report = [
            "# Heart Disease Prediction - Training Report\n\n",
            f"Experiment: {self.experiment_name}\n\n",
            "## Model Results\n\n",
        ]

        for model_name, metrics in self.results.items():
            report.extend(
                [
                    f"### {model_name}\n",
                    f"- Test Accuracy: {metrics['test_accuracy']:.4f}\n",
                    f"- Test Precision: {metrics['test_precision']:.4f}\n",
                    f"- Test Recall: {metrics['test_recall']:.4f}\n",
                    f"- Test F1-Score: {metrics['test_f1']:.4f}\n",
                    f"- Test ROC-AUC: {metrics['test_roc_auc']:.4f}\n",
                    "- CV Mean Accuracy: "
                    f"{metrics['cv_mean']:.4f} "
                    f"(+/- {metrics['cv_std']:.4f})\n",
                    f"- Overfit Gap: {metrics['overfit_gap']:.4f}\n\n",
                ]
            )

        with open("reports/training_report.md", "w") as f:
            f.writelines(report)

        logger.info("Report saved to reports/training_report.md")


def main():
    """Main training pipeline."""
    logger.info("Starting training pipeline...")

    trainer = ModelTrainer()
    df = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_features(df)

    trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)

    best_model_name, _ = trainer.select_best_model()
    trainer.save_production_model()
    trainer.generate_report()

    logger.info("Training pipeline completed successfully!")
    logger.info("Best model: %s", best_model_name)
    logger.info("Production model saved to models/")

    return trainer


if __name__ == "__main__":
    main()
