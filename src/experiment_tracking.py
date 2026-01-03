# src/experiment_tracking.py
import mlflow
from train_pipeline import train_logistic_regression, train_random_forest


def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("heart_disease_prediction")


# Run experiments
def run_all_experiments(X_train, y_train, X_test, y_test):
    setup_mlflow()

    models = {}
    models["lr"] = train_logistic_regression(X_train, y_train, X_test, y_test)
    models["rf"] = train_random_forest(X_train, y_train, X_test, y_test)

    return models
