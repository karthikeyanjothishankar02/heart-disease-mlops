# src/model_training.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model."""
    with mlflow.start_run(run_name="Logistic_Regression"):
        # Train model
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Cross-validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
        )

        # Log parameters and metrics
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(
            f"Logistic Regression - Accuracy: {accuracy:.4f}, "
            f"ROC-AUC: {roc_auc:.4f}"
        )

        return model, {"accuracy": accuracy, "roc_auc": roc_auc}


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    with mlflow.start_run(run_name="Random_Forest"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Cross-validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
        )

        # Log parameters and metrics
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Random Forest - Accuracy: {accuracy:.4f}, " f"ROC-AUC: {roc_auc:.4f}")

        return model, {"accuracy": accuracy, "roc_auc": roc_auc}
