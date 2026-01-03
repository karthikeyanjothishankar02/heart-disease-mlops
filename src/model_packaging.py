# src/model_packaging.py
import pickle
import mlflow

def save_model_artifacts(model, preprocessor, filepath_prefix="models/production"):
    """Save model and preprocessor"""
    # Save with pickle
    with open(f"{filepath_prefix}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(f"{filepath_prefix}_preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Save with MLflow
    mlflow.sklearn.save_model(model, f"{filepath_prefix}_mlflow")
    
    print(f"Model artifacts saved to {filepath_prefix}")

def load_model_artifacts(filepath_prefix="models/production"):
    """Load model and preprocessor"""
    with open(f"{filepath_prefix}_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open(f"{filepath_prefix}_preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)
    
    return model, preprocessor