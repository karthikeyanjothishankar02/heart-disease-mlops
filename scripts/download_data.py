import pandas as pd
from ucimlrepo import fetch_ucirepo
from pathlib import Path

def download_heart_disease_data():
    """Download Heart Disease UCI dataset"""
    print("Downloading Heart Disease UCI dataset...")
    
    # Create data directory
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    try:
        # Fetch dataset
        heart_disease = fetch_ucirepo(id=45)
        
        # Extract features and targets
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        # Combine into single dataframe
        df = pd.concat([X, y], axis=1)
        
        # Save to CSV
        df.to_csv('data/raw/heart_disease.csv', index=False)
        print(f"✓ Dataset downloaded successfully: {df.shape}")
        print(f"✓ Saved to: data/raw/heart_disease.csv")
        
        return df
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_heart_disease_data()