import pandas as pd
from pathlib import Path

def clean_data(input_path='data/raw/heart_disease.csv', 
               output_path='data/processed/heart_disease_clean.csv'):
    """Clean and prepare data"""
    print("Cleaning data...")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Handle missing values
    df = df.dropna()
    
    # Convert target to binary
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop('num', axis=1)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    print(f"✓ Data cleaned: {df.shape}")
    print(f"✓ Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    clean_data()