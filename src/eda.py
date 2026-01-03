# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load dataset"""
    return pd.read_csv(filepath)

def perform_eda(df):
    """Comprehensive EDA"""
    
    # Basic info
    print("Dataset Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nBasic Statistics:\n", df.describe())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Target distribution
    df['num'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Heart Disease Distribution')
    axes[0,0].set_xlabel('Disease Presence (0=No, 1-4=Yes)')
    
    # 2. Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', ax=axes[0,1], cmap='coolwarm')
    axes[0,1].set_title('Feature Correlation Heatmap')
    
    # 3. Age distribution
    df['age'].hist(bins=20, ax=axes[1,0], edgecolor='black')
    axes[1,0].set_title('Age Distribution')
    axes[1,0].set_xlabel('Age')
    
    # 4. Cholesterol by disease
    df.boxplot(column='chol', by='num', ax=axes[1,1])
    axes[1,1].set_title('Cholesterol Levels by Disease Status')
    
    plt.tight_layout()
    plt.savefig('reports/eda_visualizations.png', dpi=300)
    plt.show()
    
    return df

def clean_data(df):
    """Clean and preprocess data"""
    # Handle missing values
    df = df.dropna()
    
    # Convert target to binary (0: no disease, 1: disease)
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop('num', axis=1)
    
    # Save cleaned data
    df.to_csv('data/processed/heart_disease_clean.csv', index=False)
    
    return df