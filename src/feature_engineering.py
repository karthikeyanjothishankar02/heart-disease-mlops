# src/feature_engineering.py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_features = None
        
    def fit_transform(self, df, target_col='target'):
        """Prepare features for modeling"""
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Identify numeric features
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        
        # Scale numeric features
        X_scaled = X.copy()
        X_scaled[self.numeric_features] = self.scaler.fit_transform(X[self.numeric_features])
        
        return X_scaled, y
    
    def transform(self, df):
        """Transform new data"""
        X_scaled = df.copy()
        X_scaled[self.numeric_features] = self.scaler.transform(df[self.numeric_features])
        return X_scaled
    
    def save(self, filepath):
        """Save preprocessor"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)