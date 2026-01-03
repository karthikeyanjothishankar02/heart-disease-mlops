# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer
from src.eda import clean_data

@pytest.fixture
def sample_data():
    """Create sample heart disease data for testing"""
    data = {
        'age': [63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 0],
        'cp': [1, 4, 4, 3, 2],
        'trestbps': [145, 160, 120, 130, 130],
        'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [2, 2, 2, 0, 2],
        'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
        'slope': [3, 2, 2, 3, 1],
        'ca': [0, 3, 2, 0, 0],
        'thal': [6, 3, 7, 3, 3],
        'num': [0, 2, 1, 0, 0]
    }
    return pd.DataFrame(data)

class TestDataProcessing:
    """Test data processing functions"""
    
    def test_data_loading(self, sample_data):
        """Test that data loads correctly"""
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) > 0
        assert 'age' in sample_data.columns
        assert 'num' in sample_data.columns
    
    def test_missing_values(self, sample_data):
        """Test handling of missing values"""
        # Add some missing values
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.loc[0, 'age'] = np.nan
        
        # Clean should remove rows with missing values
        cleaned = clean_data(sample_data_with_nan)
        assert cleaned.isnull().sum().sum() == 0
    
    def test_target_creation(self, sample_data):
        """Test binary target creation"""
        cleaned = clean_data(sample_data)
        assert 'target' in cleaned.columns
        assert set(cleaned['target'].unique()).issubset({0, 1})
        assert 'num' not in cleaned.columns
    
    def test_data_shape(self, sample_data):
        """Test data dimensions are preserved"""
        cleaned = clean_data(sample_data)
        assert cleaned.shape[0] == sample_data.shape[0]
        # One less column (num replaced by target)
        assert cleaned.shape[1] == sample_data.shape[1]

class TestFeatureEngineering:
    """Test feature engineering pipeline"""
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization"""
        fe = FeatureEngineer()
        assert fe.scaler is not None
        assert fe.numeric_features is None
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform produces correct output"""
        cleaned = clean_data(sample_data)
        fe = FeatureEngineer()
        X_scaled, y = fe.fit_transform(cleaned)
        
        assert X_scaled.shape[0] == cleaned.shape[0]
        assert len(y) == cleaned.shape[0]
        assert fe.numeric_features is not None
    
    def test_scaling(self, sample_data):
        """Test that features are properly scaled"""
        cleaned = clean_data(sample_data)
        fe = FeatureEngineer()
        X_scaled, y = fe.fit_transform(cleaned)
        
        # Check that scaled features have mean close to 0 and std close to 1
        for col in fe.numeric_features:
            mean_val = X_scaled[col].mean()
            std_val = X_scaled[col].std()
            assert abs(mean_val) < 0.1  # Mean should be close to 0
            assert abs(std_val - 1.0) < 0.2  # Std should be close to 1
    
    def test_transform_consistency(self, sample_data):
        """Test that transform produces consistent results"""
        cleaned = clean_data(sample_data)
        fe = FeatureEngineer()
        X_scaled_1, _ = fe.fit_transform(cleaned)
        
        # Transform same data again
        X_scaled_2 = fe.transform(cleaned.drop('target', axis=1))
        
        # Results should be identical
        pd.testing.assert_frame_equal(X_scaled_1, X_scaled_2)
    
    def test_save_load(self, sample_data, tmp_path):
        """Test saving and loading preprocessor"""
        cleaned = clean_data(sample_data)
        fe = FeatureEngineer()
        fe.fit_transform(cleaned)
        
        # Save
        filepath = tmp_path / "preprocessor.pkl"
        fe.save(filepath)
        
        # Check file exists
        assert filepath.exists()