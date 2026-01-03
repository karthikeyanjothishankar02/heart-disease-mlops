# tests/test_model.py
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

@pytest.fixture
def sample_model_data():
    """Create sample data for model testing"""
    np.random.seed(42)
    X = np.random.randn(100, 13)
    y = np.random.randint(0, 2, 100)
    return X, y

class TestModelTraining:
    """Test model training functionality"""
    
    def test_logistic_regression_training(self, sample_model_data):
        """Test Logistic Regression can be trained"""
        X, y = sample_model_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')
        assert model.coef_.shape[1] == X.shape[1]
    
    def test_random_forest_training(self, sample_model_data):
        """Test Random Forest can be trained"""
        X, y = sample_model_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]
    
    def test_model_prediction(self, sample_model_data):
        """Test model can make predictions"""
        X, y = sample_model_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})
    
    def test_model_probability_prediction(self, sample_model_data):
        """Test model can predict probabilities"""
        X, y = sample_model_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(y), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_model_accuracy(self, sample_model_data):
        """Test model achieves reasonable accuracy"""
        X, y = sample_model_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        # Should at least be better than random guessing
        assert accuracy > 0.4
    
    def test_model_serialization(self, sample_model_data, tmp_path):
        """Test model can be saved and loaded"""
        X, y = sample_model_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Save model
        filepath = tmp_path / "test_model.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # Load model
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Check predictions are the same
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)

class TestModelValidation:
    """Test model validation procedures"""
    
    def test_train_test_split(self, sample_model_data):
        """Test that train-test split works correctly"""
        from sklearn.model_selection import train_test_split
        X, y = sample_model_data
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_cross_validation(self, sample_model_data):
        """Test cross-validation works"""
        from sklearn.model_selection import cross_val_score
        X, y = sample_model_data
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(model, X, y, cv=5)
        
        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores)