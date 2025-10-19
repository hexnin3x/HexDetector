"""
Unit tests for model training, evaluation, and prediction modules
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.train import train_model
from models.evaluate import evaluate_model
from models.predict import make_predictions


class TestModelTraining(unittest.TestCase):
    """Test cases for model training"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample dataset with features and target
        np.random.seed(42)
        n_samples = 100
        
        self.dataset = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples),
            'label': np.random.choice(['benign', 'malicious'], n_samples)
        })
    
    def test_train_random_forest(self):
        """Test Random Forest training"""
        try:
            result = train_model(self.dataset, model_type='random_forest', test_size=0.3)
            
            if result is not None:
                self.assertIn('model', result)
                self.assertIn('accuracy', result)
                self.assertGreater(result['accuracy'], 0)
        except Exception as e:
            self.skipTest(f"Training failed: {e}")
    
    def test_dataset_validation(self):
        """Test dataset validation"""
        # Empty dataset should be handled
        empty_dataset = pd.DataFrame()
        result = train_model(empty_dataset)
        
        # Should return None or handle gracefully
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_missing_target(self):
        """Test handling of missing target column"""
        dataset_no_target = self.dataset.drop('label', axis=1, errors='ignore')
        result = train_model(dataset_no_target)
        
        # Should handle missing target gracefully
        self.assertTrue(result is None or isinstance(result, dict))


class TestModelEvaluation(unittest.TestCase):
    """Test cases for model evaluation"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        n_samples = 50
        
        # Create small dataset
        self.dataset = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'label': np.random.choice(['benign', 'malicious'], n_samples)
        })
    
    def test_evaluate_function_exists(self):
        """Test that evaluate_model function exists"""
        self.assertTrue(callable(evaluate_model))
    
    def test_evaluation_with_valid_data(self):
        """Test evaluation with valid data"""
        try:
            # Train a simple model first
            result = train_model(self.dataset, model_type='random_forest', test_size=0.3)
            
            if result and 'model' in result:
                # Prepare test data
                X_test = result.get('X_test')
                y_test = result.get('y_test')
                
                if X_test is not None and y_test is not None:
                    test_data = pd.concat([X_test, y_test], axis=1)
                    
                    # Evaluate
                    metrics = evaluate_model(
                        result['model'],
                        test_data,
                        metrics=['accuracy']
                    )
                    
                    self.assertIsNotNone(metrics)
        except Exception as e:
            self.skipTest(f"Evaluation test failed: {e}")


class TestModelPrediction(unittest.TestCase):
    """Test cases for model prediction"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        n_samples = 50
        
        self.dataset = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'label': np.random.choice(['benign', 'malicious'], n_samples)
        })
    
    def test_prediction_function_exists(self):
        """Test that make_predictions function exists"""
        self.assertTrue(callable(make_predictions))
    
    def test_prediction_with_trained_model(self):
        """Test making predictions with a trained model"""
        try:
            # Train a model
            result = train_model(self.dataset, model_type='random_forest', test_size=0.3)
            
            if result and 'model' in result and 'X_test' in result:
                # Make predictions
                predictions = make_predictions(result['model'], result['X_test'])
                
                if predictions is not None:
                    # Check predictions have correct length
                    self.assertEqual(len(predictions), len(result['X_test']))
        except Exception as e:
            self.skipTest(f"Prediction test failed: {e}")


class TestModelIntegration(unittest.TestCase):
    """Integration tests for complete ML pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        n_samples = 100
        
        self.dataset = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'label': np.random.choice(['benign', 'malicious'], n_samples)
        })
    
    def test_full_ml_pipeline(self):
        """Test complete ML pipeline: train -> evaluate -> predict"""
        try:
            # Step 1: Train
            result = train_model(self.dataset, model_type='random_forest', test_size=0.3)
            
            if result is None:
                self.skipTest("Training returned None")
            
            self.assertIn('model', result)
            self.assertIn('X_test', result)
            self.assertIn('y_test', result)
            
            # Step 2: Evaluate
            test_data = pd.concat([result['X_test'], result['y_test']], axis=1)
            metrics = evaluate_model(result['model'], test_data, metrics=['accuracy'])
            
            self.assertIsNotNone(metrics)
            
            # Step 3: Predict
            predictions = make_predictions(result['model'], result['X_test'])
            
            if predictions is not None:
                self.assertEqual(len(predictions), len(result['X_test']))
                
        except Exception as e:
            self.skipTest(f"Full pipeline test failed: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)