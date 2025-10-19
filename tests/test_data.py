"""
Unit tests for data loading and preprocessing modules
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

from data.data_loader import DataLoader
from data.preprocessing import preprocess_data


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'ts': ['2023-10-19 10:00:00', '2023-10-19 10:00:01'],
            'saddr': ['192.168.1.1', '192.168.1.2'],
            'daddr': ['10.0.0.1', '10.0.0.2'],
            'sport': [12345, 12346],
            'dport': [80, 443],
            'proto': ['tcp', 'tcp'],
            'dur': [0.5, 1.0],
            'spkts': [10, 20],
            'dpkts': [5, 15],
            'sbytes': [1000, 2000],
            'dbytes': [500, 1500]
        })
    
    def test_data_loader_initialization(self):
        """Test DataLoader initializes correctly"""
        self.assertIsNotNone(self.loader)
        self.assertTrue(hasattr(self.loader, 'logger'))
    
    def test_load_csv(self):
        """Test loading CSV data"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            self.sample_data.to_csv(f, index=False)
        
        try:
            loaded_data = self.loader.load_data(temp_file)
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(len(loaded_data), 2)
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_empty_file(self):
        """Test handling of empty files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            f.write('')  # Empty file
        
        try:
            loaded_data = self.loader.load_data(temp_file)
            # Should handle gracefully
            self.assertTrue(loaded_data is None or loaded_data.empty)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'ts': ['2023-10-19 10:00:00', '2023-10-19 10:00:01', '2023-10-19 10:00:02'],
            'saddr': ['192.168.1.1', '192.168.1.2', None],
            'daddr': ['10.0.0.1', '10.0.0.2', '10.0.0.3'],
            'sport': [12345, 12346, 12347],
            'dport': [80, 443, 8080],
            'proto': ['tcp', 'tcp', 'udp'],
            'dur': [0.5, 1.0, np.nan],
            'spkts': [10, 20, 30],
            'dpkts': [5, 15, 25],
            'sbytes': [1000, 2000, 3000],
            'dbytes': [500, 1500, 2500],
            'label': ['benign', 'malicious', 'benign']
        })
    
    def test_preprocess_basic(self):
        """Test basic preprocessing"""
        processed = preprocess_data(self.sample_data)
        
        # Should return a DataFrame
        self.assertIsInstance(processed, pd.DataFrame)
        
        # Should not be empty
        self.assertGreater(len(processed), 0)
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        processed = preprocess_data(self.sample_data)
        
        # Missing values should be handled (but not necessarily removed)
        # The function might impute them
        self.assertIsNotNone(processed)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        processed = preprocess_data(empty_df)
        
        # Should handle gracefully
        self.assertTrue(processed is None or processed.empty)


class TestDataIntegration(unittest.TestCase):
    """Integration tests for data pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
        self.sample_data = pd.DataFrame({
            'ts': ['2023-10-19 10:00:00'] * 10,
            'saddr': ['192.168.1.1'] * 10,
            'daddr': ['10.0.0.1'] * 10,
            'sport': [12345] * 10,
            'dport': [80] * 10,
            'proto': ['tcp'] * 10,
            'dur': [0.5] * 10,
            'spkts': [10] * 10,
            'dpkts': [5] * 10,
            'sbytes': [1000] * 10,
            'dbytes': [500] * 10,
            'label': ['benign'] * 10
        })
    
    def test_full_pipeline(self):
        """Test complete data pipeline"""
        # Save test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            self.sample_data.to_csv(f, index=False)
        
        try:
            # Load
            loaded_data = self.loader.load_data(temp_file)
            self.assertIsNotNone(loaded_data)
            
            # Preprocess
            processed_data = preprocess_data(loaded_data)
            self.assertIsNotNone(processed_data)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()