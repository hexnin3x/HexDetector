"""
Data Loader Module for HexDetector

Handles loading and initial parsing of network traffic data from various formats
including CSV files with network flow data and PCAP files.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import Logger

try:
    from config import settings
except:
    settings = None


class DataLoader:
    """
    Handles loading and initial parsing of network traffic data from various formats
    including CSV files with network flow data and PCAP files.
    """
    
    def __init__(self):
        self.logger = Logger()
        self.supported_formats = ['csv', 'parquet', 'json']
        
    def load_raw_data(self, file_path, nrows=None):
        """
        Load raw network traffic data from the specified file path.
        
        Parameters:
        file_path (str): Path to the data file or directory
        nrows (int): Number of rows to load (None for all)
        
        Returns:
        DataFrame: The loaded raw data
        """
        self.logger.log_info(f"Loading raw data from {file_path}")
        
        if os.path.isdir(file_path):
            return self._load_from_directory(file_path, nrows)
        
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        
        if file_extension not in self.supported_formats:
            self.logger.log_error(f"Unsupported file format: {file_extension}")
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if file_extension == 'csv':
            return self._load_csv(file_path, nrows)
        elif file_extension == 'parquet':
            return self._load_parquet(file_path)
        elif file_extension == 'json':
            return self._load_json(file_path, nrows)
    
    def _load_csv(self, file_path, nrows=None):
        """Load network flow data from CSV file"""
        try:
            self.logger.log_info(f"Loading CSV file: {file_path}")
            
            # Try to load with different separators
            for sep in [',', '\t', ' ']:
                try:
                    df = pd.read_csv(file_path, nrows=nrows, sep=sep, 
                                   low_memory=False, on_bad_lines='skip')
                    if df.shape[1] > 1:  # Valid if more than one column
                        break
                except:
                    continue
            
            self.logger.log_info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Apply column mappings if specified in settings
            if settings and hasattr(settings, 'COLUMN_MAPPINGS'):
                df = df.rename(columns=settings.COLUMN_MAPPINGS)
            
            return df
            
        except Exception as e:
            self.logger.log_error(f"Error loading CSV file: {str(e)}", e)
            raise
    
    def _load_parquet(self, file_path):
        """Load data from Parquet file"""
        try:
            self.logger.log_info(f"Loading Parquet file: {file_path}")
            df = pd.read_parquet(file_path)
            self.logger.log_info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            self.logger.log_error(f"Error loading Parquet file: {str(e)}", e)
            raise
    
    def _load_json(self, file_path, nrows=None):
        """Load data from JSON file"""
        try:
            self.logger.log_info(f"Loading JSON file: {file_path}")
            df = pd.read_json(file_path, lines=True, nrows=nrows)
            self.logger.log_info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            self.logger.log_error(f"Error loading JSON file: {str(e)}", e)
            raise
    
    def _load_from_directory(self, dir_path, nrows=None):
        """Load all supported files from a directory"""
        all_data = []
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        
        self.logger.log_info(f"Found {len(files)} files in directory")
        
        for i, file in enumerate(files, 1):
            file_path = os.path.join(dir_path, file)
            file_extension = Path(file).suffix.lower().lstrip('.')
            
            if file_extension in self.supported_formats:
                self.logger.log_info(f"Loading file {i}/{len(files)}: {file}")
                try:
                    data = self.load_raw_data(file_path, nrows)
                    all_data.append(data)
                except Exception as e:
                    self.logger.log_warning(f"Skipping file {file}: {str(e)}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self.logger.log_info(f"Combined data shape: {combined.shape}")
            return combined
        
        return pd.DataFrame()

    def load_processed_data(self, file_path):
        """Load already processed data"""
        try:
            self.logger.log_info(f"Loading processed data from {file_path}")
            file_extension = Path(file_path).suffix.lower().lstrip('.')
            
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_extension == 'json':
                df = pd.read_json(file_path)
            else:
                self.logger.log_error(f"Unsupported processed data format: {file_extension}")
                raise ValueError(f"Unsupported processed data format: {file_extension}")
            
            self.logger.log_info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.log_error(f"Error loading processed data: {str(e)}", e)
            raise
    
    def save_processed_data(self, data, output_path):
        """Save processed data to file"""
        try:
            self.logger.log_info(f"Saving processed data to {output_path}")
            file_extension = Path(output_path).suffix.lower().lstrip('.')
            
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if file_extension == 'csv':
                data.to_csv(output_path, index=False)
            elif file_extension == 'parquet':
                data.to_parquet(output_path, index=False)
            elif file_extension == 'json':
                data.to_json(output_path, orient='records', lines=True)
            else:
                self.logger.log_error(f"Unsupported output format: {file_extension}")
                raise ValueError(f"Unsupported output format: {file_extension}")
            
            self.logger.log_info(f"Successfully saved {data.shape[0]} rows")
            
        except Exception as e:
            self.logger.log_error(f"Error saving processed data: {str(e)}", e)
            raise
    
    def load_iot23_scenario(self, scenario_path, attack_type=None, nrows=None):
        """
        Load data from IoT23 scenario file with optional filtering by attack type.
        
        Parameters:
        scenario_path (str): Path to IoT23 scenario file
        attack_type (str): Optional attack type to filter
        nrows (int): Number of rows to load
        
        Returns:
        DataFrame: Filtered network flow data
        """
        self.logger.log_info(f"Loading IoT23 scenario: {scenario_path}")
        
        # Load data
        df = self.load_raw_data(scenario_path, nrows=nrows)
        
        # Filter by attack type if specified
        if attack_type and 'label' in df.columns:
            original_size = len(df)
            df = df[df['label'].str.contains(attack_type, case=False, na=False)]
            self.logger.log_info(f"Filtered to {len(df)} rows (from {original_size}) for attack type: {attack_type}")
        
        return df
    
    def load_multiple_scenarios(self, scenario_paths, nrows_per_file=None):
        """
        Load data from multiple IoT23 scenario files.
        
        Parameters:
        scenario_paths (list): List of paths to scenario files
        nrows_per_file (int): Number of rows to load per file
        
        Returns:
        DataFrame: Combined data from all scenarios
        """
        all_data = []
        
        for i, path in enumerate(scenario_paths, 1):
            self.logger.log_info(f"Loading scenario {i}/{len(scenario_paths)}")
            try:
                data = self.load_iot23_scenario(path, nrows=nrows_per_file)
                all_data.append(data)
            except Exception as e:
                self.logger.log_warning(f"Error loading {path}: {str(e)}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self.logger.log_info(f"Combined data from {len(all_data)} scenarios: {combined.shape}")
            return combined
        
        return pd.DataFrame()
    
    def get_attack_types(self, data):
        """
        Extract unique attack types from the data.
        
        Parameters:
        data (DataFrame): Network flow data with label column
        
        Returns:
        list: Unique attack types
        """
        if 'label' in data.columns:
            attack_types = data['label'].unique().tolist()
            self.logger.log_info(f"Found {len(attack_types)} unique attack types")
            return attack_types
        else:
            self.logger.log_warning("No 'label' column found in data")
            return []
    
    def get_data_info(self, data):
        """
        Get comprehensive information about the loaded data.
        
        Parameters:
        data (DataFrame): Network flow data
        
        Returns:
        dict: Dictionary with data statistics
        """
        info = {
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'columns': data.columns.tolist(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'missing_values': data.isnull().sum().to_dict(),
            'dtypes': data.dtypes.to_dict()
        }
        
        if 'label' in data.columns:
            info['attack_types'] = data['label'].value_counts().to_dict()
        
        return info