"""
Feature extraction module for network traffic data
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import Logger


class FeatureExtractor:
    """Extract various features from network traffic data"""
    
    def __init__(self):
        self.logger = Logger()
    
    def extract_all_features(self, data, time_window=60):
        """
        Extract all types of features from network traffic data
        
        Parameters:
        data (DataFrame): Preprocessed network traffic data
        time_window (int): Time window in seconds for aggregating features
        
        Returns:
        DataFrame: Extracted features
        """
        if data.empty:
            self.logger.log_warning("Empty dataset provided for feature extraction")
            return pd.DataFrame()
        
        self.logger.log_info(f"Extracting features with time window of {time_window} seconds")
        
        features = pd.DataFrame()
        
        # Extract different types of features
        basic_features = self.extract_basic_features(data)
        flow_features = self.extract_flow_features(data)
        time_features = self.extract_time_window_features(data, time_window)
        network_features = self.extract_network_features(data)
        behavioral_features = self.extract_behavioral_features(data)
        
        # Combine all features
        for feat_df in [basic_features, flow_features, time_features, network_features, behavioral_features]:
            if not feat_df.empty:
                features = pd.concat([features, feat_df], axis=1)
        
        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]
        
        self.logger.log_info(f"Feature extraction complete. Total features: {features.shape[1]}")
        return features
    
    def extract_basic_features(self, data):
        """Extract basic statistical features"""
        features = pd.DataFrame(index=data.index)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['label', 'Label', 'target', 'class', 'Class']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols[:10]:  # Limit to prevent feature explosion
            if col in data.columns:
                features[f'{col}_value'] = data[col]
        
        return features
    
    def extract_flow_features(self, data):
        """Extract flow-based features"""
        features = pd.DataFrame(index=data.index)
        
        # Duration features
        if 'dur' in data.columns:
            features['flow_duration'] = data['dur']
        
        # Packet counts
        if 'spkts' in data.columns:
            features['src_packets'] = data['spkts']
        
        if 'dpkts' in data.columns:
            features['dst_packets'] = data['dpkts']
        
        # Bytes
        if 'sbytes' in data.columns:
            features['src_bytes'] = data['sbytes']
        
        if 'dbytes' in data.columns:
            features['dst_bytes'] = data['dbytes']
        
        # Ratios
        if 'spkts' in data.columns and 'dpkts' in data.columns:
            total_pkts = data['spkts'] + data['dpkts']
            features['packet_ratio'] = data['spkts'] / total_pkts.replace(0, 1)
        
        if 'sbytes' in data.columns and 'dbytes' in data.columns:
            total_bytes = data['sbytes'] + data['dbytes']
            features['byte_ratio'] = data['sbytes'] / total_bytes.replace(0, 1)
        
        return features
    
    def extract_time_window_features(self, data, window_seconds):
        """Extract time-based features"""
        features = pd.DataFrame(index=data.index)
        
        if 'ts' in data.columns or 'timestamp' in data.columns:
            ts_col = 'ts' if 'ts' in data.columns else 'timestamp'
            
            try:
                if not pd.api.types.is_datetime64_any_dtype(data[ts_col]):
                    timestamps = pd.to_datetime(data[ts_col], errors='coerce')
                else:
                    timestamps = data[ts_col]
                
                # Extract time components
                features['hour'] = timestamps.dt.hour
                features['day_of_week'] = timestamps.dt.dayofweek
                
            except Exception as e:
                self.logger.log_warning(f"Could not process timestamps: {e}")
        
        return features
    
    def extract_network_features(self, data):
        """Extract network topology features"""
        features = pd.DataFrame(index=data.index)
        
        # Protocol features
        if 'proto' in data.columns:
            # Convert protocol to numeric if not already
            if data['proto'].dtype == 'object':
                proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
                features['proto_numeric'] = data['proto'].map(proto_map).fillna(0)
            else:
                features['proto_numeric'] = data['proto']
        
        # Port features
        if 'sport' in data.columns:
            features['src_port'] = data['sport']
        
        if 'dport' in data.columns:
            features['dst_port'] = data['dport']
            # Common service ports
            features['is_http'] = (data['dport'] == 80).astype(int)
            features['is_https'] = (data['dport'] == 443).astype(int)
            features['is_dns'] = (data['dport'] == 53).astype(int)
        
        return features
    
    def extract_behavioral_features(self, data):
        """Extract behavioral pattern features"""
        features = pd.DataFrame(index=data.index)
        
        # Simple behavioral indicators
        if 'dur' in data.columns and 'spkts' in data.columns:
            # Packets per second
            features['packets_per_second'] = data['spkts'] / data['dur'].replace(0, 1)
        
        if 'sbytes' in data.columns and 'spkts' in data.columns:
            # Average packet size
            features['avg_packet_size'] = data['sbytes'] / data['spkts'].replace(0, 1)
        
        return features


def extract_features(data, time_window=60):
    """
    Convenience function for feature extraction
    
    Parameters:
    data (DataFrame): Input data
    time_window (int): Time window for feature aggregation
    
    Returns:
    DataFrame: Extracted features
    """
    extractor = FeatureExtractor()
    return extractor.extract_all_features(data, time_window)