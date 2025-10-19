"""
Feature Building Module for HexDetector

Orchestrates the feature engineering process for network traffic data
by combining multiple feature extraction techniques.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.logger import Logger
    from features.feature_extraction import extract_features
except:
    class Logger:
        def log_info(self, msg): print(f"[INFO] {msg}")
        def log_error(self, msg, e=None): print(f"[ERROR] {msg}")
        def log_warning(self, msg): print(f"[WARN] {msg}")
    
    def extract_features(data, time_window=60):
        """Fallback feature extraction"""
        return pd.DataFrame()


def build_features(raw_data, time_windows=None):
    """
    Build comprehensive features from preprocessed network traffic data.
    
    Parameters:
    raw_data (DataFrame): Preprocessed network traffic data
    time_windows (list): List of time windows for feature extraction (in seconds)
    
    Returns:
    DataFrame: DataFrame with all extracted features
    """
    logger = Logger()
    logger.log_info("Starting feature engineering")
    logger.log_info(f"Input data shape: {raw_data.shape}")
    
    if raw_data.empty:
        logger.log_error("Empty dataset provided for feature building")
        return pd.DataFrame()
    
    # Use default time windows if not provided
    if time_windows is None:
        time_windows = [60, 300, 600]  # 1 min, 5 min, 10 min
    
    # Start with basic statistical features
    features_df = extract_basic_statistical_features(raw_data)
    
    # Add flow-based features
    flow_features = extract_flow_based_features(raw_data)
    if not flow_features.empty:
        features_df = pd.concat([features_df, flow_features], axis=1)
    
    # Add protocol-specific features
    protocol_features = extract_protocol_features(raw_data)
    if not protocol_features.empty:
        features_df = pd.concat([features_df, protocol_features], axis=1)
    
    # Add time-based features
    time_features = extract_time_based_features(raw_data)
    if not time_features.empty:
        features_df = pd.concat([features_df, time_features], axis=1)
    
    # Add network topology features
    topology_features = extract_topology_features(raw_data)
    if not topology_features.empty:
        features_df = pd.concat([features_df, topology_features], axis=1)
    
    # Add behavioral features
    behavioral_features = extract_behavioral_features(raw_data)
    if not behavioral_features.empty:
        features_df = pd.concat([features_df, behavioral_features], axis=1)
    
    # Preserve target/label column if it exists
    target_cols = ['target', 'label', 'Label', 'class', 'Class']
    for col in target_cols:
        if col in raw_data.columns:
            features_df[col] = raw_data[col].iloc[0] if len(raw_data) > 0 else None
            logger.log_info(f"Preserved target column: {col}")
            break
    
    # Remove duplicate columns
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]
    
    logger.log_info(f"Feature engineering complete. Total features: {features_df.shape[1]}")
    
    return features_df


def extract_basic_statistical_features(data):
    """Extract basic statistical features from the data"""
    logger = Logger()
    logger.log_info("Extracting basic statistical features")
    
    features = pd.DataFrame(index=[0])
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Exclude target columns
    exclude_cols = ['target', 'label', 'Label', 'class', 'Class']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for col in numeric_cols[:20]:  # Limit to first 20 numeric columns
        try:
            features[f'{col}_mean'] = data[col].mean()
            features[f'{col}_std'] = data[col].std()
            features[f'{col}_min'] = data[col].min()
            features[f'{col}_max'] = data[col].max()
            features[f'{col}_median'] = data[col].median()
        except:
            pass
    
    logger.log_info(f"Extracted {len(features.columns)} basic statistical features")
    return features


def extract_flow_based_features(data):
    """Extract features based on network flows"""
    logger = Logger()
    logger.log_info("Extracting flow-based features")
    
    features = pd.DataFrame(index=[0])
    
    # Check if necessary columns exist
    if 'dur' in data.columns or 'duration' in data.columns:
        dur_col = 'dur' if 'dur' in data.columns else 'duration'
        features['flow_duration_mean'] = data[dur_col].mean()
        features['flow_duration_std'] = data[dur_col].std()
        features['flow_duration_max'] = data[dur_col].max()
    
    # Packet counts
    if 'spkts' in data.columns:
        features['src_packets_mean'] = data['spkts'].mean()
        features['src_packets_sum'] = data['spkts'].sum()
    
    if 'dpkts' in data.columns:
        features['dst_packets_mean'] = data['dpkts'].mean()
        features['dst_packets_sum'] = data['dpkts'].sum()
    
    # Byte counts
    if 'sbytes' in data.columns:
        features['src_bytes_mean'] = data['sbytes'].mean()
        features['src_bytes_sum'] = data['sbytes'].sum()
    
    if 'dbytes' in data.columns:
        features['dst_bytes_mean'] = data['dbytes'].mean()
        features['dst_bytes_sum'] = data['dbytes'].sum()
    
    # Calculate ratios if both columns exist
    if 'spkts' in data.columns and 'dpkts' in data.columns:
        total_pkts = data['spkts'] + data['dpkts']
        total_pkts = total_pkts.replace(0, 1)  # Avoid division by zero
        features['src_pkt_ratio'] = (data['spkts'] / total_pkts).mean()
    
    if 'sbytes' in data.columns and 'dbytes' in data.columns:
        total_bytes = data['sbytes'] + data['dbytes']
        total_bytes = total_bytes.replace(0, 1)
        features['src_byte_ratio'] = (data['sbytes'] / total_bytes).mean()
    
    logger.log_info(f"Extracted {len(features.columns)} flow-based features")
    return features


def extract_protocol_features(data):
    """Extract protocol-specific features"""
    logger = Logger()
    logger.log_info("Extracting protocol features")
    
    features = pd.DataFrame(index=[0])
    
    # Protocol distribution
    if 'proto' in data.columns or 'protocol' in data.columns:
        proto_col = 'proto' if 'proto' in data.columns else 'protocol'
        
        protocol_counts = data[proto_col].value_counts(normalize=True)
        for protocol, ratio in protocol_counts.items():
            features[f'proto_{protocol}_ratio'] = ratio
    
    # Service distribution
    if 'service' in data.columns:
        service_counts = data['service'].value_counts(normalize=True)
        for i, (service, ratio) in enumerate(service_counts.head(10).items()):
            features[f'service_{i}_ratio'] = ratio
    
    # State distribution
    if 'state' in data.columns:
        state_counts = data['state'].value_counts(normalize=True)
        for state, ratio in state_counts.items():
            features[f'state_{state}_ratio'] = ratio
    
    # TCP flags
    if 'swin' in data.columns:
        features['src_window_mean'] = data['swin'].mean()
    
    if 'dwin' in data.columns:
        features['dst_window_mean'] = data['dwin'].mean()
    
    logger.log_info(f"Extracted {len(features.columns)} protocol features")
    return features


def extract_time_based_features(data):
    """Extract time-based features"""
    logger = Logger()
    logger.log_info("Extracting time-based features")
    
    features = pd.DataFrame(index=[0])
    
    # Inter-arrival times
    if 'sinpkt' in data.columns:
        features['src_inter_pkt_mean'] = data['sinpkt'].mean()
        features['src_inter_pkt_std'] = data['sinpkt'].std()
    
    if 'dinpkt' in data.columns:
        features['dst_inter_pkt_mean'] = data['dinpkt'].mean()
        features['dst_inter_pkt_std'] = data['dinpkt'].std()
    
    # Jitter
    if 'sjit' in data.columns:
        features['src_jitter_mean'] = data['sjit'].mean()
    
    if 'djit' in data.columns:
        features['dst_jitter_mean'] = data['djit'].mean()
    
    # Round-trip time
    if 'tcprtt' in data.columns:
        features['tcp_rtt_mean'] = data['tcprtt'].mean()
        features['tcp_rtt_max'] = data['tcprtt'].max()
    
    logger.log_info(f"Extracted {len(features.columns)} time-based features")
    return features


def extract_topology_features(data):
    """Extract network topology features"""
    logger = Logger()
    logger.log_info("Extracting topology features")
    
    features = pd.DataFrame(index=[0])
    
    # Connection patterns
    if 'ct_srv_src' in data.columns:
        features['conn_srv_src_mean'] = data['ct_srv_src'].mean()
    
    if 'ct_srv_dst' in data.columns:
        features['conn_srv_dst_mean'] = data['ct_srv_dst'].mean()
    
    if 'ct_dst_ltm' in data.columns:
        features['conn_dst_ltm_mean'] = data['ct_dst_ltm'].mean()
    
    if 'ct_src_dport_ltm' in data.columns:
        features['conn_src_dport_ltm_mean'] = data['ct_src_dport_ltm'].mean()
    
    if 'ct_dst_sport_ltm' in data.columns:
        features['conn_dst_sport_ltm_mean'] = data['ct_dst_sport_ltm'].mean()
    
    if 'ct_dst_src_ltm' in data.columns:
        features['conn_dst_src_ltm_mean'] = data['ct_dst_src_ltm'].mean()
    
    logger.log_info(f"Extracted {len(features.columns)} topology features")
    return features


def extract_behavioral_features(data):
    """Extract behavioral features indicating attack patterns"""
    logger = Logger()
    logger.log_info("Extracting behavioral features")
    
    features = pd.DataFrame(index=[0])
    
    # FTP-related features
    if 'is_ftp_login' in data.columns:
        features['ftp_login_ratio'] = data['is_ftp_login'].mean()
    
    if 'ct_ftp_cmd' in data.columns:
        features['ftp_cmd_count_mean'] = data['ct_ftp_cmd'].mean()
    
    # HTTP-related features
    if 'ct_flw_http_mthd' in data.columns:
        features['http_method_count_mean'] = data['ct_flw_http_mthd'].mean()
    
    # Port scanning indicators
    if 'is_sm_ips_ports' in data.columns:
        features['same_ip_port_ratio'] = data['is_sm_ips_ports'].mean()
    
    # Data transfer characteristics
    if 'response_body_len' in data.columns:
        features['response_body_len_mean'] = data['response_body_len'].mean()
        features['response_body_len_max'] = data['response_body_len'].max()
    
    if 'trans_depth' in data.columns:
        features['trans_depth_mean'] = data['trans_depth'].mean()
    
    # Loss rates
    if 'sloss' in data.columns:
        features['src_loss_mean'] = data['sloss'].mean()
    
    if 'dloss' in data.columns:
        features['dst_loss_mean'] = data['dloss'].mean()
    
    # Load rates
    if 'sload' in data.columns:
        features['src_load_mean'] = data['sload'].mean()
    
    if 'dload' in data.columns:
        features['dst_load_mean'] = data['dload'].mean()
    
    logger.log_info(f"Extracted {len(features.columns)} behavioral features")
    return features