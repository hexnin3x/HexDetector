"""
HexDetector Configuration Settings

This file contains all configuration settings for the HexDetector project
including paths, model parameters, and experiment settings.
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
# Define project base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
SRC_DIR = BASE_DIR / 'src'
LOGS_DIR = BASE_DIR / 'logs'
OUTPUT_DIR = BASE_DIR / 'output'

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data paths
DATA_RAW_PATH = DATA_DIR / 'raw'
DATA_PROCESSED_PATH = DATA_DIR / 'processed'
DATA_EXPERIMENTS_PATH = DATA_DIR / 'experiments'

# IoT23 Dataset paths (configure these to match your setup)
IOT23_SCENARIOS_DIR = os.getenv('IOT23_SCENARIOS_DIR', str(DATA_RAW_PATH / 'iot23_scenarios'))
IOT23_ATTACKS_DIR = os.getenv('IOT23_ATTACKS_DIR', str(DATA_PROCESSED_PATH / 'iot23_attacks'))
IOT23_DATA_DIR = os.getenv('IOT23_DATA_DIR', str(DATA_PROCESSED_PATH / 'iot23_data'))
IOT23_EXPERIMENTS_DIR = os.getenv('IOT23_EXPERIMENTS_DIR', str(DATA_EXPERIMENTS_PATH))

# Model storage
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== MODEL CONFIGURATIONS ====================
# Available model types
MODEL_TYPES = {
    'random_forest': 'RandomForestClassifier',
    'svm': 'SVC',
    'naive_bayes': 'GaussianNB',
    'logistic_regression': 'LogisticRegression',
    'decision_tree': 'DecisionTreeClassifier',
    'xgboost': 'XGBClassifier',
    'gradient_boosting': 'GradientBoostingClassifier'
}

DEFAULT_MODEL_TYPE = 'random_forest'

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'random_state': 42
    },
    'naive_bayes': {
        'var_smoothing': 1e-9
    },
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42,
        'n_jobs': -1
    },
    'decision_tree': {
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
}

# ==================== FEATURE EXTRACTION ====================
# Feature extraction settings
DEFAULT_TIME_WINDOWS = [60, 300, 600]  # Time windows in seconds (1 min, 5 min, 10 min)

# Network flow features to extract
FLOW_FEATURES = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
    'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
    'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
    'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
    'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
    'is_sm_ips_ports'
]

# Attack types in IoT23 dataset
ATTACK_TYPES = [
    'Benign',
    'DDoS',
    'PartOfAHorizontalPortScan',
    'C&C',
    'FileDownload',
    'Okiru',
    'Torii',
    'Mirai',
    'Attack',
    'Malicious'
]

# Binary classification labels
BINARY_LABELS = {
    'Benign': 0,
    'Malicious': 1
}

# ==================== DATA PROCESSING ====================
# Data processing settings
CHUNK_SIZE = 100000  # Number of rows to process at once
MAX_MEMORY_GB = 8  # Maximum memory to use in GB
SHUFFLE_PARTITION_SIZE = 1024 * 1024 * 1024  # 1 GB partitions for shuffling

# Missing value handling
MISSING_VALUE_STRATEGY = 'median'  # 'mean', 'median', 'mode', 'drop'
MISSING_THRESHOLD = 0.7  # Drop columns with more than 70% missing values

# Feature scaling
SCALING_METHOD = 'standard'  # 'standard', 'minmax', 'robust', 'none'

# ==================== TRAINING SETTINGS ====================
# Training/testing split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
STRATIFY = True  # Use stratified sampling

# Cross-validation
CV_FOLDS = 5
CV_SCORING = 'f1_weighted'

# Class imbalance handling
HANDLE_IMBALANCE = True
IMBALANCE_METHOD = 'smote'  # 'smote', 'undersample', 'oversample', 'class_weight'

# ==================== EXPERIMENT SETTINGS ====================
# Demo mode settings
DEMO_SAMPLES_PER_FILE = 10000
DEMO_MAX_FILES = 5

# Full experiment settings
FULL_EXPERIMENT_SAMPLES = None  # None means use all available data

# Custom experiment settings
CUSTOM_SAMPLES_DEFAULT = 100000

# ==================== EVALUATION METRICS ====================
# Metrics to compute
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'confusion_matrix',
    'classification_report'
]

# Threshold for binary classification
CLASSIFICATION_THRESHOLD = 0.5

# ==================== LOGGING SETTINGS ====================
# Logging configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE = LOGS_DIR / 'hexdetector.log'

# Console logging
CONSOLE_LOG = True
CONSOLE_LOG_LEVEL = 'INFO'

# File logging
FILE_LOG = True
FILE_LOG_LEVEL = 'DEBUG'
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ==================== VISUALIZATION SETTINGS ====================
# Plot settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_SIZE = (12, 8)
PLOT_DPI = 100
DEFAULT_COLORMAP = 'viridis'

# Figure formats to save
SAVE_FORMATS = ['png', 'pdf']

# Confusion matrix settings
CM_NORMALIZE = True
CM_CMAP = 'Blues'

# Feature importance settings
TOP_N_FEATURES = 20

# ==================== PERFORMANCE MONITORING ====================
# Performance tracking
TRACK_MEMORY = True
TRACK_TIME = True
PROFILE_CODE = False

# Resource limits
MAX_CPU_PERCENT = 90
MAX_MEMORY_PERCENT = 85

# ==================== OUTPUT SETTINGS ====================
# Output file formats
OUTPUT_FORMATS = ['csv', 'xlsx', 'json']
DEFAULT_OUTPUT_FORMAT = 'csv'

# Excel settings
EXCEL_ENGINE = 'openpyxl'
EXCEL_SHEET_NAME = 'Results'

# Report generation
GENERATE_REPORT = True
REPORT_FORMAT = 'html'  # 'html', 'pdf', 'markdown'

# ==================== COLUMN MAPPINGS ====================
# Standard column names for network flows
COLUMN_MAPPINGS = {
    'ts': 'timestamp',
    'dur': 'duration',
    'proto': 'protocol',
    'orig_ip': 'src_ip',
    'orig_port': 'src_port',
    'resp_ip': 'dst_ip',
    'resp_port': 'dst_port',
    'orig_bytes': 'src_bytes',
    'resp_bytes': 'dst_bytes',
    'orig_pkts': 'src_packets',
    'resp_pkts': 'dst_packets'
}

# ==================== VALIDATION ====================
# Configuration validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if IoT23 directories exist
    required_dirs = [
        ('IOT23_SCENARIOS_DIR', IOT23_SCENARIOS_DIR),
        ('IOT23_ATTACKS_DIR', IOT23_ATTACKS_DIR),
        ('IOT23_DATA_DIR', IOT23_DATA_DIR)
    ]
    
    for name, path in required_dirs:
        if not os.path.exists(path):
            errors.append(f"{name} does not exist: {path}")
    
    # Check model types
    if DEFAULT_MODEL_TYPE not in MODEL_TYPES:
        errors.append(f"Invalid DEFAULT_MODEL_TYPE: {DEFAULT_MODEL_TYPE}")
    
    # Check test size
    if not 0 < TEST_SIZE < 1:
        errors.append(f"TEST_SIZE must be between 0 and 1, got: {TEST_SIZE}")
    
    return errors

# ==================== HELPER FUNCTIONS ====================
def get_model_params(model_type):
    """Get hyperparameters for a specific model type"""
    return MODEL_PARAMS.get(model_type, {})

def get_output_path(experiment_name, file_type='csv'):
    """Generate output path for experiment results"""
    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_model_path(model_name):
    """Generate path for saving/loading models"""
    return MODELS_DIR / f"{model_name}.pkl"