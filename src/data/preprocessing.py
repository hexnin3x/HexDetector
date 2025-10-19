"""
Data Preprocessing Module for HexDetector

Handles cleaning, transformation, and preparation of network traffic data
for machine learning models.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.logger import Logger
except:
    class Logger:
        def log_info(self, msg): print(f"[INFO] {msg}")
        def log_error(self, msg, e=None): print(f"[ERROR] {msg}")
        def log_warning(self, msg): print(f"[WARN] {msg}")


def preprocess_data(raw_data):
    """
    Preprocess raw network traffic data.
    
    Parameters:
    raw_data (DataFrame): Raw network traffic data
    
    Returns:
    DataFrame: Preprocessed data
    """
    logger = Logger()
    logger.log_info("Starting data preprocessing")
    logger.log_info(f"Input data shape: {raw_data.shape}")
    
    # Create a copy to avoid modifying the original data
    data = raw_data.copy()
    
    # Step 1: Handle missing values
    data = handle_missing_values(data)
    
    # Step 2: Remove duplicate rows
    data = remove_duplicates(data)
    
    # Step 3: Handle infinite values
    data = handle_infinite_values(data)
    
    # Step 4: Convert data types
    data = convert_data_types(data)
    
    # Step 5: Handle categorical variables
    data = encode_categorical_variables(data)
    
    # Step 6: Normalize/standardize numeric features
    data = normalize_features(data)
    
    # Step 7: Remove low variance features
    data = remove_low_variance_features(data)
    
    # Step 8: Handle outliers (optional, commented out by default)
    # data = handle_outliers(data)
    
    logger.log_info(f"Preprocessing complete. Output shape: {data.shape}")
    logger.log_info(f"Removed {raw_data.shape[0] - data.shape[0]} rows")
    logger.log_info(f"Removed {raw_data.shape[1] - data.shape[1]} columns")
    
    return data


def handle_missing_values(data, strategy='median', threshold=0.7):
    """
    Handle missing values in the dataset.
    
    Parameters:
    data (DataFrame): Input data
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'drop')
    threshold (float): Threshold for dropping columns with too many missing values
    
    Returns:
    DataFrame: Data with missing values handled
    """
    logger = Logger()
    
    # Check for missing values
    missing_count = data.isnull().sum()
    total_missing = missing_count.sum()
    
    if total_missing == 0:
        logger.log_info("No missing values found")
        return data
    
    logger.log_info(f"Found {total_missing} missing values across {(missing_count > 0).sum()} columns")
    
    # Drop columns with too many missing values
    missing_ratio = data.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if cols_to_drop:
        logger.log_info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        data = data.drop(columns=cols_to_drop)
    
    # Handle remaining missing values
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    # For numeric columns
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                data[col] = data[col].fillna(data[col].mean())
            elif strategy == 'median':
                data[col] = data[col].fillna(data[col].median())
            elif strategy == 'drop':
                data = data.dropna(subset=[col])
    
    # For categorical columns
    for col in categorical_cols:
        if data[col].isnull().any():
            if strategy in ['mode', 'median', 'mean']:
                mode_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col] = data[col].fillna(mode_value)
            elif strategy == 'drop':
                data = data.dropna(subset=[col])
    
    logger.log_info(f"Missing values handled using '{strategy}' strategy")
    
    return data


def remove_duplicates(data):
    """Remove duplicate rows from the dataset"""
    logger = Logger()
    
    n_before = len(data)
    data = data.drop_duplicates()
    n_after = len(data)
    
    n_removed = n_before - n_after
    if n_removed > 0:
        logger.log_info(f"Removed {n_removed} duplicate rows ({n_removed/n_before*100:.2f}%)")
    else:
        logger.log_info("No duplicate rows found")
    
    return data


def handle_infinite_values(data):
    """Replace infinite values with NaN and then handle them"""
    logger = Logger()
    
    # Count infinite values
    inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
    
    if inf_count > 0:
        logger.log_info(f"Found {inf_count} infinite values, replacing with NaN")
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
    else:
        logger.log_info("No infinite values found")
    
    return data


def convert_data_types(data):
    """Convert data types for optimal memory usage and processing"""
    logger = Logger()
    
    # Convert object columns that are actually numeric
    for col in data.select_dtypes(include=['object']).columns:
        try:
            data[col] = pd.to_numeric(data[col])
            logger.log_info(f"Converted column '{col}' to numeric")
        except:
            pass
    
    # Downcast numeric types for memory efficiency
    int_cols = data.select_dtypes(include=['int64']).columns
    for col in int_cols:
        data[col] = pd.to_numeric(data[col], downcast='integer')
    
    float_cols = data.select_dtypes(include=['float64']).columns
    for col in float_cols:
        data[col] = pd.to_numeric(data[col], downcast='float')
    
    logger.log_info("Data type conversion completed")
    
    return data


def encode_categorical_variables(data, keep_original=False):
    """
    Encode categorical variables using one-hot encoding or label encoding.
    
    Parameters:
    data (DataFrame): Input data
    keep_original (bool): Whether to keep original categorical columns
    
    Returns:
    DataFrame: Data with encoded categorical variables
    """
    logger = Logger()
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    # Exclude certain columns that should remain as strings
    exclude_cols = ['label', 'Label', 'target', 'Target', 'class', 'Class']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    if len(categorical_cols) == 0:
        logger.log_info("No categorical columns to encode")
        return data
    
    logger.log_info(f"Encoding {len(categorical_cols)} categorical columns")
    
    for col in categorical_cols:
        n_unique = data[col].nunique()
        
        # Use one-hot encoding for low cardinality (< 10 unique values)
        if n_unique < 10:
            logger.log_info(f"One-hot encoding '{col}' ({n_unique} unique values)")
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            
            if not keep_original:
                data = data.drop(col, axis=1)
        else:
            # Use label encoding for high cardinality
            logger.log_info(f"Label encoding '{col}' ({n_unique} unique values)")
            data[col + '_encoded'] = pd.factorize(data[col])[0]
            
            if not keep_original:
                data = data.drop(col, axis=1)
    
    return data


def normalize_features(data, method='standard', exclude_cols=None):
    """
    Normalize/standardize numeric features.
    
    Parameters:
    data (DataFrame): Input data
    method (str): Normalization method ('standard', 'minmax', 'robust', 'none')
    exclude_cols (list): Columns to exclude from normalization
    
    Returns:
    DataFrame: Data with normalized features
    """
    logger = Logger()
    
    if method == 'none':
        logger.log_info("Skipping feature normalization")
        return data
    
    if exclude_cols is None:
        exclude_cols = ['label', 'Label', 'target', 'Target', 'class', 'Class']
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(cols_to_normalize) == 0:
        logger.log_info("No columns to normalize")
        return data
    
    logger.log_info(f"Normalizing {len(cols_to_normalize)} numeric columns using '{method}' method")
    
    if method == 'standard':
        # Z-score normalization
        for col in cols_to_normalize:
            mean = data[col].mean()
            std = data[col].std()
            if std > 0:
                data[col] = (data[col] - mean) / std
    
    elif method == 'minmax':
        # Min-Max normalization
        for col in cols_to_normalize:
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val > min_val:
                data[col] = (data[col] - min_val) / (max_val - min_val)
    
    elif method == 'robust':
        # Robust normalization (using median and IQR)
        for col in cols_to_normalize:
            median = data[col].median()
            q75 = data[col].quantile(0.75)
            q25 = data[col].quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                data[col] = (data[col] - median) / iqr
    
    return data


def remove_low_variance_features(data, threshold=0.01):
    """
    Remove features with very low variance.
    
    Parameters:
    data (DataFrame): Input data
    threshold (float): Variance threshold
    
    Returns:
    DataFrame: Data with low variance features removed
    """
    logger = Logger()
    
    # Get numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Exclude target columns
    exclude_cols = ['label', 'Label', 'target', 'Target', 'class', 'Class']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate variance
    variances = data[numeric_cols].var()
    
    # Find low variance columns
    low_var_cols = variances[variances < threshold].index.tolist()
    
    if low_var_cols:
        logger.log_info(f"Removing {len(low_var_cols)} low variance features (threshold={threshold})")
        data = data.drop(columns=low_var_cols)
    else:
        logger.log_info("No low variance features to remove")
    
    return data


def handle_outliers(data, method='iqr', threshold=3):
    """
    Handle outliers in numeric features.
    
    Parameters:
    data (DataFrame): Input data
    method (str): Method for outlier detection ('iqr', 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    DataFrame: Data with outliers handled
    """
    logger = Logger()
    logger.log_info(f"Handling outliers using '{method}' method")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    exclude_cols = ['label', 'Label', 'target', 'Target', 'class', 'Class']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    n_before = len(data)
    
    for col in numeric_cols:
        if method == 'iqr':
            # IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap outliers instead of removing
            data[col] = data[col].clip(lower_bound, upper_bound)
        
        elif method == 'zscore':
            # Z-score method
            mean = data[col].mean()
            std = data[col].std()
            if std > 0:
                z_scores = np.abs((data[col] - mean) / std)
                # Cap outliers
                mask = z_scores > threshold
                if mask.any():
                    median = data[col].median()
                    data.loc[mask, col] = median
    
    n_after = len(data)
    logger.log_info(f"Outlier handling complete. Rows affected: {n_before - n_after}")
    
    return data


def balance_dataset(data, target_col='label', method='undersample'):
    """
    Balance the dataset by handling class imbalance.
    
    Parameters:
    data (DataFrame): Input data
    target_col (str): Name of target column
    method (str): Balancing method ('undersample', 'oversample', 'smote')
    
    Returns:
    DataFrame: Balanced data
    """
    logger = Logger()
    
    if target_col not in data.columns:
        logger.log_warning(f"Target column '{target_col}' not found")
        return data
    
    # Get class distribution
    class_counts = data[target_col].value_counts()
    logger.log_info(f"Original class distribution:\n{class_counts}")
    
    if method == 'undersample':
        # Undersample majority classes
        min_count = class_counts.min()
        balanced_dfs = []
        
        for class_label in class_counts.index:
            class_df = data[data[target_col] == class_label]
            sampled_df = class_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled_df)
        
        data = pd.concat(balanced_dfs, ignore_index=True)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.log_info(f"Applied undersampling. New shape: {data.shape}")
    
    elif method == 'oversample':
        # Oversample minority classes
        max_count = class_counts.max()
        balanced_dfs = []
        
        for class_label in class_counts.index:
            class_df = data[data[target_col] == class_label]
            sampled_df = class_df.sample(n=max_count, replace=True, random_state=42)
            balanced_dfs.append(sampled_df)
        
        data = pd.concat(balanced_dfs, ignore_index=True)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.log_info(f"Applied oversampling. New shape: {data.shape}")
    
    # Log new distribution
    new_class_counts = data[target_col].value_counts()
    logger.log_info(f"New class distribution:\n{new_class_counts}")
    
    return data