"""
Prediction Module for HexDetector

Provides prediction functionality for trained models including batch predictions,
real-time predictions, and probability estimates.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.logger import Logger
except:
    class Logger:
        def log_info(self, msg): print(f"[INFO] {msg}")
        def log_error(self, msg, e=None): print(f"[ERROR] {msg}")
        def log_warning(self, msg): print(f"[WARN] {msg}")


def make_predictions(model, data, return_proba=False):
    """
    Generate predictions based on the trained model.

    Parameters:
    model: The trained model to use for predictions
    data (DataFrame): The input data for which predictions are to be made
    return_proba (bool): Whether to return probability estimates

    Returns:
    predictions (array or tuple): Predicted values (and probabilities if requested)
    """
    logger = Logger()
    logger.log_info(f"Making predictions on {len(data)} samples")
    
    try:
        # Prepare data
        X = prepare_prediction_data(data)
        
        # Make predictions
        predictions = model.predict(X)
        logger.log_info(f"Generated {len(predictions)} predictions")
        
        if return_proba:
            try:
                probabilities = model.predict_proba(X)
                logger.log_info("Generated probability estimates")
                return predictions, probabilities
            except Exception as e:
                logger.log_warning(f"Could not generate probabilities: {str(e)}")
                return predictions
        
        return predictions
        
    except Exception as e:
        logger.log_error(f"Error making predictions: {str(e)}", e)
        return None


def prepare_prediction_data(data):
    """
    Prepare data for prediction by handling missing values and non-numeric columns.
    
    Parameters:
    data (DataFrame): Input data
    
    Returns:
    DataFrame: Prepared data
    """
    logger = Logger()
    
    # Create a copy
    X = data.copy()
    
    # Remove target columns if present
    target_cols = ['target', 'label', 'class', 'Label', 'Target']
    for col in target_cols:
        if col in X.columns:
            X = X.drop(col, axis=1)
            logger.log_info(f"Removed target column: {col}")
    
    # Handle non-numeric columns
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        logger.log_info(f"Dropping {len(non_numeric_cols)} non-numeric columns")
        X = X.drop(columns=non_numeric_cols)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values
    if X.isnull().any().any():
        logger.log_info("Filling missing values with column means")
        X = X.fillna(X.mean())
    
    return X


def predict_batch(model, data, batch_size=1000):
    """
    Make predictions in batches for large datasets.
    
    Parameters:
    model: Trained model
    data (DataFrame): Input data
    batch_size (int): Size of each batch
    
    Returns:
    array: All predictions
    """
    logger = Logger()
    logger.log_info(f"Making batch predictions with batch size {batch_size}")
    
    n_samples = len(data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_data = data.iloc[start_idx:end_idx]
        batch_predictions = make_predictions(model, batch_data)
        
        if batch_predictions is not None:
            all_predictions.extend(batch_predictions)
        
        if (i + 1) % 10 == 0:
            logger.log_info(f"Processed {i + 1}/{n_batches} batches")
    
    logger.log_info(f"Completed batch predictions: {len(all_predictions)} total")
    return np.array(all_predictions)


def predict_with_confidence(model, data, confidence_threshold=0.8):
    """
    Make predictions with confidence filtering.
    
    Parameters:
    model: Trained model
    data (DataFrame): Input data
    confidence_threshold (float): Minimum confidence level
    
    Returns:
    tuple: (predictions, confidences, high_confidence_mask)
    """
    logger = Logger()
    logger.log_info(f"Making predictions with confidence threshold {confidence_threshold}")
    
    try:
        predictions, probabilities = make_predictions(model, data, return_proba=True)
        
        # Get max probability for each prediction
        confidences = np.max(probabilities, axis=1)
        
        # Identify high confidence predictions
        high_confidence_mask = confidences >= confidence_threshold
        
        n_high_confidence = np.sum(high_confidence_mask)
        logger.log_info(f"{n_high_confidence}/{len(predictions)} predictions above confidence threshold")
        
        return predictions, confidences, high_confidence_mask
        
    except Exception as e:
        logger.log_error(f"Error in confidence-based prediction: {str(e)}", e)
        return None, None, None


def load_model_and_predict(model_path, data):
    """
    Load a saved model and make predictions.
    
    Parameters:
    model_path (str): Path to saved model file
    data (DataFrame): Input data
    
    Returns:
    array: Predictions
    """
    logger = Logger()
    logger.log_info(f"Loading model from {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.log_info("Model loaded successfully")
        
        predictions = make_predictions(model, data)
        return predictions
        
    except Exception as e:
        logger.log_error(f"Error loading model or making predictions: {str(e)}", e)
        return None


def predict_top_k(model, data, k=3):
    """
    Get top K predictions for each sample.
    
    Parameters:
    model: Trained model
    data (DataFrame): Input data
    k (int): Number of top predictions to return
    
    Returns:
    tuple: (top_k_classes, top_k_probabilities)
    """
    logger = Logger()
    logger.log_info(f"Getting top {k} predictions")
    
    try:
        _, probabilities = make_predictions(model, data, return_proba=True)
        
        # Get indices of top k predictions
        top_k_indices = np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]
        
        # Get corresponding probabilities
        top_k_probabilities = np.take_along_axis(
            probabilities, 
            top_k_indices, 
            axis=1
        )
        
        return top_k_indices, top_k_probabilities
        
    except Exception as e:
        logger.log_error(f"Error getting top K predictions: {str(e)}", e)
        return None, None


def predict_and_explain(model, data, feature_names=None):
    """
    Make predictions and provide explanations (for models that support it).
    
    Parameters:
    model: Trained model
    data (DataFrame): Input data
    feature_names (list): Names of features
    
    Returns:
    dict: Predictions with explanations
    """
    logger = Logger()
    logger.log_info("Making predictions with explanations")
    
    predictions = make_predictions(model, data)
    
    results = {
        'predictions': predictions
    }
    
    # Add feature importances if available
    if hasattr(model, 'feature_importances_'):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importances'] = importance_df
        logger.log_info("Added feature importances to results")
    
    return results


def predict_anomaly_scores(model, data):
    """
    Calculate anomaly scores for predictions (useful for outlier detection).
    
    Parameters:
    model: Trained model
    data (DataFrame): Input data
    
    Returns:
    array: Anomaly scores
    """
    logger = Logger()
    logger.log_info("Calculating anomaly scores")
    
    try:
        # Get probability predictions
        _, probabilities = make_predictions(model, data, return_proba=True)
        
        # Calculate entropy as anomaly score
        # Higher entropy = more uncertain = more anomalous
        epsilon = 1e-10  # To avoid log(0)
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        
        # Normalize to [0, 1]
        max_entropy = np.log(probabilities.shape[1])
        normalized_scores = entropy / max_entropy
        
        logger.log_info(f"Calculated anomaly scores (mean: {normalized_scores.mean():.4f})")
        
        return normalized_scores
        
    except Exception as e:
        logger.log_warning(f"Could not calculate anomaly scores: {str(e)}")
        return None


def save_predictions(predictions, output_path, include_data=None):
    """
    Save predictions to a file.
    
    Parameters:
    predictions (array): Prediction results
    output_path (str): Path to save predictions
    include_data (DataFrame): Optional data to include with predictions
    
    Returns:
    bool: Success status
    """
    logger = Logger()
    logger.log_info(f"Saving predictions to {output_path}")
    
    try:
        # Create DataFrame
        if include_data is not None:
            result_df = include_data.copy()
            result_df['prediction'] = predictions
        else:
            result_df = pd.DataFrame({'prediction': predictions})
        
        # Save based on file extension
        file_extension = Path(output_path).suffix.lower().lstrip('.')
        
        if file_extension == 'csv':
            result_df.to_csv(output_path, index=False)
        elif file_extension == 'parquet':
            result_df.to_parquet(output_path, index=False)
        elif file_extension == 'json':
            result_df.to_json(output_path, orient='records', lines=True)
        else:
            logger.log_warning(f"Unsupported format, saving as CSV")
            result_df.to_csv(output_path + '.csv', index=False)
        
        logger.log_info(f"Successfully saved {len(predictions)} predictions")
        return True
        
    except Exception as e:
        logger.log_error(f"Error saving predictions: {str(e)}", e)
        return False