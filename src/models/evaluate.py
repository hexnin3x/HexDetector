"""
Model Evaluation Module for HexDetector

Provides comprehensive evaluation functionality for machine learning models
including various metrics, cross-validation, and performance analysis.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score

try:
    from utils.logger import Logger
except:
    class Logger:
        def log_info(self, msg): print(f"[INFO] {msg}")
        def log_error(self, msg, e=None): print(f"[ERROR] {msg}")
        def log_warning(self, msg): print(f"[WARN] {msg}")


def evaluate_model(model, test_data, metrics=None):
    """
    Evaluate the performance of the given model on the test data using specified metrics.

    Parameters:
    model: The trained model to evaluate
    test_data (DataFrame): The data on which to evaluate the model (must include target)
    metrics (list): List of metrics to use for evaluation

    Returns:
    dict: Dictionary containing the evaluation results for each metric
    """
    logger = Logger()
    logger.log_info("Evaluating model performance")
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
    
    # Find target column
    target_col = None
    for col in ['target', 'label', 'class', 'Label', 'Target']:
        if col in test_data.columns:
            target_col = col
            break
    
    if target_col is None:
        logger.log_error("No target column found in test data")
        return {}
    
    # Separate features and target
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]
    
    # Handle non-numeric columns
    non_numeric_cols = X_test.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        X_test = X_test.drop(columns=non_numeric_cols)
    
    # Handle infinite and missing values
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(X_test.mean())
    
    # Make predictions
    try:
        y_pred = model.predict(X_test)
        
        # Try to get probability predictions if available
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            y_pred_proba = None
    except Exception as e:
        logger.log_error(f"Error making predictions: {str(e)}", e)
        return {}
    
    # Calculate requested metrics
    results = {}
    
    for metric in metrics:
        try:
            if metric == 'accuracy':
                results['accuracy'] = calculate_accuracy(y_test, y_pred)
            elif metric == 'precision':
                results['precision'] = calculate_precision(y_test, y_pred)
            elif metric == 'recall':
                results['recall'] = calculate_recall(y_test, y_pred)
            elif metric == 'f1':
                results['f1'] = calculate_f1(y_test, y_pred)
            elif metric == 'confusion_matrix':
                results['confusion_matrix'] = calculate_confusion_matrix(y_test, y_pred)
            elif metric == 'classification_report':
                results['classification_report'] = calculate_classification_report(y_test, y_pred)
            elif metric == 'roc_auc' and y_pred_proba is not None:
                results['roc_auc'] = calculate_roc_auc(y_test, y_pred_proba)
            elif metric == 'roc_curve' and y_pred_proba is not None:
                results['fpr'], results['tpr'], results['roc_thresholds'] = calculate_roc_curve(y_test, y_pred_proba)
            elif metric == 'precision_recall_curve' and y_pred_proba is not None:
                results['precision_curve'], results['recall_curve'], results['pr_thresholds'] = calculate_pr_curve(y_test, y_pred_proba)
        except Exception as e:
            logger.log_warning(f"Could not calculate {metric}: {str(e)}")
    
    # Log results
    logger.log_info("Evaluation Results:")
    for key, value in results.items():
        if key not in ['confusion_matrix', 'classification_report', 'fpr', 'tpr', 
                       'precision_curve', 'recall_curve', 'roc_thresholds', 'pr_thresholds']:
            if isinstance(value, float):
                logger.log_info(f"  {key}: {value:.4f}")
            else:
                logger.log_info(f"  {key}: {value}")
    
    return results


def calculate_accuracy(y_test, y_pred):
    """Calculate accuracy score"""
    return accuracy_score(y_test, y_pred)


def calculate_precision(y_test, y_pred):
    """Calculate precision score"""
    n_classes = len(np.unique(y_test))
    average = 'binary' if n_classes == 2 else 'weighted'
    return precision_score(y_test, y_pred, average=average, zero_division=0)


def calculate_recall(y_test, y_pred):
    """Calculate recall score"""
    n_classes = len(np.unique(y_test))
    average = 'binary' if n_classes == 2 else 'weighted'
    return recall_score(y_test, y_pred, average=average, zero_division=0)


def calculate_f1(y_test, y_pred):
    """Calculate F1 score"""
    n_classes = len(np.unique(y_test))
    average = 'binary' if n_classes == 2 else 'weighted'
    return f1_score(y_test, y_pred, average=average, zero_division=0)


def calculate_confusion_matrix(y_test, y_pred):
    """Calculate confusion matrix"""
    return confusion_matrix(y_test, y_pred)


def calculate_classification_report(y_test, y_pred):
    """Generate classification report"""
    return classification_report(y_test, y_pred, zero_division=0)


def calculate_roc_auc(y_test, y_pred_proba):
    """Calculate ROC AUC score"""
    n_classes = len(np.unique(y_test))
    
    if n_classes == 2:
        # Binary classification
        return roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        # Multiclass classification
        return roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')


def calculate_roc_curve(y_test, y_pred_proba):
    """Calculate ROC curve"""
    n_classes = len(np.unique(y_test))
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        return fpr, tpr, thresholds
    else:
        # For multiclass, return None (would need one-vs-rest approach)
        return None, None, None


def calculate_pr_curve(y_test, y_pred_proba):
    """Calculate Precision-Recall curve"""
    n_classes = len(np.unique(y_test))
    
    if n_classes == 2:
        # Binary classification
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
        return precision, recall, thresholds
    else:
        # For multiclass, return None
        return None, None, None


def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    """
    Perform cross-validation on the model.
    
    Parameters:
    model: The model to evaluate
    X (DataFrame): Features
    y (Series): Target
    cv (int): Number of cross-validation folds
    scoring (str): Scoring metric
    
    Returns:
    dict: Cross-validation results
    """
    logger = Logger()
    logger.log_info(f"Performing {cv}-fold cross-validation")
    
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max()
        }
        
        logger.log_info(f"Cross-validation {scoring}:")
        logger.log_info(f"  Mean: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        logger.log_info(f"  Min:  {results['min_score']:.4f}")
        logger.log_info(f"  Max:  {results['max_score']:.4f}")
        
        return results
        
    except Exception as e:
        logger.log_error(f"Error during cross-validation: {str(e)}", e)
        return None


def evaluate_per_class(y_test, y_pred, class_names=None):
    """
    Evaluate model performance per class.
    
    Parameters:
    y_test: True labels
    y_pred: Predicted labels
    class_names (list): Names of classes
    
    Returns:
    DataFrame: Per-class metrics
    """
    logger = Logger()
    logger.log_info("Calculating per-class metrics")
    
    # Get unique classes
    classes = np.unique(y_test)
    
    if class_names is None:
        class_names = [f"Class_{c}" for c in classes]
    
    # Calculate metrics for each class
    results = []
    
    for i, cls in enumerate(classes):
        y_test_binary = (y_test == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        
        results.append({
            'class': class_names[i] if i < len(class_names) else f"Class_{cls}",
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(y_test == cls)
        })
    
    df = pd.DataFrame(results)
    
    logger.log_info("Per-class performance:")
    logger.log_info(f"\n{df.to_string()}")
    
    return df


def compare_models(models_results):
    """
    Compare multiple models and generate a comparison report.
    
    Parameters:
    models_results (dict): Dictionary with model names as keys and results as values
    
    Returns:
    DataFrame: Comparison table
    """
    logger = Logger()
    logger.log_info("Comparing models")
    
    comparison = []
    
    for model_name, results in models_results.items():
        comparison.append({
            'model': model_name,
            'accuracy': results.get('accuracy', 0),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'f1': results.get('f1', 0)
        })
    
    df = pd.DataFrame(comparison).sort_values('f1', ascending=False)
    
    logger.log_info("Model Comparison:")
    logger.log_info(f"\n{df.to_string()}")
    
    # Find best model
    best_model = df.iloc[0]['model']
    logger.log_info(f"\nBest model: {best_model} (F1: {df.iloc[0]['f1']:.4f})")
    
    return df


def generate_evaluation_report(model, test_data, model_name="Model"):
    """
    Generate a comprehensive evaluation report.
    
    Parameters:
    model: Trained model
    test_data (DataFrame): Test data with target column
    model_name (str): Name of the model
    
    Returns:
    dict: Complete evaluation report
    """
    logger = Logger()
    logger.log_info(f"Generating evaluation report for {model_name}")
    
    # Evaluate with all metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 
               'classification_report', 'roc_auc']
    
    results = evaluate_model(model, test_data, metrics)
    
    # Add model name
    results['model_name'] = model_name
    
    return results