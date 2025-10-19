"""
Model Training Module for HexDetector

Provides training functionality for multiple machine learning algorithms
including Random Forest, SVM, Naive Bayes, Logistic Regression, Decision Tree,
XGBoost, and Gradient Boosting.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from config import settings
    from utils.logger import Logger
except:
    settings = None
    
    class Logger:
        def log_info(self, msg): print(f"[INFO] {msg}")
        def log_error(self, msg, e=None): print(f"[ERROR] {msg}")
        def log_warning(self, msg): print(f"[WARN] {msg}")


def train_model(dataset, model_type='random_forest', test_size=0.2, random_state=42, handle_imbalance=False):
    """
    Trains a machine learning model using the provided dataset.
    
    Parameters:
    dataset (DataFrame): The dataset used for training the model
    model_type (str): Type of model ('random_forest', 'svm', 'naive_bayes', 
                      'logistic_regression', 'decision_tree', 'xgboost', 'gradient_boosting')
    test_size (float): Proportion of dataset to use for testing
    random_state (int): Random seed for reproducibility
    handle_imbalance (bool): Whether to handle class imbalance
    
    Returns:
    dict: Dictionary containing the trained model and evaluation metrics
    """
    logger = Logger()
    logger.log_info(f"Training {model_type} model")
    
    # Check if dataset is empty
    if dataset.empty:
        logger.log_error("Empty dataset provided for training")
        return None
    
    # Check if target column exists
    target_col = None
    for col in ['target', 'label', 'class', 'Label', 'Target']:
        if col in dataset.columns:
            target_col = col
            break
    
    if target_col is None:
        logger.log_error("No target column found in dataset. Expected 'target', 'label', or 'class'")
        return None
    
    logger.log_info(f"Using '{target_col}' as target column")
    
    # Prepare data
    X, y, X_train, X_test, y_train, y_test = prepare_data(
        dataset, target_col, test_size, random_state
    )
    
    if X_train is None:
        return None
    
    # Handle class imbalance if requested
    if handle_imbalance:
        X_train, y_train = handle_class_imbalance(X_train, y_train, logger)
    
    # Get model and hyperparameters
    model = get_model(model_type, random_state)
    
    if model is None:
        logger.log_error(f"Failed to initialize {model_type} model")
        return None
    
    # Train the model
    logger.log_info(f"Training model on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    try:
        model.fit(X_train, y_train)
        logger.log_info("Model training completed")
    except Exception as e:
        logger.log_error(f"Error during model training: {str(e)}", e)
        return None
    
    # Make predictions
    logger.log_info("Making predictions on test set")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, logger)
    
    # Get feature importance if available
    feature_importance = get_feature_importance(model, X.columns, logger)
    
    # Prepare results
    results = {
        'model': model,
        'model_type': model_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'features': X.columns.tolist(),
        'feature_importance': feature_importance,
        **metrics
    }
    
    return results


def prepare_data(dataset, target_col, test_size, random_state):
    """Prepare data for training"""
    logger = Logger()
    
    try:
        # Split features and target
        X = dataset.drop(target_col, axis=1)
        y = dataset[target_col]
        
        # Handle non-numeric columns in features
        non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(non_numeric_cols) > 0:
            logger.log_info(f"Dropping {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        if X.isnull().any().any():
            logger.log_info("Filling missing values with column means")
            X = X.fillna(X.mean())
        
        # Encode target if it's categorical
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            logger.log_info("Encoding categorical target variable")
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.log_info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.log_info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X, y, X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.log_error(f"Error preparing data: {str(e)}", e)
        return None, None, None, None, None, None


def get_model(model_type, random_state):
    """Get model instance based on type"""
    logger = Logger()
    
    # Get hyperparameters from settings if available
    if settings and hasattr(settings, 'MODEL_PARAMS'):
        params = settings.MODEL_PARAMS.get(model_type, {})
    else:
        params = {}
    
    # Override random_state
    if 'random_state' in params:
        params['random_state'] = random_state
    
    try:
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=random_state,
                n_jobs=params.get('n_jobs', -1)
            )
        
        elif model_type == 'svm':
            return SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                gamma=params.get('gamma', 'scale'),
                random_state=random_state,
                probability=True
            )
        
        elif model_type == 'naive_bayes':
            return GaussianNB(
                var_smoothing=params.get('var_smoothing', 1e-9)
            )
        
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=random_state,
                n_jobs=params.get('n_jobs', -1)
            )
        
        elif model_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=random_state
            )
        
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                logger.log_error("XGBoost is not installed. Install it with: pip install xgboost")
                return None
            return XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=random_state,
                n_jobs=params.get('n_jobs', -1),
                eval_metric='logloss'
            )
        
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                random_state=random_state
            )
        
        else:
            logger.log_error(f"Unsupported model type: {model_type}")
            return None
            
    except Exception as e:
        logger.log_error(f"Error creating model: {str(e)}", e)
        return None


def calculate_metrics(y_test, y_pred, logger):
    """Calculate evaluation metrics"""
    try:
        # Determine if binary or multiclass
        n_classes = len(np.unique(y_test))
        average_method = 'binary' if n_classes == 2 else 'weighted'
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.log_info(f"Model Accuracy:  {accuracy:.4f}")
        logger.log_info(f"Model Precision: {precision:.4f}")
        logger.log_info(f"Model Recall:    {recall:.4f}")
        logger.log_info(f"Model F1 Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
    except Exception as e:
        logger.log_error(f"Error calculating metrics: {str(e)}", e)
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'confusion_matrix': None
        }


def get_feature_importance(model, feature_names, logger):
    """Extract feature importance if available"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.log_info(f"Top 10 most important features:")
            for i, (feature, importance) in enumerate(zip(
                importance_df['feature'][:10], 
                importance_df['importance'][:10]
            )):
                logger.log_info(f"  {i+1}. {feature}: {importance:.4f}")
            
            return importance_df
        
        elif hasattr(model, 'coef_'):
            # For linear models
            importance = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            logger.log_info("Model does not support feature importance")
            return None
            
    except Exception as e:
        logger.log_warning(f"Could not extract feature importance: {str(e)}")
        return None


def handle_class_imbalance(X_train, y_train, logger):
    """Handle class imbalance using SMOTE or other techniques"""
    try:
        from imblearn.over_sampling import SMOTE
        
        logger.log_info("Handling class imbalance with SMOTE")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.log_info(f"Original class distribution: {np.bincount(y_train)}")
        logger.log_info(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
        
    except ImportError:
        logger.log_warning("imbalanced-learn not installed. Skipping class imbalance handling")
        return X_train, y_train
    except Exception as e:
        logger.log_warning(f"Error handling class imbalance: {str(e)}")
        return X_train, y_train


def train_multiple_models(dataset, model_types=None, test_size=0.2, random_state=42):
    """
    Train multiple models and compare their performance.
    
    Parameters:
    dataset (DataFrame): The dataset used for training
    model_types (list): List of model types to train
    test_size (float): Proportion of dataset to use for testing
    random_state (int): Random seed for reproducibility
    
    Returns:
    dict: Dictionary with model results for each model type
    """
    logger = Logger()
    logger.log_info("Training multiple models for comparison")
    
    if model_types is None:
        model_types = ['random_forest', 'svm', 'naive_bayes', 'logistic_regression', 'decision_tree']
    
    results = {}
    
    for model_type in model_types:
        logger.log_info(f"\n{'='*60}")
        logger.log_info(f"Training {model_type}")
        logger.log_info(f"{'='*60}")
        
        try:
            model_results = train_model(
                dataset, 
                model_type=model_type, 
                test_size=test_size, 
                random_state=random_state
            )
            
            if model_results:
                results[model_type] = model_results
                
        except Exception as e:
            logger.log_error(f"Error training {model_type}: {str(e)}", e)
    
    # Print comparison
    logger.log_info(f"\n{'='*60}")
    logger.log_info("Model Comparison")
    logger.log_info(f"{'='*60}")
    logger.log_info(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    logger.log_info(f"{'-'*60}")
    
    for model_type, result in results.items():
        logger.log_info(
            f"{model_type:<20} "
            f"{result['accuracy']:<12.4f} "
            f"{result['precision']:<12.4f} "
            f"{result['recall']:<12.4f} "
            f"{result['f1']:<12.4f}"
        )
    
    return results