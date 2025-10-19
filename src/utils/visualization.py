"""
Visualization Module for HexDetector

Provides comprehensive visualization functions for model evaluation,
feature analysis, and results presentation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import additional visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False

try:
    import scikitplot as skplt
    SKPLT_AVAILABLE = True
except:
    SKPLT_AVAILABLE = False

# Try to import settings
try:
    from ..config import settings
    PLOT_STYLE = settings.PLOT_STYLE
    PLOT_SIZE = settings.PLOT_SIZE
    PLOT_DPI = settings.PLOT_DPI
    OUTPUT_DIR = settings.OUTPUT_DIR
except:
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    PLOT_SIZE = (12, 8)
    PLOT_DPI = 100
    OUTPUT_DIR = Path('output')

# Set style
try:
    plt.style.use(PLOT_STYLE)
except:
    sns.set_style('darkgrid')


def plot_results(results, save_path=None):
    """
    Plot model training results including accuracy and loss over epochs.
    
    Parameters:
    results (dict): Dictionary containing epoch, accuracy, and loss data
    save_path (str): Optional path to save the plot
    """
    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    
    if 'epoch' in results and 'accuracy' in results:
        plt.plot(results['epoch'], results['accuracy'], 
                label='Accuracy', color='blue', marker='o', linewidth=2)
    
    if 'epoch' in results and 'loss' in results:
        plt.plot(results['epoch'], results['loss'], 
                label='Loss', color='red', marker='s', linewidth=2)
    
    plt.title('Model Performance Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def plot_feature_importance(importances, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    importances (array): Feature importance values
    feature_names (list): Names of features
    top_n (int): Number of top features to display
    save_path (str): Optional path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_features = [feature_names[i] for i in indices]
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)), dpi=PLOT_DPI)
    
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = ax.barh(range(top_n), top_importances, color=colors, alpha=0.8)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest importance on top
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names, normalize=True, save_path=None):
    """
    Plot confusion matrix with customizable normalization.
    
    Parameters:
    cm (array): Confusion matrix
    class_names (list): Names of classes
    normalize (bool): Whether to normalize the confusion matrix
    save_path (str): Optional path to save the plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """
    Plot ROC curve with AUC score.
    
    Parameters:
    fpr (array): False positive rates
    tpr (array): True positive rates
    roc_auc (float): AUC score
    save_path (str): Optional path to save the plot
    """
    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
             fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(precision, recall, average_precision, save_path=None):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    precision (array): Precision values
    recall (array): Recall values
    average_precision (float): Average precision score
    save_path (str): Optional path to save the plot
    """
    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    
    plt.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AP = {average_precision:.3f})')
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def plot_model_comparison(model_results, metric='accuracy', save_path=None):
    """
    Plot comparison of multiple models on a specific metric.
    
    Parameters:
    model_results (dict): Dictionary with model names as keys and metrics as values
    metric (str): Metric to compare
    save_path (str): Optional path to save the plot
    """
    models = list(model_results.keys())
    values = [model_results[model].get(metric, 0) for model in models]
    
    plt.figure(figsize=(12, 6), dpi=PLOT_DPI)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
    plt.title(f'Model Comparison: {metric.capitalize()}', 
             fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def plot_class_distribution(y, class_names=None, save_path=None):
    """
    Plot distribution of classes in the dataset.
    
    Parameters:
    y (array): Target labels
    class_names (list): Optional class names
    save_path (str): Optional path to save the plot
    """
    unique, counts = np.unique(y, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    plt.figure(figsize=(10, 6), dpi=PLOT_DPI)
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique)))
    bars = plt.bar(class_names, counts, color=colors, alpha=0.8, edgecolor='black')
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels and percentages
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def plot_training_history(history, metrics=['accuracy', 'loss'], save_path=None):
    """
    Plot training and validation metrics over epochs.
    
    Parameters:
    history (dict): Training history with train and validation metrics
    metrics (list): List of metrics to plot
    save_path (str): Optional path to save the plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5), dpi=PLOT_DPI)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if f'train_{metric}' in history and f'val_{metric}' in history:
            ax.plot(history['epoch'], history[f'train_{metric}'], 
                   label=f'Train {metric}', marker='o', linewidth=2)
            ax.plot(history['epoch'], history[f'val_{metric}'], 
                   label=f'Validation {metric}', marker='s', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} Over Time', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def plot_learning_curve(train_sizes, train_scores, val_scores, save_path=None):
    """
    Plot learning curve showing model performance vs training set size.
    
    Parameters:
    train_sizes (array): Sizes of training sets
    train_scores (array): Training scores
    val_scores (array): Validation scores
    save_path (str): Optional path to save the plot
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
    
    plt.plot(train_sizes, train_mean, label='Training score', 
            color='blue', marker='o', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, label='Cross-validation score',
            color='red', marker='s', linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Learning Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


def create_dashboard(results, save_path=None):
    """
    Create a comprehensive dashboard with multiple subplots.
    
    Parameters:
    results (dict): Dictionary containing all necessary data for plotting
    save_path (str): Optional path to save the plot
    """
    fig = plt.figure(figsize=(16, 12), dpi=PLOT_DPI)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Confusion Matrix
    if 'confusion_matrix' in results and 'class_names' in results:
        ax1 = fig.add_subplot(gs[0, 0])
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=results['class_names'],
                   yticklabels=results['class_names'])
        ax1.set_title('Confusion Matrix', fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
    
    # Plot 2: Feature Importance
    if 'feature_importance' in results:
        ax2 = fig.add_subplot(gs[0, 1])
        top_n = 10
        indices = np.argsort(results['feature_importance'])[-top_n:]
        ax2.barh(range(top_n), results['feature_importance'][indices])
        ax2.set_yticks(range(top_n))
        if 'feature_names' in results:
            ax2.set_yticklabels([results['feature_names'][i] for i in indices])
        ax2.set_title('Top 10 Feature Importances', fontweight='bold')
        ax2.set_xlabel('Importance')
    
    # Plot 3: ROC Curve
    if 'fpr' in results and 'tpr' in results:
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(results['fpr'], results['tpr'], linewidth=2)
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax3.set_title('ROC Curve', fontweight='bold')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.grid(alpha=0.3)
    
    # Plot 4: Metrics Comparison
    if 'metrics' in results:
        ax4 = fig.add_subplot(gs[1, 1])
        metrics = results['metrics']
        ax4.bar(metrics.keys(), metrics.values(), color='skyblue', alpha=0.8)
        ax4.set_title('Performance Metrics', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim([0, 1])
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 5: Class Distribution
    if 'class_distribution' in results:
        ax5 = fig.add_subplot(gs[2, :])
        dist = results['class_distribution']
        ax5.bar(dist.keys(), dist.values(), color='lightgreen', alpha=0.8)
        ax5.set_title('Class Distribution', fontweight='bold')
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Count')
        ax5.grid(axis='y', alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('HexDetector Model Evaluation Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()


# Utility function to save all plots for an experiment
def save_experiment_plots(results, experiment_name):
    """
    Save all plots for a specific experiment.
    
    Parameters:
    results (dict): All results and data for plotting
    experiment_name (str): Name of the experiment
    """
    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save individual plots
    if 'confusion_matrix' in results:
        plot_confusion_matrix(
            results['confusion_matrix'],
            results.get('class_names', []),
            save_path=output_dir / 'confusion_matrix.png'
        )
    
    if 'feature_importance' in results:
        plot_feature_importance(
            results['feature_importance'],
            results.get('feature_names', []),
            save_path=output_dir / 'feature_importance.png'
        )
    
    if 'fpr' in results and 'tpr' in results:
        plot_roc_curve(
            results['fpr'],
            results['tpr'],
            results.get('roc_auc', 0),
            save_path=output_dir / 'roc_curve.png'
        )
    
    # Create dashboard
    create_dashboard(results, save_path=output_dir / 'dashboard.png')
    
    print(f"All plots saved to: {output_dir}")