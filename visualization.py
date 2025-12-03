"""
visualization.py
Visualization and plotting functions for model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_feature_importance(importance_df, title, top_n=15, save_path=None):
    """
    Plot feature importance bar chart
    
    Args:
        importance_df (DataFrame): Feature importance dataframe
        title (str): Plot title
        top_n (int): Number of top features to show
        save_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = importance_df.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['Importance'], 
            color=config.COLORS['primary'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(top_features['Importance']):
        ax.text(v, i, f' {v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred, title, save_path=None):
    """
    Plot actual vs predicted values for regression
    
    Args:
        y_test (array): Actual values
        y_pred (array): Predicted values
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color=config.COLORS['primary'])
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    
    ax.set_xlabel('Actual Salary ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Salary ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nR² = {r2:.4f}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def plot_residuals(y_test, y_pred, title, save_path=None):
    """
    Plot residuals for regression model
    
    Args:
        y_test (array): Actual values
        y_pred (array): Predicted values
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residual scatter plot
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20, color=config.COLORS['secondary'])
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Salary ($)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residual histogram
    ax2.hist(residuals, bins=50, color=config.COLORS['secondary'], edgecolor='black')
    ax2.set_xlabel('Residuals ($)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_test, y_pred, labels, title, save_path=None):
    """
    Plot confusion matrix heatmap
    
    Args:
        y_test (array): Actual values
        y_pred (array): Predicted values
        labels (list): Class labels
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def plot_roc_curve(y_test, y_proba, title, save_path=None):
    """
    Plot ROC curve for binary classification
    
    Args:
        y_test (array): Actual binary labels
        y_proba (array): Predicted probabilities
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color=config.COLORS['primary'], lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def plot_model_comparison(metrics_dict, title, save_path=None):
    """
    Plot comparison of multiple metrics
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = ax.bar(metrics, values, color=config.COLORS['success'], 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def plot_learning_curve(model, X_train, y_train, title, cv=5, save_path=None):
    """
    Plot learning curve to diagnose bias/variance
    
    Args:
        model: Trained model
        X_train (DataFrame): Training features
        y_train (Series): Training target
        title (str): Plot title
        cv (int): Cross-validation folds
        save_path (str): Path to save the plot
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2' if hasattr(model, 'predict') else 'accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, train_mean, label='Training Score',
            color=config.COLORS['primary'], lw=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color=config.COLORS['primary'])
    
    ax.plot(train_sizes, val_mean, label='Validation Score',
            color=config.COLORS['secondary'], lw=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color=config.COLORS['secondary'])
    
    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def create_model1_visualizations(model, X_test, y_test, importance_df, save_dir=None):
    """
    Create all visualizations for Model 1 (Salary Prediction)
    
    Args:
        model: Trained model
        X_test (DataFrame): Test features
        y_test (Series): Test target
        importance_df (DataFrame): Feature importance
        save_dir (str): Directory to save plots
    """
    print("\n" + "="*80)
    print("GENERATING MODEL 1 VISUALIZATIONS")
    print("="*80)
    
    y_pred = model.predict(X_test)
    
    # 1. Feature Importance
    plot_feature_importance(
        importance_df,
        "Model 1: Salary Prediction - Feature Importance",
        save_path=f"{save_dir}model1_feature_importance.png" if save_dir else None
    )
    
    # 2. Actual vs Predicted
    plot_actual_vs_predicted(
        y_test, y_pred,
        "Model 1: Actual vs Predicted Salary",
        save_path=f"{save_dir}model1_actual_vs_predicted.png" if save_dir else None
    )
    
    # 3. Residuals
    plot_residuals(
        y_test, y_pred,
        "Model 1: Residual Analysis",
        save_path=f"{save_dir}model1_residuals.png" if save_dir else None
    )
    
    print("✓ Model 1 visualizations complete!")


def create_model2_visualizations(model, X_test, y_test, importance_df, save_dir=None):
    """
    Create all visualizations for Model 2 (Overtime Risk Classification)
    
    Args:
        model: Trained model
        X_test (DataFrame): Test features
        y_test (Series): Test target
        importance_df (DataFrame): Feature importance
        save_dir (str): Directory to save plots
    """
    print("\n" + "="*80)
    print("GENERATING MODEL 2 VISUALIZATIONS")
    print("="*80)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 1. Feature Importance
    plot_feature_importance(
        importance_df,
        "Model 2: Overtime Risk - Feature Importance",
        save_path=f"{save_dir}model2_feature_importance.png" if save_dir else None
    )
    
    # 2. Confusion Matrix
    plot_confusion_matrix(
        y_test, y_pred,
        labels=['Not At Risk', 'At Risk'],
        title="Model 2: Overtime Risk - Confusion Matrix",
        save_path=f"{save_dir}model2_confusion_matrix.png" if save_dir else None
    )
    
    # 3. ROC Curve
    plot_roc_curve(
        y_test, y_proba,
        "Model 2: Overtime Risk - ROC Curve",
        save_path=f"{save_dir}model2_roc_curve.png" if save_dir else None
    )
    
    print("✓ Model 2 visualizations complete!")