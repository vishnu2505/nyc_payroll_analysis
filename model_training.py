"""
model_training.py
Machine learning model training and evaluation functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, f1_score, precision_score, recall_score,
                            confusion_matrix, classification_report)
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
import joblib
import config


def split_data(X, y, test_size=None, random_state=None, stratify=None):
    """
    Split data into training and testing sets
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        test_size (float): Proportion of test set
        random_state (int): Random seed
        stratify (Series): Stratification column for classification
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    print(f" Data split: {len(X_train):,} train, {len(X_test):,} test")
    
    return X_train, X_test, y_train, y_test


def train_xgboost_regressor(X_train, y_train, params=None):
    """
    Train XGBoost regression model for salary prediction
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        params (dict): Model hyperparameters
        
    Returns:
        XGBRegressor: Trained model
    """
    if params is None:
        params = config.XGBOOST_PARAMS
    
    print("\n" + "="*80)
    print("TRAINING XGBOOST REGRESSOR")
    print("="*80)
    print(f"Configuration: {params}")
    
    model = XGBRegressor(**params)
    
    print("\n Training in progress...")
    model.fit(X_train, y_train)
    print(" Training complete!")
    
    return model


def train_random_forest_classifier(X_train, y_train, params=None):
    """
    Train Random Forest classifier for overtime risk prediction
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        params (dict): Model hyperparameters
        
    Returns:
        RandomForestClassifier: Trained model
    """
    if params is None:
        params = config.RANDOM_FOREST_PARAMS
    
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*80)
    print(f"Configuration: {params}")
    
    model = RandomForestClassifier(**params)
    
    print("\n Training in progress...")
    model.fit(X_train, y_train)
    print(" Training complete!")
    
    return model


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate regression model performance
    
    Args:
        model: Trained regression model
        X_train, X_test: Feature matrices
        y_train, y_test: Target variables
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("\n" + "="*80)
    print("REGRESSION MODEL EVALUATION")
    print("="*80)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    # Display results
    print(f"\n Performance Metrics:")
    print(f"  Training R²:     {metrics['train_r2']:.4f}")
    print(f"  Test R²:         {metrics['test_r2']:.4f}")
    print(f"  Test RMSE:       ${metrics['test_rmse']:,.2f}")
    print(f"  Test MAE:        ${metrics['test_mae']:,.2f}")
    
    # Interpretation
    print(f"\n Interpretation:")
    print(f"  • Model explains {metrics['test_r2']*100:.2f}% of salary variance")
    print(f"  • Average prediction error: ${metrics['test_mae']:,.0f}")
    
    # Check overfitting
    gap = metrics['train_r2'] - metrics['test_r2']
    if gap > 0.05:
        print(f"    Overfitting detected (gap: {gap*100:.2f}%)")
    else:
        print(f"  Good generalization (gap: {gap*100:.2f}%)")
    
    # Sample predictions
    print(f"\n Sample Predictions:")
    sample_df = pd.DataFrame({
        'Actual': y_test.head(10).values,
        'Predicted': y_pred_test[:10],
        'Error': y_test.head(10).values - y_pred_test[:10]
    })
    sample_df['Error_%'] = (sample_df['Error'].abs() / sample_df['Actual'] * 100)
    print(sample_df.to_string(index=False))
    
    # Check if model meets minimum requirements
    if metrics['test_r2'] >= config.MIN_ACCEPTABLE_R2:
        print(f"\n Model PASSED: R² ≥ {config.MIN_ACCEPTABLE_R2}")
    else:
        print(f"\n Model FAILED: R² < {config.MIN_ACCEPTABLE_R2}")
    
    return metrics


def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate classification model performance
    
    Args:
        model: Trained classification model
        X_train, X_test: Feature matrices
        y_train, y_test: Target variables
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("\n" + "="*80)
    print("CLASSIFICATION MODEL EVALUATION")
    print("="*80)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0)
    }
    
    # Display results
    print(f"\n Performance Metrics:")
    print(f"  Training Accuracy:   {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:       {metrics['test_accuracy']:.4f}")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F1 Score:            {metrics['f1']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n Confusion Matrix:")
    print(f"                     Predicted Not At Risk | Predicted At Risk")
    print(f"  Actual Not At Risk:    {cm[0,0]:8,}         |    {cm[0,1]:6,}")
    print(f"  Actual At Risk:        {cm[1,0]:8,}         |    {cm[1,1]:6,}")
    
    # Classification Report
    print(f"\n Detailed Classification Report:")
    print(classification_report(y_test, y_pred_test, 
                              target_names=['Not At Risk', 'At Risk'],
                              digits=4))
    
    # Business interpretation
    print(f"\n Business Impact:")
    print(f"  • Of employees flagged, {metrics['precision']*100:.1f}% are truly at risk")
    print(f"  • Model identifies {metrics['recall']*100:.1f}% of actual at-risk employees")
    
    # Check if model meets minimum requirements
    if metrics['recall'] >= config.MIN_ACCEPTABLE_RECALL:
        print(f"\n Model PASSED: Recall ≥ {config.MIN_ACCEPTABLE_RECALL}")
    else:
        print(f"\n Model FAILED: Recall < {config.MIN_ACCEPTABLE_RECALL}")
    
    return metrics


def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract and display feature importance
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
        
    Returns:
        DataFrame: Feature importance dataframe
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n Top {top_n} Most Important Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        feature_name = row['Feature'][:50]  # Truncate long names
        print(f"  {feature_name:50s} : {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")
    
    # Categorize importance
    print(f"\n Feature Category Importance:")
    
    categories = {
        'Years of Service': importance_df[importance_df['Feature'] == 'Years of Service']['Importance'].sum(),
        'Base Salary': importance_df[importance_df['Feature'] == 'Base Salary']['Importance'].sum(),
        'Fiscal Year': importance_df[importance_df['Feature'].str.contains('Fiscal Year', na=False)]['Importance'].sum(),
        'Agency': importance_df[importance_df['Feature'].str.contains('Agency Name', na=False)]['Importance'].sum(),
        'Borough': importance_df[importance_df['Feature'].str.contains('Borough', na=False)]['Importance'].sum(),
        'Job Title': importance_df[importance_df['Feature'].str.contains('Title Description', na=False)]['Importance'].sum(),
    }
    
    for category, importance in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        if importance > 0:
            print(f"  {category:20s} : {importance:.4f} ({importance*100:.2f}%)")
    
    return importance_df


def save_model(model, filepath):
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"✓ Model saved to: {filepath}")


def load_model(filepath):
    """
    Load trained model from disk
    
    Args:
        filepath (str): Path to saved model
        
    Returns:
        Loaded model object
    """
    model = joblib.load(filepath)
    print(f" Model loaded from: {filepath}")
    return model


def export_predictions(y_test, y_pred, filepath, probabilities=None):
    """
    Export predictions to CSV
    
    Args:
        y_test (Series): Actual values
        y_pred (array): Predicted values
        filepath (str): Output file path
        probabilities (array): Prediction probabilities (for classification)
    """
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    
    if probabilities is not None:
        results_df['Probability'] = probabilities
    
    results_df['Error'] = results_df['Actual'] - results_df['Predicted']
    
    results_df.to_csv(filepath, index=False)
    print(f" Predictions saved to: {filepath}")


def train_and_evaluate_salary_model(X_train, X_test, y_train, y_test):
    """
    Complete pipeline for Model 1: Salary Prediction
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target variables
        
    Returns:
        tuple: (trained_model, metrics_dict, importance_df)
    """
    print("\n" + "="*80)
    print("MODEL 1: SALARY PREDICTION PIPELINE")
    print("="*80)
    
    # Train model
    model = train_xgboost_regressor(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test)
    
    # Feature importance
    importance_df = get_feature_importance(model, X_train.columns)
    
    # Export predictions
    y_pred = model.predict(X_test)
    export_predictions(
        y_test, 
        y_pred, 
        f"{config.OUTPUT_DIR}{config.SALARY_PREDICTIONS_FILE}"
    )
    
    # Export feature importance
    importance_df.to_csv(
        f"{config.OUTPUT_DIR}{config.SALARY_FEATURE_IMPORTANCE_FILE}",
        index=False
    )
    
    # Save model
    save_model(model, config.XGBOOST_MODEL_PATH)
    
    return model, metrics, importance_df


def train_and_evaluate_risk_model(X_train, X_test, y_train, y_test):
    """
    Complete pipeline for Model 2: Overtime Risk Classification
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target variables
        
    Returns:
        tuple: (trained_model, metrics_dict, importance_df)
    """
    print("\n" + "="*80)
    print("MODEL 2: OVERTIME RISK CLASSIFICATION PIPELINE")
    print("="*80)
    
    # Train model
    model = train_random_forest_classifier(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)
    
    # Feature importance
    importance_df = get_feature_importance(model, X_train.columns)
    
    # Export predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    export_predictions(
        y_test, 
        y_pred, 
        f"{config.OUTPUT_DIR}{config.RISK_PREDICTIONS_FILE}",
        probabilities=y_proba
    )
    
    # Export feature importance
    importance_df.to_csv(
        f"{config.OUTPUT_DIR}{config.RISK_FEATURE_IMPORTANCE_FILE}",
        index=False
    )
    
    # Save model
    save_model(model, config.RF_MODEL_PATH)
    
    return model, metrics, importance_df