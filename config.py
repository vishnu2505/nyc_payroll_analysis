"""
config.py
Configuration file for NYC Payroll Analysis Project
Contains all constants, file paths, and model hyperparameters
"""

# ============================================================================
# PROJECT METADATA
# ============================================================================
PROJECT_NAME = "NYC Municipal Payroll Analysis"
AUTHORS = ["Vishnu S", "Sharan Venkatesh"]
COURSE = "ALY6110 - Big Data and Management"
INSTITUTION = "Northeastern University"

# ============================================================================
# FILE PATHS
# ============================================================================
DATA_PATH = "/content/nyc_payroll_FINAL_FOR_TABLEAU.csv"
OUTPUT_DIR = "output/"
MODELS_DIR = "models/"
PLOTS_DIR = "plots/"

# Output file names
SALARY_PREDICTIONS_FILE = "salary_predictions.csv"
RISK_PREDICTIONS_FILE = "overtime_risk_predictions.csv"
SALARY_FEATURE_IMPORTANCE_FILE = "salary_feature_importance.csv"
RISK_FEATURE_IMPORTANCE_FILE = "risk_feature_importance.csv"

# Model save paths
XGBOOST_MODEL_PATH = "models/xgboost_salary_model.pkl"
RF_MODEL_PATH = "models/random_forest_risk_model.pkl"

# ============================================================================
# DATA PARAMETERS
# ============================================================================
# Fiscal years to analyze
FISCAL_YEARS = [2023, 2024, 2025]

# Reference date for calculating Years of Service
REFERENCE_DATE = "2025-06-30"

# Valid salary range (filter outliers)
MIN_SALARY = 10000
MAX_SALARY = 500000

# Valid tenure range
MIN_YEARS_SERVICE = 0
MAX_YEARS_SERVICE = 50

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
# Number of top categories to keep (reduce cardinality)
TOP_N_AGENCIES = 15
TOP_N_TITLES = 30

# Overtime risk thresholds (hours per year)
OT_LOW_THRESHOLD = 200
OT_MEDIUM_THRESHOLD = 600

# ============================================================================
# MODEL 1: XGBOOST SALARY PREDICTION PARAMETERS
# ============================================================================
# MODEL 1: XGBOOST SALARY PREDICTION PARAMETERS
XGBOOST_PARAMS = {
    'n_estimators': 300,          # Increased from 200
    'max_depth': 6,               # Reduced from 8 (less overfitting)
    'learning_rate': 0.05,        # Reduced from 0.1 (slower, more accurate)
    'subsample': 0.8,
    'colsample_bytree': 0.6,      # Reduced from 0.8 (less focus on categoricals)
    'min_child_weight': 3,        # NEW: Prevents overfitting
    'gamma': 0.1,                 # NEW: Regularization
    'reg_alpha': 0.1,             # NEW: L1 regularization
    'reg_lambda': 1.0,            # NEW: L2 regularization
    'random_state': 42,
    'n_jobs': -1
}

# ============================================================================
# MODEL 2: RANDOM FOREST CLASSIFICATION PARAMETERS
# ============================================================================
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# ============================================================================
# TRAIN-TEST SPLIT PARAMETERS
# ============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# SPARK CONFIGURATION
# ============================================================================
SPARK_CONFIG = {
    'spark.driver.memory': '8g',
    'spark.executor.memory': '8g',
    'spark.sql.shuffle.partitions': '200'
}

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 6)
DPI = 100

# Color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# BUSINESS RULES
# ============================================================================
# Percentile thresholds for flagging anomalies
HIGH_RISK_PERCENTILE = 95

# Minimum recall for acceptable model performance
MIN_ACCEPTABLE_RECALL = 0.50

# Minimum R² for acceptable regression model
MIN_ACCEPTABLE_R2 = 0.60