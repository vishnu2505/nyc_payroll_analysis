"""
data_processing.py
Data loading, cleaning, and feature engineering functions
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import config


def initialize_spark(app_name="NYC_Payroll_Analysis"):
    """
    Initialize Spark session with configuration from config.py
    
    Args:
        app_name (str): Name of the Spark application
        
    Returns:
        SparkSession: Configured Spark session
    """
    builder = SparkSession.builder.appName(app_name)
    
    for key, value in config.SPARK_CONFIG.items():
        builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    print(f"✓ Spark {spark.version} initialized successfully")
    return spark


def load_data(spark, file_path=None):
    """
    Load NYC Payroll data from CSV
    
    Args:
        spark (SparkSession): Active Spark session
        file_path (str): Path to CSV file. If None, uses config.DATA_PATH
        
    Returns:
        DataFrame: Loaded Spark DataFrame
    """
    if file_path is None:
        file_path = config.DATA_PATH
    
    print(f"Loading data from: {file_path}")
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    print(f"✓ Loaded {df.count():,} records with {len(df.columns)} columns")
    return df


def filter_fiscal_years(df, years=None):
    """
    Filter DataFrame for specific fiscal years
    
    Args:
        df (DataFrame): Input Spark DataFrame
        years (list): List of fiscal years. If None, uses config.FISCAL_YEARS
        
    Returns:
        DataFrame: Filtered DataFrame
    """
    if years is None:
        years = config.FISCAL_YEARS
    
    df_filtered = df.filter(col("Fiscal Year").isin(years))
    
    print(f"✓ Filtered to fiscal years {years}: {df_filtered.count():,} records")
    
    # Show distribution
    print("\nRecords by Fiscal Year:")
    df_filtered.groupBy("Fiscal Year").count().orderBy("Fiscal Year").show()
    
    return df_filtered


def clean_currency_columns(df):
    """
    Clean currency columns by removing $ and commas, converting to float
    
    Args:
        df (DataFrame): Input Spark DataFrame
        
    Returns:
        DataFrame: DataFrame with cleaned currency columns
    """
    currency_columns = [
        "Base Salary",
        "Regular Gross Paid",
        "Total OT Paid",
        "Total Other Pay",
        "Regular Hours",
        "OT Hours"
    ]
    
    print("✓ Cleaning currency columns...")
    
    for col_name in currency_columns:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                regexp_replace(col(col_name), "[$,]", "").cast("float")
            )
    
    return df


def calculate_years_of_service(df, reference_date=None):
    """
    Calculate years of service from agency start date
    
    Args:
        df (DataFrame): Input Spark DataFrame
        reference_date (str): Reference date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame: DataFrame with Years of Service column
    """
    if reference_date is None:
        reference_date = config.REFERENCE_DATE
    
    print(f"✓ Calculating Years of Service (reference: {reference_date})")
    
    df = df.withColumn(
        "Agency Start Date",
        to_date(col("Agency Start Date"), "MM/dd/yyyy")
    )
    
    df = df.withColumn(
        "Years of Service",
        datediff(lit(reference_date), col("Agency Start Date")) / 365.25
    )
    
    return df


def create_derived_features(df):
    """
    Create derived features for analysis
    
    Args:
        df (DataFrame): Input Spark DataFrame
        
    Returns:
        DataFrame: DataFrame with additional derived features
    """
    print("✓ Creating derived features...")
    
    # Total Compensation
    df = df.withColumn(
        "Total Compensation",
        col("Base Salary") + col("Total OT Paid") + col("Total Other Pay")
    )
    
    # Overtime as percentage of base salary
    df = df.withColumn(
        "OT_Percentage",
        when(col("Base Salary") > 0,
             (col("Total OT Paid") / col("Base Salary")) * 100)
        .otherwise(0)
    )
    
    # Overtime risk category
    df = df.withColumn(
        "OT_Risk_Category",
        when(col("OT Hours") == 0, "No OT")
        .when(col("OT Hours") < config.OT_LOW_THRESHOLD, "Low")
        .when(col("OT Hours") < config.OT_MEDIUM_THRESHOLD, "Medium")
        .otherwise("High")
    )
    
    # Binary high-risk flag for classification
    df = df.withColumn(
        "At_Risk",
        when(col("OT Hours") >= config.OT_MEDIUM_THRESHOLD, 1)
        .otherwise(0)
    )
    
    return df


def remove_invalid_records(df):
    """
    Remove records with invalid or missing critical data
    
    Args:
        df (DataFrame): Input Spark DataFrame
        
    Returns:
        DataFrame: Cleaned DataFrame
    """
    initial_count = df.count()
    
    df_clean = df.filter(
        (col("Base Salary").isNotNull()) &
        (col("Base Salary") > config.MIN_SALARY) &
        (col("Base Salary") < config.MAX_SALARY) &
        (col("Years of Service").isNotNull()) &
        (col("Years of Service") >= config.MIN_YEARS_SERVICE) &
        (col("Years of Service") <= config.MAX_YEARS_SERVICE)
    )
    
    # Fill missing values for non-critical fields
    df_clean = df_clean.fillna({
        "OT Hours": 0,
        "Total OT Paid": 0,
        "Work Location Borough": "UNKNOWN"
    })
    
    final_count = df_clean.count()
    removed = initial_count - final_count
    
    print(f"✓ Removed {removed:,} invalid records ({removed/initial_count*100:.2f}%)")
    print(f"✓ Clean dataset: {final_count:,} records")
    
    return df_clean


def reduce_cardinality(df_pd, top_n_agencies=None, top_n_titles=None):
    """
    Reduce cardinality of high-cardinality categorical variables
    
    Args:
        df_pd (DataFrame): Pandas DataFrame
        top_n_agencies (int): Number of top agencies to keep
        top_n_titles (int): Number of top job titles to keep
        
    Returns:
        tuple: (DataFrame, list of top agencies, list of top titles)
    """
    if top_n_agencies is None:
        top_n_agencies = config.TOP_N_AGENCIES
    if top_n_titles is None:
        top_n_titles = config.TOP_N_TITLES
    
    # Get top agencies
    top_agencies = df_pd['Agency Name'].value_counts().head(top_n_agencies).index.tolist()
    df_pd['Agency Name'] = df_pd['Agency Name'].apply(
        lambda x: x if x in top_agencies else 'OTHER'
    )
    
    # Get top titles
    top_titles = df_pd['Title Description'].value_counts().head(top_n_titles).index.tolist()
    df_pd['Title Description'] = df_pd['Title Description'].apply(
        lambda x: x if x in top_titles else 'OTHER'
    )
    
    print(f"✓ Reduced cardinality: Top {top_n_agencies} agencies, Top {top_n_titles} titles")
    
    return df_pd, top_agencies, top_titles


def prepare_model1_features(df):
    """
    Prepare features specifically for Model 1 (Salary Prediction)
    
    Args:
        df (DataFrame): Cleaned Spark DataFrame
        
    Returns:
        DataFrame: DataFrame with features for Model 1
    """
    features = [
        "Fiscal Year",
        "Base Salary",
        "Years of Service",
        "Agency Name",
        "Work Location Borough",
        "Title Description",
        "Pay Basis",
        "Leave Status as of June 30"
    ]
    
    df_model = df.select(*features).dropna()
    
    print(f"✓ Model 1 features prepared: {df_model.count():,} samples")
    return df_model


def prepare_model2_features(df):
    """
    Prepare features specifically for Model 2 (Overtime Risk Classification)
    
    Args:
        df (DataFrame): Cleaned Spark DataFrame
        
    Returns:
        DataFrame: DataFrame with features for Model 2
    """
    features = [
        "Fiscal Year",
        "Base Salary",
        "Years of Service",
        "OT Hours",
        "Total OT Paid",
        "Agency Name",
        "Title Description",
        "Work Location Borough",
        "At_Risk",
        "OT_Risk_Category"
    ]
    
    df_model = df.select(*features).dropna()
    
    print(f"✓ Model 2 features prepared: {df_model.count():,} samples")
    
    # Show class distribution
    print("\nOvertime Risk Distribution:")
    df_model.groupBy("At_Risk").count().show()
    
    return df_model


def encode_features(df_pd, target_col, exclude_cols=None):
    """
    One-hot encode categorical features
    
    Args:
        df_pd (DataFrame): Pandas DataFrame
        target_col (str): Name of target column
        exclude_cols (list): Columns to exclude from features
        
    Returns:
        tuple: (X, y) - Feature matrix and target vector
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Identify categorical columns
    categorical_cols = df_pd.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target and excluded columns
    categorical_cols = [col for col in categorical_cols 
                       if col != target_col and col not in exclude_cols]
    
    # One-hot encode
    df_encoded = pd.get_dummies(df_pd, columns=categorical_cols, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop([target_col] + exclude_cols, axis=1, errors='ignore')
    y = df_encoded[target_col]
    
    print(f"✓ Encoded features: {X.shape[1]} features, {len(y)} samples")
    
    return X, y


def get_summary_statistics(df):
    """
    Get summary statistics of the dataset
    
    Args:
        df (DataFrame): Spark DataFrame
        
    Returns:
        None (prints statistics)
    """
    print("\n" + "="*80)
    print("DATASET SUMMARY STATISTICS")
    print("="*80)
    
    # Numerical summaries
    print("\nNumerical Columns:")
    df.select(
        "Base Salary",
        "Years of Service",
        "OT Hours",
        "Total OT Paid"
    ).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show()
    
    # Categorical summaries
    print("\nAverage Salary by Borough:")
    df.groupBy("Work Location Borough").agg(
        count("*").alias("Count"),
        round(avg("Base Salary"), 2).alias("Avg_Salary")
    ).orderBy(desc("Avg_Salary")).show(10)
    
    print("\nTop 10 Agencies by Employee Count:")
    df.groupBy("Agency Name").agg(
        count("*").alias("Count")
    ).orderBy(desc("Count")).limit(10).show(truncate=False)


def full_data_pipeline(spark, file_path=None, years=None):
    """
    Execute complete data processing pipeline
    
    Args:
        spark (SparkSession): Active Spark session
        file_path (str): Path to data file
        years (list): Fiscal years to include
        
    Returns:
        DataFrame: Fully processed Spark DataFrame
    """
    print("\n" + "="*80)
    print("EXECUTING FULL DATA PROCESSING PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Load data
    df = load_data(spark, file_path)
    
    # Step 2: Filter fiscal years
    df = filter_fiscal_years(df, years)
    
    # Step 3: Clean currency columns
    df = clean_currency_columns(df)
    
    # Step 4: Calculate years of service
    df = calculate_years_of_service(df)
    
    # Step 5: Create derived features
    df = create_derived_features(df)
    
    # Step 6: Remove invalid records
    df = remove_invalid_records(df)
    
    # Step 7: Get summary statistics
    get_summary_statistics(df)
    
    print("\n✓ Data processing pipeline completed successfully!")
    
    return df