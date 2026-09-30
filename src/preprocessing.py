"""
Preprocessing and Scikit-Learn Pipeline Module
Credit Risk Prediction and Explainable Default Analytics

Constructs leak-free ColumnTransformers, Scalers, Encoders, and Stratified train/test partitions.
"""

import os
import sys
import logging
from typing import Tuple, List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

try:
    from src.features import engineer_credit_features, get_feature_lists, TARGET_COLUMN, ID_COLUMN
except ImportError:
    from features import engineer_credit_features, get_feature_lists, TARGET_COLUMN, ID_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("preprocessing")

PROCESSED_DATA_PATH = os.path.join("data", "processed", "clean_credit_card_data.csv")
DEFAULT_RANDOM_STATE = 42
TEST_SIZE = 0.20


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified split into train and test sets to maintain default class proportion.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned credit card dataset.
    test_size : float
        Proportion for held-out test evaluation (default: 0.20).
    random_state : int
        Seed for deterministic reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' missing from dataframe.")

    # Exclude ID and Target from features
    feature_cols = [c for c in df.columns if c not in [TARGET_COLUMN, ID_COLUMN]]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    logger.info("Data split completed: Train=%d samples (default rate=%.2f%%), Test=%d samples (default rate=%.2f%%)",
                len(X_train), y_train.mean() * 100, len(X_test), y_test.mean() * 100)

    return X_train, X_test, y_train, y_test


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer with scaling for numeric and one-hot encoding for categoricals.

    Parameters
    ----------
    numeric_features : List[str]
        List of numerical feature names.
    categorical_features : List[str]
        List of categorical feature names.

    Returns
    -------
    ColumnTransformer
        Fitted or unfitted scikit-learn ColumnTransformer.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def get_preprocessed_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """
    Extract readable feature names after ColumnTransformer transformation (including OneHot categories).
    """
    names = list(numeric_features)
    if "cat" in preprocessor.named_transformers_:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        if hasattr(onehot, "get_feature_names_out"):
            cat_encoded_names = list(onehot.get_feature_names_out(categorical_features))
            names.extend(cat_encoded_names)
        else:
            names.extend(categorical_features)
    return names


def prepare_full_pipeline_data(
    data_path: str = PROCESSED_DATA_PATH,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    """
    Load data, engineer features, perform stratified split, fit preprocessor on train only,
    and return transformed train/test arrays along with metadata.
    """
    df = pd.read_csv(data_path)
    
    # 1. Feature Engineering
    df_feat = engineer_credit_features(df)
    
    # 2. Split features and target
    X_train_df, X_test_df, y_train, y_test = split_data(df_feat, random_state=random_state)
    
    feature_groups = get_feature_lists(X_train_df)
    numeric_cols = feature_groups["numeric"]
    categorical_cols = feature_groups["categorical"]
    
    # 3. Build & fit preprocessor ON TRAIN ONLY to prevent target/test leakage
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_trans = preprocessor.fit_transform(X_train_df)
    X_test_trans = preprocessor.transform(X_test_df)
    
    feature_names = get_preprocessed_feature_names(preprocessor, numeric_cols, categorical_cols)
    
    return {
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names
    }


if __name__ == "__main__":
    prep_data = prepare_full_pipeline_data()
    print("\n--- Preprocessing Pipeline Check ---")
    print(f"X_train transformed shape: {prep_data['X_train_trans'].shape}")
    print(f"X_test transformed shape: {prep_data['X_test_trans'].shape}")
    print(f"Total processed feature count: {len(prep_data['feature_names'])}")
    print(f"Sample processed feature names: {prep_data['feature_names'][:10]}")
