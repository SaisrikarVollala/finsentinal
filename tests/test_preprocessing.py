"""
Tests for Preprocessing and Pipeline Integrity
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import (
    split_data,
    build_preprocessor,
    prepare_full_pipeline_data,
    DEFAULT_RANDOM_STATE
)
from src.features import TARGET_COLUMN, ID_COLUMN


def test_split_data_proportions_and_stratification():
    """Verify stratified splitting maintains exact default proportion."""
    # Synthetic dataset with 20% default rate
    data = {
        "ID": list(range(100)),
        "LIMIT_BAL": np.random.randint(10000, 100000, size=100),
        "SEX": np.random.choice([1, 2], size=100),
        "EDUCATION": np.random.choice([1, 2, 3, 4], size=100),
        "MARRIAGE": np.random.choice([1, 2, 3], size=100),
        "AGE": np.random.randint(20, 60, size=100),
        TARGET_COLUMN: [1]*20 + [0]*80
    }
    df = pd.DataFrame(data)

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=DEFAULT_RANDOM_STATE)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert y_train.mean() == pytest.approx(0.20, abs=0.01)
    assert y_test.mean() == pytest.approx(0.20, abs=0.01)
    assert ID_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_train.columns


def test_preprocessor_transformation_shapes():
    """Verify ColumnTransformer scales numeric and one-hot encodes categorical features."""
    num_cols = ["LIMIT_BAL", "AGE"]
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = build_preprocessor(num_cols, cat_cols)

    df_train = pd.DataFrame({
        "LIMIT_BAL": [50000, 100000, 150000],
        "AGE": [25, 35, 45],
        "SEX": [1, 2, 1],
        "EDUCATION": [1, 2, 3],
        "MARRIAGE": [1, 2, 1]
    })

    X_trans = preprocessor.fit_transform(df_train)
    # Check transformed shape is numeric 2D array
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 3
    # 2 numeric + onehot columns
    assert X_trans.shape[1] >= 5
