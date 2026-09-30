"""
Tests for Data Loader and Schema Validation
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    validate_and_clean_data,
    TARGET_COLUMN,
    ID_COLUMN,
    DATA_DICTIONARY
)


@pytest.fixture
def sample_raw_df():
    """Create a synthetic raw dataframe mirroring UCI dataset schema."""
    data = {
        "ID": [1, 2, 3, 4, 5],
        "LIMIT_BAL": [20000, 50000, 100000, 200000, 500000],
        "SEX": [1, 2, 1, 2, 1],
        "EDUCATION": [1, 2, 0, 5, 3],  # Contains undocumented 0 and 5
        "MARRIAGE": [1, 2, 0, 3, 1],   # Contains undocumented 0
        "AGE": [24, 35, 45, 50, 29],
        "PAY_0": [2, -1, 0, 0, 1],
        "PAY_2": [2, -1, 0, 0, -1],
        "PAY_3": [-1, -1, 0, 0, 0],
        "PAY_4": [-1, 0, 0, 0, 0],
        "PAY_5": [-2, 0, 0, 0, 0],
        "PAY_6": [-2, 0, 0, 0, 0],
        "BILL_AMT1": [3913, 2682, 29239, 46990, 8617],
        "BILL_AMT2": [3102, 1725, 14027, 48233, 5670],
        "BILL_AMT3": [689, 2682, 13559, 49291, 35835],
        "BILL_AMT4": [0, 3272, 14331, 28314, 20940],
        "BILL_AMT5": [0, 3455, 14948, 28959, 19146],
        "BILL_AMT6": [0, 3261, 15549, 29547, 19131],
        "PAY_AMT1": [0, 0, 1518, 2000, 2000],
        "PAY_AMT2": [689, 1000, 1500, 2019, 36681],
        "PAY_AMT3": [0, 1000, 1000, 1200, 10000],
        "PAY_AMT4": [0, 1000, 1000, 1100, 9000],
        "PAY_AMT5": [0, 0, 1000, 1069, 1089],
        "PAY_AMT6": [0, 2000, 5000, 1000, 1000],
        "default_payment_next_month": [1, 0, 0, 0, 1]
    }
    return pd.DataFrame(data)


def test_validation_and_cleaning(sample_raw_df):
    """Test schema validation, education grouping, and duplicate removal."""
    clean_df, report = validate_and_clean_data(sample_raw_df)

    assert report["validation_passed"] is True
    assert len(clean_df) == 5
    assert TARGET_COLUMN in clean_df.columns
    
    # Check EDUCATION grouping (0 and 5 mapped to 4)
    assert set(clean_df["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert clean_df.loc[clean_df["ID"] == 3, "EDUCATION"].iloc[0] == 4
    assert clean_df.loc[clean_df["ID"] == 4, "EDUCATION"].iloc[0] == 4

    # Check MARRIAGE grouping (0 mapped to 3)
    assert set(clean_df["MARRIAGE"].unique()).issubset({1, 2, 3})
    assert clean_df.loc[clean_df["ID"] == 3, "MARRIAGE"].iloc[0] == 3


def test_missing_target_raises_error(sample_raw_df):
    """Ensure ValueError is raised if target column is absent."""
    df_no_target = sample_raw_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="Target column"):
        validate_and_clean_data(df_no_target)


def test_data_dictionary_completeness():
    """Verify data dictionary has all expected raw columns."""
    assert "LIMIT_BAL" in DATA_DICTIONARY
    assert "PAY_0" in DATA_DICTIONARY
    assert "default_payment_next_month" in DATA_DICTIONARY
