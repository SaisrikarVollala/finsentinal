"""
Tests for Feature Engineering and Edge-Case Safety
"""

import pytest
import numpy as np
import pandas as pd

from src.features import (
    compute_monthly_pay_to_bill_ratio,
    engineer_credit_features,
    CreditFeatureTransformer
)


def test_compute_monthly_pay_to_bill_ratio_edge_cases():
    """Test zero-division, negative bills, overpayments, and standard ratios."""
    # Zero bill, positive payment
    assert compute_monthly_pay_to_bill_ratio(pay_amt=1000, bill_amt=0) == 1.0

    # Negative bill (credit balance/refund)
    assert compute_monthly_pay_to_bill_ratio(pay_amt=0, bill_amt=-500) == 1.0
    assert compute_monthly_pay_to_bill_ratio(pay_amt=500, bill_amt=-500) == 1.0

    # Standard positive bill and payment
    assert compute_monthly_pay_to_bill_ratio(pay_amt=500, bill_amt=1000) == pytest.approx(0.5)

    # Full payment
    assert compute_monthly_pay_to_bill_ratio(pay_amt=1000, bill_amt=1000) == pytest.approx(1.0)

    # Extreme overpayment capped at 2.0
    assert compute_monthly_pay_to_bill_ratio(pay_amt=5000, bill_amt=1000) == 2.0


def test_engineer_credit_features_outputs():
    """Verify calculated features: aggregations, utilization, delays, trends."""
    df_raw = pd.DataFrame([{
        "LIMIT_BAL": 100000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 2,
        "PAY_2": 1,
        "PAY_3": 0,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": 0,
        "BILL_AMT1": 50000,
        "BILL_AMT2": 40000,
        "BILL_AMT3": 30000,
        "BILL_AMT4": 20000,
        "BILL_AMT5": 10000,
        "BILL_AMT6": 0,
        "PAY_AMT1": 2000,
        "PAY_AMT2": 2000,
        "PAY_AMT3": 2000,
        "PAY_AMT4": 2000,
        "PAY_AMT5": 2000,
        "PAY_AMT6": 2000
    }])

    feat_df = engineer_credit_features(df_raw)

    # 1. Delinquency features
    # PAY_0=2 (>0), PAY_2=1 (>0), others <= 0 -> total delayed = 2
    assert feat_df["TOTAL_DELAYED_MONTHS"].iloc[0] == 2
    assert feat_df["MAX_DELAY_MONTHS"].iloc[0] == 2
    # Delay trend = PAY_0 (2) - PAY_6 (0) = 2
    assert feat_df["DELAY_TREND"].iloc[0] == 2
    assert feat_df["EVER_SEVERE_DELAY"].iloc[0] == 1

    # 2. Utilization features
    # BILL1 = 50000 / LIMIT = 100000 -> 0.50
    assert feat_df["UTILIZATION_1"].iloc[0] == pytest.approx(0.50)
    assert feat_df["LATEST_UTILIZATION"].iloc[0] == pytest.approx(0.50)

    # 3. Bill Growth: BILL1 (50000) - BILL6 (0) = 50000
    assert feat_df["BILL_GROWTH"].iloc[0] == pytest.approx(50000)

    # 4. No NaNs or Infs
    assert not feat_df.isnull().any().any()
    assert not np.isinf(feat_df.select_dtypes(include=[np.number]).values).any()


def test_credit_feature_transformer():
    """Verify scikit-learn transformer compatibility."""
    df_raw = pd.DataFrame([{
        "LIMIT_BAL": 50000,
        "SEX": 2,
        "EDUCATION": 1,
        "MARRIAGE": 2,
        "AGE": 28,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 1000,
        "BILL_AMT2": 1000,
        "BILL_AMT3": 1000,
        "BILL_AMT4": 1000,
        "BILL_AMT5": 1000,
        "BILL_AMT6": 1000,
        "PAY_AMT1": 1000,
        "PAY_AMT2": 1000,
        "PAY_AMT3": 1000,
        "PAY_AMT4": 1000,
        "PAY_AMT5": 1000,
        "PAY_AMT6": 1000
    }])

    transformer = CreditFeatureTransformer()
    res = transformer.fit_transform(df_raw)
    assert isinstance(res, pd.DataFrame)
    assert "AVG_UTILIZATION" in res.columns
