"""
Feature Engineering Module
Credit Risk Prediction and Explainable Default Analytics

This module constructs domain-specific financial risk indicators from raw account
and repayment histories available at prediction time (t = September 2005).
All calculations are strictly backward-looking to prevent temporal leakage.
"""

import logging
from typing import List, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("features")

# Feature Name Constants
RAW_BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
RAW_PAY_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
TARGET_COLUMN = "default_payment_next_month"
ID_COLUMN = "ID"
EXCLUDE_COLS = [ID_COLUMN, TARGET_COLUMN]


def compute_monthly_pay_to_bill_ratio(pay_amt: float, bill_amt: float) -> float:
    """
    Compute payment-to-bill ratio for a single month with robust zero-denominator handling.

    Rules:
    - If bill <= 0 and pay >= 0: Customer had no positive debt; return 1.0 (satisfactory)
    - If bill > 0 and pay >= 0: pay / bill (capped at 2.0 to prevent outlier distortion)
    - If bill > 0 and pay < 0: return 0.0
    """
    if bill_amt <= 0:
        return 1.0 if pay_amt >= 0 else 0.0
    ratio = pay_amt / float(bill_amt)
    return float(np.clip(ratio, 0.0, 2.0))


def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered credit risk, delinquency, utilization, and liquidity features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw or cleaned credit card data.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with engineered features.
    """
    out = df.copy()

    # Ensure numeric types
    for col in RAW_BILL_COLS + RAW_PAY_COLS + RAW_PAY_STATUS_COLS + ["LIMIT_BAL", "AGE"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    # 1. Aggregated Statement & Payment Metrics
    out["AVG_BILL_AMT"] = out[RAW_BILL_COLS].mean(axis=1)
    out["MAX_BILL_AMT"] = out[RAW_BILL_COLS].max(axis=1)
    out["MIN_BILL_AMT"] = out[RAW_BILL_COLS].min(axis=1)
    out["STD_BILL_AMT"] = out[RAW_BILL_COLS].std(axis=1).fillna(0.0)

    out["AVG_PAY_AMT"] = out[RAW_PAY_COLS].mean(axis=1)
    out["MAX_PAY_AMT"] = out[RAW_PAY_COLS].max(axis=1)
    out["MIN_PAY_AMT"] = out[RAW_PAY_COLS].min(axis=1)
    out["SUM_PAY_AMT"] = out[RAW_PAY_COLS].sum(axis=1)

    # 2. Credit Utilization Proxies (Bill / Limit)
    # Capped between -0.5 and 3.0 to manage negative bill balances and extreme over-limit charges
    for i in range(1, 7):
        bill_col = f"BILL_AMT{i}"
        util_col = f"UTILIZATION_{i}"
        # Guard against zero or negative limit
        limit_safe = np.maximum(out["LIMIT_BAL"].values, 1.0)
        out[util_col] = np.clip(out[bill_col].values / limit_safe, -0.5, 3.0)

    util_cols = [f"UTILIZATION_{i}" for i in range(1, 7)]
    out["AVG_UTILIZATION"] = out[util_cols].mean(axis=1)
    out["LATEST_UTILIZATION"] = out["UTILIZATION_1"]
    out["MAX_UTILIZATION"] = out[util_cols].max(axis=1)

    # 3. Delinquency and Delay Dynamics
    # Count how many months of delay (>0) over the 6-month historical observation window
    delay_flags = (out[RAW_PAY_STATUS_COLS] > 0).astype(int)
    out["TOTAL_DELAYED_MONTHS"] = delay_flags.sum(axis=1)
    
    # Maximum consecutive/historical delay
    out["MAX_DELAY_MONTHS"] = np.maximum(out[RAW_PAY_STATUS_COLS].max(axis=1), 0)
    
    # Recent vs Old Delinquency Trajectory (PAY_0 - PAY_6)
    # Positive means delinquency is worsening; negative means improving
    out["DELAY_TREND"] = out["PAY_0"] - out["PAY_6"]
    
    # Severe delay indicator (any month >= 2 months overdue)
    out["EVER_SEVERE_DELAY"] = (out[RAW_PAY_STATUS_COLS] >= 2).any(axis=1).astype(int)

    # 4. Payment-to-Bill Ratios (Liquidity and Repayment Completeness)
    pay_to_bill_cols = []
    for i in range(1, 7):
        col_name = f"PAY_TO_BILL_RATIO_{i}"
        pay_to_bill_cols.append(col_name)
        
        bills = out[f"BILL_AMT{i}"].values
        pays = out[f"PAY_AMT{i}"].values
        
        ratios = np.where(
            bills <= 0,
            np.where(pays >= 0, 1.0, 0.0),
            np.clip(pays / np.maximum(bills, 1e-6), 0.0, 2.0)
        )
        out[col_name] = ratios

    out["AVG_PAY_TO_BILL_RATIO"] = out[pay_to_bill_cols].mean(axis=1)
    out["LATEST_PAY_TO_BILL_RATIO"] = out["PAY_TO_BILL_RATIO_1"]

    # 5. Financial Trajectory & Debt Expansion
    # Net Balance accumulation over 6 months: BILL_AMT1 - BILL_AMT6
    out["BILL_GROWTH"] = out["BILL_AMT1"] - out["BILL_AMT6"]
    
    # Bill to Payment Spread (September): Net unpaid statement exposure
    out["NET_UNPAID_EXPOSURE_1"] = np.maximum(out["BILL_AMT1"] - out["PAY_AMT1"], 0.0)

    # Handle any remaining NaNs or Infs gracefully
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    return out


class CreditFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for reproducible feature engineering in ML pipelines.
    """
    def __init__(self):
        self.engineered_feature_names_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            # If array, convert to DataFrame using standard raw column names if available
            raise ValueError("CreditFeatureTransformer expects a pandas DataFrame with named columns.")
        
        enriched_df = engineer_credit_features(X)
        self.engineered_feature_names_ = [c for c in enriched_df.columns if c not in EXCLUDE_COLS]
        return enriched_df


def get_feature_lists(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns into numeric, categorical, and excluded for downstream Pipeline.
    """
    all_cols = list(df.columns)
    cat_cols = [c for c in CATEGORICAL_COLS if c in all_cols]
    exclude = [c for c in EXCLUDE_COLS if c in all_cols]
    num_cols = [c for c in all_cols if c not in cat_cols and c not in exclude]
    
    return {
        "numeric": num_cols,
        "categorical": cat_cols,
        "exclude": exclude
    }
