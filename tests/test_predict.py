"""
Tests for Inference, Explainability, and Risk Tier Assignment
"""

import pytest
import numpy as np
import pandas as pd

from src.predict import CreditRiskPredictor, get_default_sample_customer


@pytest.fixture
def predictor():
    """Load production predictor instance."""
    return CreditRiskPredictor(model_name="xgboost")


def test_predict_single_sample_structure(predictor):
    """Verify inference output schema, types, and probability bounds."""
    sample = get_default_sample_customer()
    result = predictor.predict_single(sample)

    assert "predicted_probability" in result
    assert "risk_tier" in result
    assert "flagged_for_intervention" in result
    assert "top_risk_drivers" in result
    assert "top_protective_factors" in result

    prob = result["predicted_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

    flagged = result["flagged_for_intervention"]
    assert isinstance(flagged, bool)
    assert flagged == (prob >= result["decision_threshold"])


def test_risk_tier_assignment():
    """Verify tier assignment boundaries."""
    assert CreditRiskPredictor.assign_risk_tier(0.05) == "Low Risk (Tier 1)"
    assert CreditRiskPredictor.assign_risk_tier(0.20) == "Moderate Risk (Tier 2)"
    assert CreditRiskPredictor.assign_risk_tier(0.45) == "Elevated Risk (Tier 3)"
    assert CreditRiskPredictor.assign_risk_tier(0.75) == "High Risk / Imminent Default (Tier 4)"


def test_global_feature_importance_format(predictor):
    """Verify global feature importances dataframe."""
    df_imp = predictor.get_global_feature_importance(top_n=10)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 10
    assert "Feature" in df_imp.columns
    assert "Importance" in df_imp.columns
    assert (df_imp["Importance"] >= 0).all()
