"""
Inference and Model Explainability Module (SHAP & Global Importances)
Credit Risk Prediction and Explainable Default Analytics

Provides end-to-end prediction capabilities, risk tier assignment,
and local / global feature attribution using SHAP and Tree Explainer.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Union, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import engineer_credit_features
from src.preprocessing import get_feature_lists

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predict")

MODELS_DIR = os.path.join("models")


class CreditRiskPredictor:
    """
    Production-grade predictor for Next-Month Credit Card Default.
    Loads saved model, preprocessor, and provides explainability via exact Tree SHAP.
    """

    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name.lower()
        self.model = None
        self.preprocessor = None
        self.feature_names: List[str] = []
        self.optimal_threshold: float = 0.50
        self._load_artifacts()

    def _load_artifacts(self):
        """Load fitted models, preprocessor, and feature names."""
        model_file = os.path.join(MODELS_DIR, f"{self.model_name}_model.joblib")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}. Please run train.py first.")
        self.model = joblib.load(model_file)

        prep_file = os.path.join(MODELS_DIR, "preprocessor.joblib")
        if not os.path.exists(prep_file):
            raise FileNotFoundError(f"Preprocessor file not found at {prep_file}. Please run train.py first.")
        self.preprocessor = joblib.load(prep_file)

        names_file = os.path.join(MODELS_DIR, "feature_names.json")
        if os.path.exists(names_file):
            with open(names_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])

        cv_file = os.path.join(MODELS_DIR, "cv_results.json")
        if os.path.exists(cv_file):
            with open(cv_file, "r", encoding="utf-8") as f:
                cv_data = json.load(f)
                model_key = "XGBoost" if "xgboost" in self.model_name else ("RandomForest" if "randomforest" in self.model_name else "LogisticRegression")
                self.optimal_threshold = float(cv_data.get(model_key, {}).get("optimal_threshold", 0.50))

    def preprocess_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> np.ndarray:
        """
        Transform raw input dictionary or DataFrame using feature engineering and preprocessor.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Engineer features
        df_feat = engineer_credit_features(df)
        
        # Preprocessor transform
        X_trans = self.preprocessor.transform(df_feat)
        return X_trans

    @staticmethod
    def assign_risk_tier(probability: float) -> str:
        """
        Map predicted default probability to standard credit risk bands.
        """
        if probability < 0.15:
            return "Low Risk (Tier 1)"
        elif probability < 0.35:
            return "Moderate Risk (Tier 2)"
        elif probability < 0.60:
            return "Elevated Risk (Tier 3)"
        else:
            return "High Risk / Imminent Default (Tier 4)"

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict next-month default probability for a single customer profile with SHAP attribution.
        """
        X_trans = self.preprocess_input(customer_data)
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        risk_tier = self.assign_risk_tier(prob)
        flagged = bool(prob >= self.optimal_threshold)

        explanation = self.explain_single(X_trans)

        return {
            "predicted_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "flagged_for_intervention": flagged,
            "decision_threshold": round(self.optimal_threshold, 2),
            "top_risk_drivers": explanation.get("top_positive_features", []),
            "top_protective_factors": explanation.get("top_negative_features", []),
            "all_shap_contributions": explanation.get("feature_contributions", {})
        }

    def explain_single(self, X_trans: np.ndarray) -> Dict[str, Any]:
        """
        Calculate local SHAP feature contributions for a preprocessed sample.
        """
        try:
            if hasattr(self.model, "get_booster"):
                dmat = xgb.DMatrix(X_trans)
                contribs = self.model.get_booster().predict(dmat, pred_contribs=True)[0]
                # First len(feature_names) are feature SHAP values; last element is base value
                vals = contribs[:len(self.feature_names)]
            elif hasattr(self.model, "coef_"):
                # Linear model feature attribution: w_i * x_i
                vals = (self.model.coef_[0] * X_trans[0])[:len(self.feature_names)]
            else:
                vals = np.zeros(len(self.feature_names))

            contributions = {}
            for name, val in zip(self.feature_names, vals):
                contributions[name] = float(val)

            sorted_by_abs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Risk drivers (positive contribution pushing towards default)
            pos_drivers = [{"feature": k, "impact": round(v, 4)} for k, v in sorted_by_abs if v > 0][:5]
            
            # Protective factors (negative contribution pushing away from default)
            neg_factors = [{"feature": k, "impact": round(v, 4)} for k, v in sorted_by_abs if v < 0][:5]

            return {
                "top_positive_features": pos_drivers,
                "top_negative_features": neg_factors,
                "feature_contributions": contributions
            }
        except Exception as e:
            logger.warning("Local explanation failed: %s", e)
            return {}

    def get_global_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Return global feature importances from the underlying model.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            df_imp = pd.DataFrame({
                "Feature": self.feature_names[:len(importances)],
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
            return df_imp.head(top_n)
        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_[0])
            df_imp = pd.DataFrame({
                "Feature": self.feature_names[:len(coefs)],
                "Importance": coefs
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
            return df_imp.head(top_n)
        else:
            return pd.DataFrame()


def get_default_sample_customer() -> Dict[str, Any]:
    """Return a representative customer dictionary for interactive testing and default forms."""
    return {
        "LIMIT_BAL": 50000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 48000,
        "BILL_AMT2": 47000,
        "BILL_AMT3": 45000,
        "BILL_AMT4": 40000,
        "BILL_AMT5": 38000,
        "BILL_AMT6": 35000,
        "PAY_AMT1": 2000,
        "PAY_AMT2": 2000,
        "PAY_AMT3": 1500,
        "PAY_AMT4": 1500,
        "PAY_AMT5": 1000,
        "PAY_AMT6": 1000
    }


if __name__ == "__main__":
    predictor = CreditRiskPredictor(model_name="xgboost")
    sample = get_default_sample_customer()
    result = predictor.predict_single(sample)
    print("\n--- Sample Customer Inference & SHAP Explanation ---")
    print(json.dumps(result, indent=2))
    
    print("\n--- Global Top 10 Features ---")
    print(predictor.get_global_feature_importance(10))
