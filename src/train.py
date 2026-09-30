"""
Model Training and Hyperparameter Tuning Module
Credit Risk Prediction and Explainable Default Analytics

Trains and cross-validates Logistic Regression, Random Forest, and XGBoost models
using Stratified K-Fold CV, tunes decision thresholds, and persists model artifacts.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss
)

from src.preprocessing import prepare_full_pipeline_data, DEFAULT_RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")

MODELS_DIR = os.path.join("models")


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal classification probability threshold using validation predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for positive class (default).
    metric : str
        Metric to optimize ('f1' or 'balanced_f1').

    Returns
    -------
    Tuple[float, float]
        Optimal threshold and corresponding peak metric score.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.50
    best_score = 0.0

    for th in thresholds:
        preds = (y_proba >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = float(th)

    return best_thresh, best_score


def train_and_cross_validate_models(
    prep_data: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    """
    Perform 5-Fold Stratified Cross-Validation on training set for candidate models.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    X_train = prep_data["X_train_trans"]
    y_train = np.array(prep_data["y_train"])
    
    # Calculate scale_pos_weight for XGBoost to handle class imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / float(num_pos)  # ~3.52

    logger.info("Class imbalance ratio in training set: %.2f (Neg: %d, Pos: %d)",
                scale_pos_weight, num_neg, num_pos)

    # Candidate Models with tuned regularizations and class weightings
    candidate_models = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        ),
        "RandomForest": XGBClassifier(
            learning_rate=1.0,
            subsample=0.8,
            colsample_bynode=0.8,
            num_parallel_tree=150,
            n_estimators=1,
            max_depth=8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state
        )
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = {}

    for name, model in candidate_models.items():
        logger.info("Cross-validating %s (%d folds)...", name, n_splits)
        
        # Out-of-fold probability predictions via CV to prevent any leakage
        oof_probas = cross_val_predict(
            model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        
        roc_auc = roc_auc_score(y_train, oof_probas)
        pr_auc = average_precision_score(y_train, oof_probas)
        brier = brier_score_loss(y_train, oof_probas)
        
        # Determine optimal decision threshold on CV out-of-fold probabilities
        opt_threshold, opt_f1 = find_optimal_threshold(y_train, oof_probas, metric="f1")
        
        # Standard threshold metrics (0.50)
        std_preds = (oof_probas >= 0.50).astype(int)
        std_prec = precision_score(y_train, std_preds, zero_division=0)
        std_rec = recall_score(y_train, std_preds, zero_division=0)
        std_f1 = f1_score(y_train, std_preds, zero_division=0)
        
        # Optimal threshold metrics
        opt_preds = (oof_probas >= opt_threshold).astype(int)
        opt_prec = precision_score(y_train, opt_preds, zero_division=0)
        opt_rec = recall_score(y_train, opt_preds, zero_division=0)

        cv_results[name] = {
            "cv_roc_auc": float(roc_auc),
            "cv_pr_auc": float(pr_auc),
            "cv_brier_score": float(brier),
            "std_threshold": 0.50,
            "std_precision": float(std_prec),
            "std_recall": float(std_rec),
            "std_f1": float(std_f1),
            "optimal_threshold": float(opt_threshold),
            "opt_precision": float(opt_prec),
            "opt_recall": float(opt_rec),
            "opt_f1": float(opt_f1),
            "oof_probas": oof_probas
        }
        
        logger.info(
            "[%s CV] ROC-AUC: %.4f | PR-AUC: %.4f | Std F1: %.4f | Opt Thresh: %.2f (F1: %.4f, Prec: %.4f, Rec: %.4f)",
            name, roc_auc, pr_auc, std_f1, opt_threshold, opt_f1, opt_prec, opt_rec
        )
        
        # Fit model on the full training set
        model.fit(X_train, y_train)
        
        # Save fitted model
        model_save_path = os.path.join(MODELS_DIR, f"{name.lower()}_model.joblib")
        joblib.dump(model, model_save_path)
        logger.info("Saved fitted model artifact to %s", model_save_path)

    # Save Preprocessor and Feature Metadata
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(prep_data["preprocessor"], preprocessor_path)
    logger.info("Saved fitted preprocessor to %s", preprocessor_path)
    
    meta_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "numeric_cols": prep_data["numeric_cols"],
            "categorical_cols": prep_data["categorical_cols"],
            "feature_names": prep_data["feature_names"]
        }, f, indent=2)

    # Determine winning model based on CV PR-AUC and ROC-AUC
    # In imbalanced credit default prediction, PR-AUC and ROC-AUC are paramount
    best_model_name = max(cv_results, key=lambda k: cv_results[k]["cv_roc_auc"] + cv_results[k]["cv_pr_auc"])
    logger.info("Best model selected by CV metrics: %s", best_model_name)
    
    cv_summary_path = os.path.join(MODELS_DIR, "cv_results.json")
    # Exclude raw probability arrays from json export
    serializable_cv = {
        k: {metric: v for metric, v in res.items() if metric != "oof_probas"}
        for k, res in cv_results.items()
    }
    serializable_cv["winning_model"] = best_model_name
    serializable_cv["winning_threshold"] = cv_results[best_model_name]["optimal_threshold"]
    
    with open(cv_summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable_cv, f, indent=2)
        
    return {
        "candidate_models": candidate_models,
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "best_threshold": cv_results[best_model_name]["optimal_threshold"]
    }


def run_training_pipeline() -> Dict[str, Any]:
    """Execute end-to-end data preparation, model training, and CV evaluation."""
    logger.info("Starting model training pipeline...")
    prep_data = prepare_full_pipeline_data()
    train_results = train_and_cross_validate_models(prep_data)
    logger.info("Training pipeline completed successfully.")
    return train_results


if __name__ == "__main__":
    results = run_training_pipeline()
    print("\n--- Model Training & Cross-Validation Complete ---")
    print(f"Winning Model: {results['best_model_name']}")
    print(f"Optimal Threshold: {results['best_threshold']:.2f}")
