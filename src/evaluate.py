"""
Model Evaluation and Diagnostic Visualization Module
Credit Risk Prediction and Explainable Default Analytics

Evaluates fitted models on the untouched held-out test partition (6,000 samples),
computes comprehensive discrimination and calibration metrics, and generates
diagnostic publication-quality figures.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    balanced_accuracy_score
)
from sklearn.calibration import calibration_curve

from src.preprocessing import prepare_full_pipeline_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

MODELS_DIR = os.path.join("models")
REPORTS_FIGURES_DIR = os.path.join("reports", "figures")

MODEL_COLORS = {
    "LogisticRegression": "#2b5c8f",
    "RandomForest": "#2ca02c",
    "XGBoost": "#d9534f"
}


def load_trained_models() -> Dict[str, Any]:
    """Load all saved model artifacts from models/ directory."""
    models = {}
    model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
    for name in model_names:
        path = os.path.join(MODELS_DIR, f"{name.lower()}_model.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            logger.info("Loaded %s from %s", name, path)
        else:
            logger.warning("Model file not found: %s", path)
    return models


def evaluate_models_on_test_set(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute full suite of test metrics for all models.
    """
    test_results = {}
    y_test_arr = np.array(y_test)
    test_default_rate = float(y_test_arr.mean())

    for name, model in models.items():
        probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test_arr, probas)
        pr_auc = average_precision_score(y_test_arr, probas)
        brier = brier_score_loss(y_test_arr, probas)

        # Retrieve CV-tuned optimal threshold
        opt_thresh = float(cv_summary.get(name, {}).get("optimal_threshold", 0.50))
        
        # Standard threshold (0.50)
        std_preds = (probas >= 0.50).astype(int)
        std_cm = confusion_matrix(y_test_arr, std_preds).tolist()
        tn, fp, fn, tp = confusion_matrix(y_test_arr, std_preds).ravel()
        std_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Optimal threshold
        opt_preds = (probas >= opt_thresh).astype(int)
        opt_cm = confusion_matrix(y_test_arr, opt_preds).tolist()
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test_arr, opt_preds).ravel()
        opt_spec = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0

        test_results[name] = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
            "default_rate": test_default_rate,
            "standard_0_50": {
                "threshold": 0.50,
                "precision": float(precision_score(y_test_arr, std_preds, zero_division=0)),
                "recall": float(recall_score(y_test_arr, std_preds, zero_division=0)),
                "f1": float(f1_score(y_test_arr, std_preds, zero_division=0)),
                "specificity": float(std_spec),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_arr, std_preds)),
                "confusion_matrix": std_cm,
                "counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
            },
            "optimal_threshold_eval": {
                "threshold": opt_thresh,
                "precision": float(precision_score(y_test_arr, opt_preds, zero_division=0)),
                "recall": float(recall_score(y_test_arr, opt_preds, zero_division=0)),
                "f1": float(f1_score(y_test_arr, opt_preds, zero_division=0)),
                "specificity": float(opt_spec),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_arr, opt_preds)),
                "confusion_matrix": opt_cm,
                "counts": {"tn": int(tn_opt), "fp": int(fp_opt), "fn": int(fn_opt), "tp": int(tp_opt)}
            },
            "probas": probas
        }

        logger.info(
            "[%s Held-out Test] ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | Opt Thresh (%.2f) -> Prec: %.4f, Rec: %.4f, F1: %.4f",
            name, roc_auc, pr_auc, brier, opt_thresh,
            test_results[name]["optimal_threshold_eval"]["precision"],
            test_results[name]["optimal_threshold_eval"]["recall"],
            test_results[name]["optimal_threshold_eval"]["f1"]
        )

    return test_results


def plot_roc_curves(test_results: Dict[str, Any], y_test: np.ndarray, output_dir: str = REPORTS_FIGURES_DIR):
    """Generate ROC curves comparison on held-out test set."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, res in test_results.items():
        probas = res["probas"]
        fpr, tpr, _ = roc_curve(y_test, probas)
        auc_val = res["roc_auc"]
        color = MODEL_COLORS.get(name, "#333333")
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2.2)

    # Diagonal no-skill line
    ax.plot([0, 1], [0, 1], "k--", label="Chance Baseline (AUC = 0.500)", linewidth=1.2)
    ax.set_title("Held-Out Test Set: Receiver Operating Characteristic (ROC) Curves", pad=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(output_dir, "07_roc_curves.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_pr_curves(test_results: Dict[str, Any], y_test: np.ndarray, output_dir: str = REPORTS_FIGURES_DIR):
    """Generate Precision-Recall curves comparison with empirical baseline."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    baseline = np.mean(y_test)
    for name, res in test_results.items():
        probas = res["probas"]
        prec, rec, _ = precision_recall_curve(y_test, probas)
        pr_auc_val = res["pr_auc"]
        color = MODEL_COLORS.get(name, "#333333")
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", color=color, linewidth=2.2)

    # Baseline no-skill horizontal line
    ax.axhline(baseline, color="gray", linestyle="--", label=f"No-Skill Baseline (Rate = {baseline:.3f})", linewidth=1.2)
    ax.set_title("Held-Out Test Set: Precision-Recall (PR) Curves", pad=12, fontweight="bold")
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(output_dir, "08_pr_curves.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_calibration_curves(test_results: Dict[str, Any], y_test: np.ndarray, output_dir: str = REPORTS_FIGURES_DIR):
    """Generate reliability diagrams / calibration curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, res in test_results.items():
        probas = res["probas"]
        prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10, strategy="uniform")
        brier = res["brier_score"]
        color = MODEL_COLORS.get(name, "#333333")
        ax.plot(prob_pred, prob_true, marker="s", label=f"{name} (Brier = {brier:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.2)
    ax.set_title("Probability Calibration (Reliability Diagram)", pad=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Default Probability")
    ax.set_ylabel("Observed Fraction of Defaults")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(output_dir, "09_calibration_curves.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_confusion_matrices(
    best_res: Dict[str, Any],
    best_name: str,
    output_dir: str = REPORTS_FIGURES_DIR
):
    """Plot confusion matrices comparing default 0.50 threshold vs optimal threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Standard 0.50
    cm_std = np.array(best_res["standard_0_50"]["confusion_matrix"])
    sns.heatmap(cm_std, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Pred: Non-Default", "Pred: Default"],
                yticklabels=["Actual: Non-Default", "Actual: Default"])
    axes[0].set_title(f"{best_name} @ Standard Threshold (0.50)\nPrec: {best_res['standard_0_50']['precision']:.3f}, Rec: {best_res['standard_0_50']['recall']:.3f}, F1: {best_res['standard_0_50']['f1']:.3f}")

    # 2. Optimal Threshold
    cm_opt = np.array(best_res["optimal_threshold_eval"]["confusion_matrix"])
    opt_th = best_res["optimal_threshold_eval"]["threshold"]
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=["Pred: Non-Default", "Pred: Default"],
                yticklabels=["Actual: Non-Default", "Actual: Default"])
    axes[1].set_title(f"{best_name} @ Optimal Threshold ({opt_th:.2f})\nPrec: {best_res['optimal_threshold_eval']['precision']:.3f}, Rec: {best_res['optimal_threshold_eval']['recall']:.3f}, F1: {best_res['optimal_threshold_eval']['f1']:.3f}")

    fig.suptitle("Confusion Matrix Dynamics Under Operating Threshold Adjustment", fontsize=13, fontweight="bold", y=1.02)
    save_path = os.path.join(output_dir, "10_confusion_matrices.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_threshold_tradeoff(
    probas: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    output_dir: str = REPORTS_FIGURES_DIR
):
    """Plot Precision, Recall, and F1 curves across probability thresholds."""
    thresholds = np.linspace(0.05, 0.95, 91)
    precisions = []
    recalls = []
    f1s = []

    for th in thresholds:
        preds = (probas >= th).astype(int)
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision (PPV)", color="#2b5c8f", linewidth=2.2)
    ax.plot(thresholds, recalls, label="Recall (Sensitivity)", color="#d9534f", linewidth=2.2)
    ax.plot(thresholds, f1s, label="F1-Score", color="#2ca02c", linewidth=2.2)

    best_idx = np.argmax(f1s)
    best_th = thresholds[best_idx]
    ax.axvline(best_th, color="black", linestyle=":", label=f"Max F1 Threshold ({best_th:.2f})", linewidth=1.5)

    ax.set_title(f"Operating Threshold Trade-Off ({model_name})", pad=12, fontweight="bold")
    ax.set_xlabel("Classification Probability Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="center right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(output_dir, "11_threshold_tradeoff.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def run_evaluation_pipeline() -> Dict[str, Any]:
    """Execute complete test evaluation, generate all diagnostic charts, and save metrics."""
    logger.info("Starting model evaluation on held-out test partition...")
    os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
    
    prep_data = prepare_full_pipeline_data()
    X_test = prep_data["X_test_trans"]
    y_test = np.array(prep_data["y_test"])
    
    models = load_trained_models()
    
    # Load CV summary to get winning threshold
    cv_summary_path = os.path.join(MODELS_DIR, "cv_results.json")
    if os.path.exists(cv_summary_path):
        with open(cv_summary_path, "r", encoding="utf-8") as f:
            cv_summary = json.load(f)
    else:
        cv_summary = {}

    test_results = evaluate_models_on_test_set(models, X_test, y_test, cv_summary)
    
    # Generate Figures
    plot_roc_curves(test_results, y_test, REPORTS_FIGURES_DIR)
    plot_pr_curves(test_results, y_test, REPORTS_FIGURES_DIR)
    plot_calibration_curves(test_results, y_test, REPORTS_FIGURES_DIR)
    
    winning_model = cv_summary.get("winning_model", "XGBoost")
    if winning_model in test_results:
        plot_confusion_matrices(test_results[winning_model], winning_model, REPORTS_FIGURES_DIR)
        plot_threshold_tradeoff(test_results[winning_model]["probas"], y_test, winning_model, REPORTS_FIGURES_DIR)

    # Save Test Evaluation Metrics JSON (excluding probability arrays)
    clean_test_metrics = {
        name: {k: v for k, v in res.items() if k != "probas"}
        for name, res in test_results.items()
    }
    
    metrics_path = os.path.join(MODELS_DIR, "test_evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(clean_test_metrics, f, indent=2)
    logger.info("Saved test evaluation metrics to %s", metrics_path)

    return clean_test_metrics


if __name__ == "__main__":
    eval_metrics = run_evaluation_pipeline()
    print("\n--- Held-Out Test Evaluation Results ---")
    print(json.dumps(eval_metrics, indent=2))
