"""
Exploratory Data Analysis (EDA) Module
Credit Risk Prediction and Explainable Default Analytics

Generates high-resolution diagnostic visualizations and statistical summaries
saved to reports/figures/.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eda")

REPORTS_FIGURES_DIR = os.path.join("reports", "figures")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "clean_credit_card_data.csv")

# Color palette for consistent reporting
PALETTE = {"Non-Default": "#2b5c8f", "Default": "#d9534f"}
HEX_COLORS = ["#2b5c8f", "#d9534f", "#2ca02c", "#ff7f0e", "#9467bd"]


def setup_style():
    """Apply clean, publication-ready style for Matplotlib/Seaborn."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True,
        "figure.dpi": 300
    })


def plot_target_distribution(df: pd.DataFrame, output_dir: str = REPORTS_FIGURES_DIR):
    """Plot target class distribution and class imbalance."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    counts = df["default_payment_next_month"].value_counts()
    percentages = df["default_payment_next_month"].value_counts(normalize=True) * 100
    labels = ["Non-Default (0)", "Default (1)"]
    colors = [PALETTE["Non-Default"], PALETTE["Default"]]
    
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        height = bar.get_height()
        ax.annotate(f"{count:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax.set_title("Target Distribution: Next-Month Default Frequency", pad=15, fontweight="bold")
    ax.set_ylabel("Number of Cardholders")
    ax.set_ylim(0, max(counts.values) * 1.15)
    
    # Add textual note
    plt.figtext(0.5, -0.05, "Note: ~22.1% default rate reflects class imbalance (approx. 3.5:1 ratio). "
                           "Requires PR-AUC evaluation and cost-sensitive threshold tuning.",
                ha="center", fontsize=9, style="italic")
    
    save_path = os.path.join(output_dir, "01_target_distribution.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_demographics(df: pd.DataFrame, output_dir: str = REPORTS_FIGURES_DIR):
    """Plot default rates across demographic features: Sex, Education, Marriage."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Sex
    sex_labels = {1: "Male", 2: "Female"}
    sex_df = df.groupby("SEX")["default_payment_next_month"].agg(["mean", "count"]).reset_index()
    sex_df["SEX_LABEL"] = sex_df["SEX"].map(sex_labels)
    
    bars1 = axes[0].bar(sex_df["SEX_LABEL"], sex_df["mean"] * 100, color="#4682B4", edgecolor="black", width=0.5)
    for b, val in zip(bars1, sex_df["mean"] * 100):
        axes[0].annotate(f"{val:.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
    axes[0].set_title("Default Rate by Gender")
    axes[0].set_ylabel("Default Rate (%)")
    axes[0].set_ylim(0, 35)

    # 2. Education
    edu_labels = {1: "Grad School", 2: "University", 3: "High School", 4: "Others/Unk"}
    edu_df = df.groupby("EDUCATION")["default_payment_next_month"].agg(["mean", "count"]).reset_index()
    edu_df["EDU_LABEL"] = edu_df["EDUCATION"].map(edu_labels)
    
    bars2 = axes[1].bar(edu_df["EDU_LABEL"], edu_df["mean"] * 100, color="#5F9EA0", edgecolor="black", width=0.5)
    for b, val in zip(bars2, edu_df["mean"] * 100):
        axes[1].annotate(f"{val:.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
    axes[1].set_title("Default Rate by Education Level")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylim(0, 35)

    # 3. Marriage
    mar_labels = {1: "Married", 2: "Single", 3: "Others"}
    mar_df = df.groupby("MARRIAGE")["default_payment_next_month"].agg(["mean", "count"]).reset_index()
    mar_df["MAR_LABEL"] = mar_df["MARRIAGE"].map(mar_labels)
    
    bars3 = axes[2].bar(mar_df["MAR_LABEL"], mar_df["mean"] * 100, color="#6495ED", edgecolor="black", width=0.5)
    for b, val in zip(bars3, mar_df["mean"] * 100):
        axes[2].annotate(f"{val:.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
    axes[2].set_title("Default Rate by Marital Status")
    axes[2].set_ylabel("Default Rate (%)")
    axes[2].set_ylim(0, 35)
    
    fig.suptitle("Default Risk Across Demographic Cohorts", fontsize=14, fontweight="bold", y=1.02)
    save_path = os.path.join(output_dir, "02_demographic_default_rates.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_credit_limit_and_age(df: pd.DataFrame, output_dir: str = REPORTS_FIGURES_DIR):
    """Plot credit limit distributions and age group risk dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Credit Limit Boxplot by Target
    df_plot = df.copy()
    df_plot["Status"] = df_plot["default_payment_next_month"].map({0: "Non-Default", 1: "Default"})
    
    sns.boxplot(x="Status", y="LIMIT_BAL", data=df_plot, ax=axes[0],
                palette=PALETTE, width=0.4, fliersize=1)
    axes[0].set_title("Credit Limit (NT$) by Next-Month Default Status")
    axes[0].set_ylabel("Credit Limit (NT$)")
    axes[0].set_xlabel("")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    
    # 2. Default rate by Age Decile
    df_plot["Age_Group"] = pd.cut(df_plot["AGE"], bins=[18, 25, 30, 40, 50, 60, 80],
                                  labels=["18-25", "26-30", "31-40", "41-50", "51-60", "60+"])
    age_risk = df_plot.groupby("Age_Group", observed=False)["default_payment_next_month"].agg(["mean", "count"]).reset_index()
    
    bars = axes[1].bar(age_risk["Age_Group"].astype(str), age_risk["mean"] * 100, color="#d9534f", edgecolor="black", width=0.5)
    for b, val in zip(bars, age_risk["mean"] * 100):
        axes[1].annotate(f"{val:.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
        
    axes[1].set_title("Default Rate Across Age Cohorts")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].set_xlabel("Age Bracket (Years)")
    axes[1].set_ylim(0, 35)

    save_path = os.path.join(output_dir, "03_age_and_credit_limit_distribution.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_repayment_status(df: pd.DataFrame, output_dir: str = REPORTS_FIGURES_DIR):
    """Plot repayment status (PAY_0) distribution and default probability gradient."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Distribution of PAY_0
    pay_counts = df["PAY_0"].value_counts().sort_index()
    axes[0].bar(pay_counts.index.astype(str), pay_counts.values, color="#2b5c8f", edgecolor="black", width=0.6)
    axes[0].set_title("Distribution of Most Recent Repayment Status (PAY_0: Sept 2005)")
    axes[0].set_xlabel("Repayment Status Code (-2=No use, -1=Full, 0=Revolving, 1-8=Months Delay)")
    axes[0].set_ylabel("Cardholder Count")

    # 2. Default Rate by PAY_0
    pay_risk = df.groupby("PAY_0")["default_payment_next_month"].agg(["mean", "count"]).reset_index()
    # Filter out extremely rare status codes (>5 months delay) with <10 samples to prevent visual clutter
    pay_risk_filtered = pay_risk[pay_risk["count"] >= 10]
    
    bars = axes[1].bar(pay_risk_filtered["PAY_0"].astype(str), pay_risk_filtered["mean"] * 100,
                       color="#d9534f", edgecolor="black", width=0.6)
    for b, val in zip(bars, pay_risk_filtered["mean"] * 100):
        axes[1].annotate(f"{val:.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        
    axes[1].set_title("Default Rate by Repayment Status (PAY_0)")
    axes[1].set_xlabel("Repayment Status Code")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].set_ylim(0, 100)

    save_path = os.path.join(output_dir, "04_repayment_status_trends.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_bill_vs_pay(df: pd.DataFrame, output_dir: str = REPORTS_FIGURES_DIR):
    """Plot comparison of Bill Amounts and Payment Amounts distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log1p transforms for visualization of skewed financial values
    bill1_pos = np.log10(np.maximum(df["BILL_AMT1"], 1))
    pay1_pos = np.log10(np.maximum(df["PAY_AMT1"], 1))
    
    sns.histplot(bill1_pos, ax=axes[0], color="#2b5c8f", kde=True, bins=30)
    axes[0].set_title("Distribution of Log10(Bill Amount 1) [NT$]")
    axes[0].set_xlabel("Log10(Bill Amount in NT$)")
    axes[0].set_ylabel("Frequency")
    
    sns.histplot(pay1_pos, ax=axes[1], color="#2ca02c", kde=True, bins=30)
    axes[1].set_title("Distribution of Log10(Previous Payment Amount 1) [NT$]")
    axes[1].set_xlabel("Log10(Payment Amount in NT$)")
    axes[1].set_ylabel("Frequency")
    
    save_path = os.path.join(output_dir, "05_bill_vs_pay_amounts.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str = REPORTS_FIGURES_DIR):
    """Plot correlation heatmap with target and explain multicollinearity."""
    # Exclude ID
    cols = [c for c in df.columns if c != "ID"]
    corr = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap="vlag", vmin=-0.5, vmax=1.0, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.5, ax=ax)
    
    ax.set_title("Correlation Matrix of Predictive Features & Target (Pearson r)", pad=15, fontweight="bold")
    
    save_path = os.path.join(output_dir, "06_correlation_matrix.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


def run_eda(data_path: str = PROCESSED_DATA_PATH, output_dir: str = REPORTS_FIGURES_DIR):
    """Generate all figures and log summary statistics."""
    os.makedirs(output_dir, exist_ok=True)
    setup_style()
    
    logger.info("Loading cleaned data from %s", data_path)
    df = pd.read_csv(data_path)
    
    logger.info("Generating EDA figures...")
    plot_target_distribution(df, output_dir)
    plot_demographics(df, output_dir)
    plot_credit_limit_and_age(df, output_dir)
    plot_repayment_status(df, output_dir)
    plot_bill_vs_pay(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    logger.info("All EDA figures generated successfully in %s", output_dir)


if __name__ == "__main__":
    run_eda()
