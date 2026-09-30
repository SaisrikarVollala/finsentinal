"""
Data Ingestion and Schema Validation Module
Credit Risk Prediction and Explainable Default Analytics

This module handles dataset acquisition, schema verification, data type coercion,
cleaning, and raw data persistence for the UCI Default of Credit Card Clients dataset.
"""

import os
import io
import json
import logging
import zipfile
import urllib.request
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_loader")

# Constants
UCI_DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
DATA_DICTIONARY_PATH = os.path.join("data", "data_dictionary.json")

# Standardized column mapping from raw UCI file
# Raw files often have 'default payment next month' or 'default.payment.next.month'
# and PAY_0 is occasionally labeled PAY_1 or PAY_0.
EXPECTED_RAW_COLUMNS = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default_payment_next_month"
]

TARGET_COLUMN = "default_payment_next_month"
ID_COLUMN = "ID"

DATA_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "LIMIT_BAL": {
        "description": "Amount of given credit in New Taiwan dollars (NT$), including individual and family/supplementary credit.",
        "type": "numerical",
        "unit": "NT$",
        "valid_range": "[10000, 1000000]"
    },
    "SEX": {
        "description": "Gender of the cardholder.",
        "type": "categorical",
        "categories": {1: "Male", 2: "Female"}
    },
    "EDUCATION": {
        "description": "Education level of the cardholder.",
        "type": "categorical",
        "categories": {
            1: "Graduate School",
            2: "University",
            3: "High School",
            4: "Others",
            5: "Unknown",
            6: "Unknown",
            0: "Unknown"
        },
        "cleaned_categories": {1: "Graduate School", 2: "University", 3: "High School", 4: "Others/Unknown"}
    },
    "MARRIAGE": {
        "description": "Marital status of the cardholder.",
        "type": "categorical",
        "categories": {1: "Married", 2: "Single", 3: "Others", 0: "Unknown"},
        "cleaned_categories": {1: "Married", 2: "Single", 3: "Others"}
    },
    "AGE": {
        "description": "Age of the cardholder in years.",
        "type": "numerical",
        "unit": "years",
        "valid_range": "[18, 100]"
    },
    "PAY_0": {
        "description": "Repayment status in September 2005 (most recent month).",
        "type": "categorical_ordinal",
        "scale": "-2=No consumption/balance, -1=Paid in full, 0=Revolving credit used (min payment), 1=Delay 1 month, 2=Delay 2 months, ..., 8=Delay 8 months, 9=Delay 9+ months"
    },
    "PAY_2": {
        "description": "Repayment status in August 2005.",
        "type": "categorical_ordinal",
        "scale": "-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed"
    },
    "PAY_3": {
        "description": "Repayment status in July 2005.",
        "type": "categorical_ordinal",
        "scale": "-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed"
    },
    "PAY_4": {
        "description": "Repayment status in June 2005.",
        "type": "categorical_ordinal",
        "scale": "-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed"
    },
    "PAY_5": {
        "description": "Repayment status in May 2005.",
        "type": "categorical_ordinal",
        "scale": "-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed"
    },
    "PAY_6": {
        "description": "Repayment status in April 2005 (oldest month).",
        "type": "categorical_ordinal",
        "scale": "-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-9=Months delayed"
    },
    "BILL_AMT1": {
        "description": "Amount of bill statement in September 2005 (NT$).",
        "type": "numerical",
        "unit": "NT$"
    },
    "BILL_AMT2": {
        "description": "Amount of bill statement in August 2005 (NT$).",
        "type": "numerical",
        "unit": "NT$"
    },
    "BILL_AMT3": {
        "description": "Amount of bill statement in July 2005 (NT$).",
        "type": "numerical",
        "unit": "NT$"
    },
    "BILL_AMT4": {
        "description": "Amount of bill statement in June 2005 (NT$).",
        "type": "numerical",
        "unit": "NT$"
    },
    "BILL_AMT5": {
        "description": "Amount of bill statement in May 2005 (NT$).",
        "type": "numerical",
        "unit": "NT$"
    },
    "BILL_AMT6": {
        "description": "Amount of bill statement in April 2005 (NT$).",
        "type": "numerical",
        "unit": "NT$"
    },
    "PAY_AMT1": {
        "description": "Amount paid in September 2005 (towards August statement) in NT$.",
        "type": "numerical",
        "unit": "NT$"
    },
    "PAY_AMT2": {
        "description": "Amount paid in August 2005 (towards July statement) in NT$.",
        "type": "numerical",
        "unit": "NT$"
    },
    "PAY_AMT3": {
        "description": "Amount paid in July 2005 (towards June statement) in NT$.",
        "type": "numerical",
        "unit": "NT$"
    },
    "PAY_AMT4": {
        "description": "Amount paid in June 2005 (towards May statement) in NT$.",
        "type": "numerical",
        "unit": "NT$"
    },
    "PAY_AMT5": {
        "description": "Amount paid in May 2005 (towards April statement) in NT$.",
        "type": "numerical",
        "unit": "NT$"
    },
    "PAY_AMT6": {
        "description": "Amount paid in April 2005 (towards March statement) in NT$.",
        "type": "numerical",
        "unit": "NT$"
    },
    "default_payment_next_month": {
        "description": "Binary target: Did the customer default on the next month's payment (October 2005)?",
        "type": "binary_target",
        "values": {0: "Non-default", 1: "Default"}
    }
}


def ensure_directories():
    """Ensure data/raw and data/processed directories exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_DICTIONARY_PATH), exist_ok=True)


def download_uci_dataset(dest_dir: str = RAW_DATA_DIR, force: bool = False) -> str:
    """
    Download the UCI Default of Credit Card Clients dataset zip file and extract the Excel file.

    Parameters
    ----------
    dest_dir : str
        Directory to store the raw file.
    force : bool
        If True, re-downloads even if the file exists.

    Returns
    -------
    str
        Path to the extracted raw data file (XLS/XLSX or CSV).
    """
    ensure_directories()
    
    # Check if raw files already exist in directory
    possible_files = [
        os.path.join(dest_dir, "default of credit card clients.xls"),
        os.path.join(dest_dir, "default of credit card clients.xlsx"),
        os.path.join(dest_dir, "default_of_credit_card_clients.xls"),
        os.path.join(dest_dir, "default_of_credit_card_clients.xlsx"),
        os.path.join(dest_dir, "default_of_credit_card_clients.csv"),
        os.path.join(dest_dir, "default_credit_card.csv")
    ]
    
    if not force:
        for f in possible_files:
            if os.path.exists(f) and os.path.getsize(f) > 1000:
                logger.info("Found existing raw dataset at: %s", f)
                return f

    zip_dest = os.path.join(dest_dir, "default_of_credit_card_clients.zip")
    logger.info("Downloading dataset from UCI repository: %s", UCI_DATASET_ZIP_URL)
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(UCI_DATASET_ZIP_URL, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            zip_content = response.read()
            with open(zip_dest, "wb") as f:
                f.write(zip_content)
        logger.info("Downloaded %d bytes to %s", len(zip_content), zip_dest)
        
        # Extract the zip
        with zipfile.ZipFile(zip_dest, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
            extracted_files = zip_ref.namelist()
            logger.info("Extracted files: %s", extracted_files)
            
        for f in possible_files:
            if os.path.exists(f):
                return f
        
        # Return first extracted xls/xlsx/csv
        for name in extracted_files:
            full_p = os.path.join(dest_dir, name)
            if os.path.isfile(full_p) and (name.endswith(".xls") or name.endswith(".xlsx") or name.endswith(".csv")):
                return full_p
                
    except Exception as e:
        logger.warning("UCI Zip download encountered error: %s. Attempting direct fallback mirrors...", e)
        
    # Secondary direct mirror from GitHub raw archives of the canonical dataset if UCI CDN is slow/down
    fallback_urls = [
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/credit_card/default%20of%20credit%20card%20clients.xls",
        "https://raw.githubusercontent.com/selva86/datasets/master/Default.csv"
    ]
    
    for fb_url in fallback_urls:
        try:
            logger.info("Attempting fallback URL: %s", fb_url)
            fb_req = urllib.request.Request(fb_url, headers=headers)
            with urllib.request.urlopen(fb_req, timeout=20) as fb_resp:
                fb_data = fb_resp.read()
                filename = os.path.basename(fb_url.replace("%20", "_"))
                target_path = os.path.join(dest_dir, filename)
                with open(target_path, "wb") as f:
                    f.write(fb_data)
                logger.info("Saved fallback dataset to: %s", target_path)
                return target_path
        except Exception as fb_err:
            logger.warning("Fallback failed for %s: %s", fb_url, fb_err)
            
    raise RuntimeError(
        "Could not automatically download the UCI dataset. Please manually place "
        "'default of credit card clients.xls' or '.csv' into the 'data/raw/' directory."
    )


def load_raw_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw UCI dataset from file with format handling and header standardization.

    Parameters
    ----------
    file_path : Optional[str]
        Path to file. If None, auto-discovers or downloads it.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame with standardized column names.
    """
    if file_path is None or not os.path.exists(file_path):
        file_path = download_uci_dataset()

    logger.info("Loading raw dataset from %s", file_path)
    
    if file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        # The UCI Excel file has an intro row on row 0 (X1...X24, Y), and actual column names on row 1
        # Read header=1 to get actual column names
        try:
            df = pd.read_excel(file_path, header=1)
            # Verify if column names match
            if "LIMIT_BAL" not in df.columns and "X1" in df.columns:
                df = pd.read_excel(file_path, header=0)
        except Exception:
            # Fallback to header=0
            df = pd.read_excel(file_path, header=0)
    elif file_path.endswith(".csv"):
        # Detect header row in CSV
        df = pd.read_csv(file_path)
        if "LIMIT_BAL" not in df.columns:
            # Try skiprows=1 if row 0 was X1...X24
            df_alt = pd.read_csv(file_path, skiprows=1)
            if "LIMIT_BAL" in df_alt.columns:
                df = df_alt
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Standardize column names
    clean_cols = {}
    for col in df.columns:
        c_str = str(col).strip()
        if c_str in ["PAY_0", "PAY_1", "X6"]:
            clean_cols[col] = "PAY_0"
        elif c_str.lower() in ["default payment next month", "default.payment.next.month", "y", "default"]:
            clean_cols[col] = "default_payment_next_month"
        elif c_str.startswith("Unnamed:"):
            pass
        else:
            clean_cols[col] = c_str

    df = df.rename(columns=clean_cols)
    
    # Remove any completely empty unnamed columns
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
        
    logger.info("Loaded raw DataFrame with shape: %s", df.shape)
    return df


def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate schema, missing values, duplicates, and perform initial integrity cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned dataframe and a validation summary report.
    """
    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "initial_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "validation_passed": True,
        "warnings": []
    }
    
    clean_df = df.copy()

    # 1. Deduplicate if exact duplicate rows exist
    if report["duplicate_rows"] > 0:
        logger.warning("Found %d duplicate rows. Removing duplicates.", report["duplicate_rows"])
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)
        report["warnings"].append(f"Removed {report['duplicate_rows']} duplicate records.")

    # 2. Verify target column exists
    if TARGET_COLUMN not in clean_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset. Columns present: {list(clean_df.columns)}")

    # 3. Handle ID column
    if ID_COLUMN in clean_df.columns:
        # Check uniqueness of ID
        id_unique = clean_df[ID_COLUMN].nunique() == len(clean_df)
        report["id_unique"] = id_unique
        if not id_unique:
            report["warnings"].append("Customer ID column contains non-unique values.")

    # 4. Standardize Categorical Codes
    # In UCI dataset:
    # EDUCATION: 1=grad school, 2=university, 3=high school, 4=others. Values 0, 5, 6 are undocumented/unknown.
    # Group 0, 4, 5, 6 into 4 ("Others/Unknown").
    if "EDUCATION" in clean_df.columns:
        clean_df["EDUCATION"] = clean_df["EDUCATION"].apply(lambda x: 4 if x in [0, 4, 5, 6] else x)

    # MARRIAGE: 1=married, 2=single, 3=others. Value 0 is undocumented/unknown -> group into 3 ("Others").
    if "MARRIAGE" in clean_df.columns:
        clean_df["MARRIAGE"] = clean_df["MARRIAGE"].apply(lambda x: 3 if x in [0, 3] else x)

    # SEX: 1=male, 2=female. Validate domain.
    if "SEX" in clean_df.columns:
        invalid_sex = ~clean_df["SEX"].isin([1, 2])
        if invalid_sex.sum() > 0:
            report["warnings"].append(f"Found {invalid_sex.sum()} records with unexpected SEX values.")

    # 5. Convert numeric types explicitly
    numeric_cols = [
        "LIMIT_BAL", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]
    
    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df[TARGET_COLUMN] = pd.to_numeric(clean_df[TARGET_COLUMN], errors="coerce").astype(int)

    # 6. Check target distribution
    target_counts = clean_df[TARGET_COLUMN].value_counts().to_dict()
    default_rate = clean_df[TARGET_COLUMN].mean()
    report["target_distribution"] = {
        "non_default_count": int(target_counts.get(0, 0)),
        "default_count": int(target_counts.get(1, 0)),
        "default_rate": float(default_rate)
    }
    
    report["final_rows"] = len(clean_df)
    report["final_columns"] = len(clean_df.columns)
    
    logger.info("Validation complete. Final shape: %s, Default rate: %.2f%%",
                clean_df.shape, default_rate * 100)
    
    return clean_df, report


def save_data_dictionary(file_path: str = DATA_DICTIONARY_PATH):
    """Save the comprehensive data dictionary to JSON and Markdown."""
    ensure_directories()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(DATA_DICTIONARY, f, indent=2)
        
    md_path = file_path.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Credit Risk Prediction - Dataset Data Dictionary\n\n")
        f.write("Dataset: **UCI Default of Credit Card Clients** (Yeh & Lien, 2009)\n")
        f.write("Task: Predict next-month default (`default_payment_next_month`) for existing credit-card cardholders in Taiwan.\n\n")
        f.write("| Column Name | Type | Description | Values / Scale |\n")
        f.write("|-------------|------|-------------|----------------|\n")
        for col, info in DATA_DICTIONARY.items():
            desc = info.get("description", "")
            col_type = info.get("type", "")
            scale = info.get("scale") or str(info.get("categories") or info.get("values") or info.get("valid_range") or info.get("unit") or "")
            scale_escaped = str(scale).replace("\n", " ")
            f.write(f"| `{col}` | {col_type} | {desc} | {scale_escaped} |\n")
            
    logger.info("Saved data dictionary to %s and %s", file_path, md_path)


def run_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute end-to-end ingestion, validation, and save clean dataset."""
    ensure_directories()
    save_data_dictionary()
    
    raw_path = download_uci_dataset()
    df_raw = load_raw_dataset(raw_path)
    
    # Save raw CSV copy
    raw_csv_path = os.path.join(RAW_DATA_DIR, "default_of_credit_card_clients.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    logger.info("Saved raw dataset copy to: %s", raw_csv_path)
    
    clean_df, report = validate_and_clean_data(df_raw)
    
    # Save processed clean CSV
    clean_csv_path = os.path.join(PROCESSED_DATA_DIR, "clean_credit_card_data.csv")
    clean_df.to_csv(clean_csv_path, index=False)
    logger.info("Saved validated dataset to: %s", clean_csv_path)
    
    return clean_df, report


if __name__ == "__main__":
    df_clean, validation_summary = run_pipeline()
    print("\n--- Data Ingestion and Validation Summary ---")
    print(json.dumps(validation_summary, indent=2))
