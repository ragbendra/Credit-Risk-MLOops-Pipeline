"""
Download UCI German Credit Dataset.

This script downloads the German Credit dataset from the UCI ML Repository
and saves it as a CSV file for further processing.

Usage:
    python src/data/download.py
"""

import pandas as pd
from pathlib import Path


# UCI German Credit Dataset URL
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column names for the dataset (original attribute names)
COLUMNS = [
    "status",           # Status of existing checking account
    "duration",         # Duration in months
    "credit_history",   # Credit history
    "purpose",          # Purpose of the loan
    "amount",           # Credit amount
    "savings",          # Savings account/bonds
    "employment",       # Present employment since
    "installment_rate", # Installment rate in percentage of disposable income
    "personal_status",  # Personal status and sex
    "other_debtors",    # Other debtors / guarantors
    "residence",        # Present residence since
    "property",         # Property
    "age",              # Age in years
    "other_plans",      # Other installment plans
    "housing",          # Housing
    "existing_credits", # Number of existing credits at this bank
    "job",              # Job
    "dependents",       # Number of people being liable to provide maintenance for
    "telephone",        # Telephone
    "foreign_worker",   # Foreign worker
    "risk",             # Credit risk (1 = Good, 2 = Bad)
]


def download_dataset(output_path: str = "data/raw/german.csv") -> pd.DataFrame:
    """
    Download the UCI German Credit dataset.
    
    Args:
        output_path: Path to save the downloaded CSV file.
        
    Returns:
        DataFrame with the downloaded data.
    """
    print("📥 Downloading UCI German Credit Dataset...")
    
    # Read the space-separated data file
    df = pd.read_csv(DATA_URL, sep=" ", header=None, names=COLUMNS)
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"✅ Downloaded {len(df)} records to {output_path}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Risk distribution: Good={len(df[df['risk']==1])}, Bad={len(df[df['risk']==2])}")
    
    return df


if __name__ == "__main__":
    download_dataset()
