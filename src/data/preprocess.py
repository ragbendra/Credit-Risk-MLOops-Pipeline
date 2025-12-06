"""
Feature engineering and preprocessing pipeline.

This script handles:
- Loading raw data
- Encoding categorical variables
- Scaling numerical features
- Train/test split
- Saving processed data

Usage:
    python src/data/preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# Numerical columns to scale
NUMERICAL_COLS = ["duration", "amount", "age", "installment_rate", "residence", "existing_credits", "dependents"]

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/test split ratio
TEST_SIZE = 0.2


def load_raw_data(path: str = "data/raw/german.csv") -> pd.DataFrame:
    """Load raw data from CSV."""
    df = pd.read_csv(path)
    print(f"📂 Loaded {len(df)} records from {path}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode target variable.
    Original: 1 = Good, 2 = Bad
    Encoded: 0 = Good, 1 = Bad (for binary classification)
    """
    df = df.copy()
    df["risk"] = df["risk"].map({1: 0, 2: 1})
    print(f"🎯 Target encoded: Good(0)={len(df[df['risk']==0])}, Bad(1)={len(df[df['risk']==1])}")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns using LabelEncoder.
    
    Returns:
        Tuple of (encoded DataFrame, dict of encoders)
    """
    df = df.copy()
    encoders = {}
    
    # Identify categorical columns (non-numerical, excluding target)
    categorical_cols = [col for col in df.columns 
                       if col not in NUMERICAL_COLS + ["risk"]]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    print(f"🏷️  Encoded {len(categorical_cols)} categorical columns")
    return df, encoders


def scale_numericals(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numerical columns using StandardScaler.
    
    Fits on training data, transforms both train and test.
    """
    scaler = StandardScaler()
    
    # Fit and transform training data
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])
    
    print(f"📏 Scaled {len(NUMERICAL_COLS)} numerical columns")
    return X_train, X_test, scaler


def preprocess(
    input_path: str = "data/raw/german.csv",
    output_dir: str = "data/processed",
    save_artifacts: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline.
    
    Args:
        input_path: Path to raw data CSV
        output_dir: Directory to save processed data
        save_artifacts: Whether to save scaler and encoders
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print("\n" + "="*50)
    print("🔧 PREPROCESSING PIPELINE")
    print("="*50 + "\n")
    
    # 1. Load data
    df = load_raw_data(input_path)
    
    # 2. Encode target
    df = encode_target(df)
    
    # 3. Encode categoricals
    df, encoders = encode_categoricals(df)
    
    # 4. Split features and target
    X = df.drop("risk", axis=1)
    y = df["risk"]
    
    # 5. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    print(f"✂️  Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # 6. Scale numerical features
    X_train, X_test, scaler = scale_numericals(X_train, X_test)
    
    # 7. Combine features and target
    train_df = pd.concat([X_train.reset_index(drop=True), 
                          y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), 
                         y_test.reset_index(drop=True)], axis=1)
    
    # 8. Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    print(f"💾 Saved train.csv ({len(train_df)} rows) and test.csv ({len(test_df)} rows)")
    
    # 9. Save artifacts (scaler, encoders)
    if save_artifacts:
        artifacts_path = Path("models")
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(scaler, artifacts_path / "scaler.pkl")
        joblib.dump(encoders, artifacts_path / "encoders.pkl")
        print(f"💾 Saved preprocessing artifacts to models/")
    
    print("\n✅ Preprocessing complete!\n")
    return train_df, test_df


if __name__ == "__main__":
    preprocess()
