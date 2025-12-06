"""
Tests for data preprocessing.

Run with: pytest tests/test_preprocessing.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataDownload:
    """Tests for data download functionality."""
    
    def test_raw_data_exists(self):
        """Raw data file should exist after download."""
        raw_path = Path("data/raw/german.csv")
        
        if not raw_path.exists():
            pytest.skip("Run 'python src/data/download.py' first")
        
        assert raw_path.exists()
    
    def test_raw_data_has_correct_columns(self):
        """Raw data should have 21 columns."""
        raw_path = Path("data/raw/german.csv")
        
        if not raw_path.exists():
            pytest.skip("Run 'python src/data/download.py' first")
        
        df = pd.read_csv(raw_path)
        assert len(df.columns) == 21
    
    def test_raw_data_has_1000_rows(self):
        """Raw data should have 1000 rows."""
        raw_path = Path("data/raw/german.csv")
        
        if not raw_path.exists():
            pytest.skip("Run 'python src/data/download.py' first")
        
        df = pd.read_csv(raw_path)
        assert len(df) == 1000


class TestPreprocessing:
    """Tests for preprocessing pipeline."""
    
    @pytest.fixture
    def processed_data(self):
        """Load processed data if available."""
        train_path = Path("data/processed/train.csv")
        test_path = Path("data/processed/test.csv")
        
        if not train_path.exists() or not test_path.exists():
            pytest.skip("Run 'python src/data/preprocess.py' first")
        
        return pd.read_csv(train_path), pd.read_csv(test_path)
    
    def test_train_test_split_ratio(self, processed_data):
        """Train/test split should be approximately 80/20."""
        train, test = processed_data
        total = len(train) + len(test)
        
        train_ratio = len(train) / total
        assert 0.75 <= train_ratio <= 0.85, f"Train ratio {train_ratio} not in expected range"
    
    def test_no_missing_values(self, processed_data):
        """Processed data should have no missing values."""
        train, test = processed_data
        
        assert train.isna().sum().sum() == 0, "Train has missing values"
        assert test.isna().sum().sum() == 0, "Test has missing values"
    
    def test_target_is_binary(self, processed_data):
        """Target should be binary (0 or 1)."""
        train, test = processed_data
        
        assert set(train["risk"].unique()).issubset({0, 1})
        assert set(test["risk"].unique()).issubset({0, 1})
    
    def test_stratified_split(self, processed_data):
        """Train and test should have similar class distributions."""
        train, test = processed_data
        
        train_ratio = train["risk"].mean()
        test_ratio = test["risk"].mean()
        
        # Should be within 5% of each other
        assert abs(train_ratio - test_ratio) < 0.05
    
    def test_features_are_numeric(self, processed_data):
        """All features should be numeric after encoding."""
        train, _ = processed_data
        
        for col in train.columns:
            assert pd.api.types.is_numeric_dtype(train[col]), f"{col} is not numeric"
