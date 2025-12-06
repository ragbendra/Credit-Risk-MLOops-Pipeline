"""
Generate Evidently drift report.

This script generates a manual drift report comparing training data
with recent predictions or test data.

Usage:
    python scripts/generate_report.py
    
Output:
    reports/drift_report_YYYYMMDD_HHMMSS.html
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset


def generate_drift_report(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/processed/test.csv",
    output_dir: str = "reports"
) -> str:
    """
    Generate an Evidently drift report.
    
    Args:
        reference_path: Path to reference data (training data)
        current_path: Path to current data (test or production data)
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    print("\n" + "="*50)
    print("📊 GENERATING DRIFT REPORT")
    print("="*50 + "\n")
    
    # Load data
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    
    print(f"📂 Reference data: {len(reference)} rows")
    print(f"📂 Current data: {len(current)} rows")
    
    # Create report
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    # Run report
    print("🔄 Running drift analysis...")
    report.run(
        reference_data=reference,
        current_data=current,
    )
    
    # Save HTML report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"drift_report_{timestamp}.html"
    
    report.save_html(str(report_file))
    
    print(f"\n✅ Report saved: {report_file}")
    print("   Open in browser to view drift analysis")
    
    # Print summary
    report_dict = report.as_dict()
    if "metrics" in report_dict:
        for metric in report_dict["metrics"]:
            if "result" in metric:
                result = metric["result"]
                if "number_of_drifted_columns" in result:
                    drifted = result["number_of_drifted_columns"]
                    total = result["number_of_columns"]
                    print(f"\n📈 Drift Summary: {drifted}/{total} columns drifted")
    
    return str(report_file)


if __name__ == "__main__":
    generate_drift_report()
