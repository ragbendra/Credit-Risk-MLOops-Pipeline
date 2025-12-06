"""
PostgreSQL backup script.

Creates a backup of the MLflow PostgreSQL database and optionally
pushes it to DVC remote (Google Drive).

Usage:
    python scripts/backup_db.py
    python scripts/backup_db.py --push  # Also push to DVC remote

Requirements:
    - PostgreSQL client (pg_dump) installed
    - DVC configured with remote
"""

import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import os


# Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "mlflow")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlflow")
DB_NAME = os.getenv("DB_NAME", "mlflow")


def create_backup(output_dir: str = "backups") -> str:
    """
    Create a PostgreSQL backup using pg_dump.
    
    Args:
        output_dir: Directory to save backup files
        
    Returns:
        Path to the backup file
    """
    print("\n" + "="*50)
    print("💾 POSTGRESQL BACKUP")
    print("="*50 + "\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = output_path / f"mlflow_backup_{timestamp}.sql"
    
    print(f"🔄 Creating backup of database '{DB_NAME}'...")
    
    # Set password via environment variable
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD
    
    try:
        subprocess.run(
            [
                "pg_dump",
                "-h", DB_HOST,
                "-p", DB_PORT,
                "-U", DB_USER,
                "-d", DB_NAME,
                "-f", str(backup_file),
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True
        )
        
        # Get file size
        size_kb = backup_file.stat().st_size / 1024
        print(f"✅ Backup created: {backup_file} ({size_kb:.1f} KB)")
        
        return str(backup_file)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Backup failed: {e.stderr}")
        raise
    except FileNotFoundError:
        print("❌ pg_dump not found. Please install PostgreSQL client.")
        print("   On Windows: Install PostgreSQL or add pg_dump to PATH")
        print("   On Linux: sudo apt install postgresql-client")
        raise


def push_to_dvc(backup_file: str):
    """
    Add backup to DVC and push to remote.
    
    Args:
        backup_file: Path to the backup file
    """
    print("\n🔄 Pushing to DVC remote...")
    
    try:
        # Add to DVC
        subprocess.run(
            ["dvc", "add", backup_file],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Push to remote
        subprocess.run(
            ["dvc", "push"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Backup pushed to DVC remote")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  DVC push failed: {e.stderr}")
        print("   Backup is saved locally but not synced to remote")


def main():
    parser = argparse.ArgumentParser(description="Backup PostgreSQL database")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push backup to DVC remote after creating"
    )
    parser.add_argument(
        "--output-dir",
        default="backups",
        help="Directory to save backup files"
    )
    
    args = parser.parse_args()
    
    # Create backup
    backup_file = create_backup(args.output_dir)
    
    # Optionally push to DVC
    if args.push:
        push_to_dvc(backup_file)
    
    print("\n✅ Backup complete!\n")


if __name__ == "__main__":
    main()
