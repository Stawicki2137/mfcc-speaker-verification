import sys
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ui.app import run_app

if __name__ == "__main__":
    run_app()