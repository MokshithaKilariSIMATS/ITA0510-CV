"""Task 1 entry point for Computer Vision CO1.
Run the complete reproducible implementation from the repository root:
    python run_all.py
Task-specific outputs are stored in this folder's Output directory.
"""
from pathlib import Path
import runpy
ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "run_all.py"), run_name="__main__")
