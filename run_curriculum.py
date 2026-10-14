"""
Master Runner Script for All Curriculum Modules (M3, M4, M5, M6)
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

SCRIPTS = [
    ("Module 3: Tree-Based Models", PROJECT_ROOT / "src" / "curriculum" / "m3_tree_models.py"),
    ("Module 4: Unsupervised Learning", PROJECT_ROOT / "src" / "curriculum" / "m4_unsupervised_learning.py"),
    ("Module 5: Model Evaluation & Calibration", PROJECT_ROOT / "src" / "curriculum" / "m5_evaluation_calibration.py"),
    ("Module 6: ML Engineering & Production", PROJECT_ROOT / "src" / "curriculum" / "m6_ml_engineering_production.py"),
]

def main():
    print("=" * 70)
    print(" EXECUTING CURRICULUM MODULES (M3, M4, M5, M6)")
    print("=" * 70)
    
    for title, script_path in SCRIPTS:
        print(f"\n>>> Running {title} ...")
        res = subprocess.run([PYTHON_EXE, str(script_path)])
        if res.returncode != 0:
            print(f"Error executing {title}")
            sys.exit(res.returncode)
            
    print("\n" + "=" * 70)
    print(" ALL MODULES COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
