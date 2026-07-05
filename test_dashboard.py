"""
Prerequisite checker for the Air Quality Prediction Dashboard.

run_dashboard.bat calls this script before launching Streamlit. It did not
exist in the repository, so the batch script always failed at this step.
This checks that required packages are importable and that the model
artifacts / data file the app needs are present, printing a clear pass/fail
summary instead of letting app.py fail deep inside a Streamlit session.
"""
import importlib
import os
import sys

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "torch",
    "plotly",
    "xgboost",
    "prophet",
    "sklearn",
    "joblib",
    "matplotlib",
]

REQUIRED_FILES = [
    "app.py",
    "uk_air_quality_data_complete.csv",
    os.path.join("Saved_Model", "config.json"),
    os.path.join("Saved_Model", "feature_cols.pkl"),
    os.path.join("Saved_Model", "scaler_X.pkl"),
    os.path.join("Saved_Model", "scaler_y.pkl"),
    os.path.join("Saved_Model", "bilstm_attention.pt"),
    os.path.join("Saved_Model", "xgb_pm25.json"),
    os.path.join("Saved_Model", "xgb_no2.json"),
    os.path.join("Saved_Model", "xgb_o3.json"),
    os.path.join("Saved_Model", "prophet_pm25.json"),
    os.path.join("Saved_Model", "prophet_no2.json"),
    os.path.join("Saved_Model", "prophet_o3.json"),
    os.path.join("Saved_Model", "ensemble_weights.npy"),
]


def check_packages():
    print("Checking required packages...")
    all_ok = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"  [OK]   {package}")
        except ImportError:
            print(f"  [FAIL] {package} is not installed")
            all_ok = False
    return all_ok


def check_files():
    print("\nChecking required files...")
    all_ok = True
    for path in REQUIRED_FILES:
        if os.path.exists(path):
            print(f"  [OK]   {path}")
        else:
            print(f"  [FAIL] {path} is missing")
            all_ok = False
    return all_ok


def main():
    packages_ok = check_packages()
    files_ok = check_files()

    print("\n" + "=" * 60)
    if packages_ok and files_ok:
        print("All prerequisites satisfied. Starting the dashboard...")
        sys.exit(0)
    else:
        print("One or more prerequisites are missing.")
        if not packages_ok:
            print("  -> Run: pip install -r requirements_dashboard.txt")
        if not files_ok:
            print("  -> See README.md 'Project Structure' for expected file locations.")
        # Exit non-zero, but run_dashboard.bat still attempts to start
        # Streamlit afterwards so the user sees Streamlit's own error too.
        sys.exit(1)


if __name__ == "__main__":
    main()
