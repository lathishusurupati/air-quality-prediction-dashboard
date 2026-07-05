# Real-Time Air Quality Prediction and Health Impact Analysis

A Streamlit dashboard that forecasts UK air quality (PM2.5, NO2, O3) using an
ensemble of BiLSTM + Attention, XGBoost, and Prophet models, and estimates the
associated health impact.

## Features

- Multi-model ensemble forecasting (BiLSTM-Attention, XGBoost, Prophet)
- Interactive Plotly visualizations of predictions and trends
- Health impact estimation based on relative-risk coefficients
- Configurable email alerting when air quality crosses thresholds

## Project Structure

```
.
├── app.py                          # Streamlit dashboard entry point
├── alert_system.py                 # Email alert / notification logic
├── alert_config_example.json       # Template config (copy to alert_config.json)
├── requirements_dashboard.txt      # Python dependencies
├── run_dashboard.bat               # Windows launcher script
├── uk_air_quality_data_complete.csv# Historical training/inference data
├── V1_Final_Code.ipynb             # Model development / training notebook
├── System_Design.png               # Architecture diagram
├── Instruction.md                  # Project brief / assignment writeup
└── Saved_Model/                    # Pretrained model artifacts
    ├── bilstm_attention.pt
    ├── xgb_no2.json / xgb_o3.json / xgb_pm25.json
    ├── prophet_no2.json / prophet_o3.json / prophet_pm25.json
    ├── scaler_X.pkl / scaler_y.pkl
    ├── feature_cols.pkl
    ├── ensemble_weights.npy
    └── config.json
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements_dashboard.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

On Windows, you can also double-click / run `run_dashboard.bat`.

## Email Alerts (optional)

Copy `alert_config_example.json` to `alert_config.json` and fill in your own
SMTP credentials (e.g. a Gmail App Password) — do **not** commit this file,
it's already excluded via `.gitignore`.

## Notes

This repository was produced as part of module **7151CEM – Computing and
Individual Research Project**. See `Instruction.md` for the full project
brief.
