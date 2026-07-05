@echo off
echo ============================================================
echo Air Quality Prediction Dashboard
echo ============================================================
echo.

echo Checking prerequisites...
python test_dashboard.py

echo.
echo ============================================================
echo Starting Streamlit Dashboard...
echo ============================================================
echo.
echo The dashboard will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.

streamlit run app.py

pause
