@echo off
echo Activating virtual environment...
call .venv\Scripts\Activate.bat

echo Starting Laptop Price Prediction App...
streamlit run app.py

pause



