@echo off
set SCRIPT_DIR=%~dp0
set MLOPS_DIR=%SCRIPT_DIR%..
cd /d "%MLOPS_DIR%"
set AUREUS_MODEL_PATH=%MLOPS_DIR%\models\fraud_detection_model.pkl
if not exist "%AUREUS_MODEL_PATH%" (
  echo Model not found. Run run_train.bat first.
  exit /b 1
)
echo Starting ML serving on http://0.0.0.0:8000
python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
pause
