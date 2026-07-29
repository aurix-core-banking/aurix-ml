@echo off
set SCRIPT_DIR=%~dp0
set MLOPS_DIR=%SCRIPT_DIR%..
cd /d "%MLOPS_DIR%"
if not exist models mkdir models
if not exist artifacts mkdir artifacts
echo Running training pipeline...
python -m pipelines.train_pipeline --config config/config.yaml --model-dir models
echo Training finished. Model saved to %MLOPS_DIR%\models
pause
