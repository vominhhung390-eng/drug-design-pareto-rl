@echo off
setlocal
cd /d "%~dp0"
echo Alternate predictor condition: retrain EGFR/VEGFR2 from provenance-marked recovered rows.
echo This is not the formal historical-oracle condition used by the default launcher.
pwsh -NoProfile -ExecutionPolicy Bypass -File ".\scripts\reproduce_all.ps1" -Stage All -AllowRecoveredEgfrVegfr2
if errorlevel 1 (
  echo.
  echo Reproduction stopped. See results\reproduction\preflight.json and logs.
  pause
  exit /b 1
)
echo.
echo Reproduction completed.
pause
