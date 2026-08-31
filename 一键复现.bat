@echo off
setlocal
cd /d "%~dp0"
pwsh -NoProfile -ExecutionPolicy Bypass -File ".\scripts\reproduce_all.ps1" -Stage All
if errorlevel 1 (
  echo.
  echo Reproduction stopped. See results\reproduction\preflight.json and logs.
  pause
  exit /b 1
)
echo.
echo Reproduction completed.
pause
