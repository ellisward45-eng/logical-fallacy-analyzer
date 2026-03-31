@echo off
cd /d "%~dp0"
echo Running smoke test...
python smoke_test.py
echo.
echo Done. Press any key to close.
pause >nul