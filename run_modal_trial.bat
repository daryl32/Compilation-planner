@echo off
setlocal

rem ============================================================
rem  The Modal GPU trial (modal_trial.py). Upload the pack first:
rem      pack_for_modal.bat --upload
rem
rem  Double-click: a SMOKE TEST - 1 video, first 90 s, a few cents -
rem  to check everything works before the real run.
rem
rem  From a terminal:
rem      run_modal_trial.bat --full                  the whole trial pack
rem      run_modal_trial.bat --stage report          just rebuild the report
rem      run_modal_trial.bat --stage tag --videos vid05,vid06
rem  The report is saved here as trial_report.md
rem  (trial_report_smoke.md for the smoke test).
rem ============================================================

cd /d "%~dp0"
set VENV=.venv

if not exist "%VENV%\Scripts\activate.bat" (
    echo Could not find the virtual environment at "%CD%\%VENV%".
    goto :end
)
for %%F in (modal_trial.py trial_logic.py) do (
    if not exist "%%F" (
        echo Missing required file: %%F - it must be in "%CD%"
        goto :end
    )
)

call "%VENV%\Scripts\activate.bat"
set PYTHONUTF8=1

where modal >nul 2>nul
if errorlevel 1 (
    echo Installing the modal command...
    python -m pip install -q modal
    echo.
    echo Now sign in to Modal - a browser window opens:
    modal setup
)

if "%~1"=="" (
    modal run modal_trial.py --smoke
) else if /i "%~1"=="--full" (
    modal run modal_trial.py
) else (
    modal run modal_trial.py %*
)
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" (
    echo Stopped with an error - exit code %EXITCODE%. Scroll up for details.
) else (
    echo Finished.
)

:end
echo.
pause
endlocal
