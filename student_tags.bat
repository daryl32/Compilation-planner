@echo off
setlocal

rem ============================================================
rem  The student tag model - trained on THIS laptop's GPU.
rem  No Modal GPU credit is used.
rem
rem  Double-click: fetch what's new from Modal (a free download),
rem  copy your latest labels, train model v1 and score it on the
rem  test set. The report is saved here as student_report.md
rem
rem  First press "Save edits to Drive" in the app so your newest
rem  labels are on Drive.
rem
rem  From a terminal you can pass options through, e.g.:
rem      student_tags.bat fetch                     just download
rem      student_tags.bat train --name v1b          train another
rem      student_tags.bat train --label-weight 8    labels count more
rem      student_tags.bat score --name v1           score again
rem ============================================================

cd /d "%~dp0"
set VENV=.venv

if not exist "%VENV%\Scripts\activate.bat" (
    echo Could not find the virtual environment at "%CD%\%VENV%".
    echo Run run_audio_analysis.bat once first - it sets up the venv with GPU torch.
    goto :end
)
for %%F in (student_tags.py trial_logic.py pack_for_modal.py config.py) do (
    if not exist "%%F" (
        echo Missing required file: %%F - it must be in "%CD%"
        goto :end
    )
)

call "%VENV%\Scripts\activate.bat"
set PYTHONUTF8=1

python -c "import torch, modal" 2>nul
if errorlevel 1 (
    echo Installing modal into the venv...
    python -m pip install --quiet modal
)
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none - training will use the CPU')"

echo.
if "%~1"=="" (
    python student_tags.py all
) else (
    python student_tags.py %*
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
