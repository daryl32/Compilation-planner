@echo off
setlocal

rem ============================================================
rem  New videos: the Modal vision stage on THIS laptop's GPU.
rem  Scene cuts, body pose -> movement curve + rhythm, picture
rem  embeddings, then the student's tags every half second.
rem  No Modal credit, no teacher.
rem
rem  Run your usual local pipeline first (catalogue + preview copy).
rem
rem  Double-click: lists the videos that have no vision data yet.
rem  From a terminal:
rem      local_vision.bat run --newest 5        the 5 newest of them
rem      local_vision.bat run --only vidA vidB  just these
rem      local_vision.bat run --newest 5 --model v5b
rem                                             score with student v5b
rem      local_vision.bat check                 one Modal video re-run
rem                                             here and compared
rem ============================================================

cd /d "%~dp0"
set VENV=.venv

if not exist "%VENV%\Scripts\activate.bat" (
    echo Could not find the virtual environment at "%CD%\%VENV%".
    echo Run run_audio_analysis.bat once first - it sets up the venv with GPU torch.
    goto :end
)
for %%F in (local_vision.py student_tags.py trial_logic.py pack_for_modal.py modal_trial.py config.py) do (
    if not exist "%%F" (
        echo Missing required file: %%F - it must be in "%CD%"
        goto :end
    )
)

call "%VENV%\Scripts\activate.bat"
set PYTHONUTF8=1

python -c "import torch, modal, ultralytics, lap, transnetv2_pytorch, transformers, cv2" 2>nul
if errorlevel 1 (
    echo Installing the vision packages into the venv (once^)...
    python -m pip install --quiet modal ultralytics lap transnetv2-pytorch "transformers>=4.57" opencv-python-headless huggingface_hub
)
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none - this needs the GPU')"

echo.
if "%~1"=="" (
    python local_vision.py list
) else (
    python local_vision.py %*
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
