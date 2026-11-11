@echo off
setlocal

rem ============================================================
rem  The SigLIP2 fine-tune experiment - on THIS laptop's GPU.
rem  Teaches the last blocks of the picture model your tags,
rem  then makes every video's embeddings again with it (next to
rem  the old ones - nothing is replaced). No Modal credit.
rem
rem  Double-click: the check (does it fit, how long will it take).
rem  From a terminal:
rem      siglip_tune.bat check                 run this first
rem      siglip_tune.bat all --name ft1        cut pictures, train, embed
rem      siglip_tune.bat frames                just cut the pictures
rem      siglip_tune.bat train --name ft1      just train
rem      siglip_tune.bat embed --name ft1      just the embeddings
rem                                            (picks up where it stopped)
rem      ... --batch 8                         if the GPU runs out of memory
rem  Then:
rem      student_tags.bat train --name v16-ft1 --emb ft1
rem ============================================================

cd /d "%~dp0scripts"
set VENV=..\.venv

if not exist "%VENV%\Scripts\activate.bat" (
    echo Could not find the virtual environment at "%CD%\%VENV%".
    echo Run run_audio_analysis.bat once first - it sets up the venv with GPU torch.
    goto :end
)
for %%F in (siglip_tune.py student_tags.py trial_logic.py pack_for_modal.py modal_trial.py config.py) do (
    if not exist "%%F" (
        echo Missing required file: %%F - it must be in "%CD%"
        goto :end
    )
)

call "%VENV%\Scripts\activate.bat"
set PYTHONUTF8=1

python -c "import torch, transformers, cv2" 2>nul
if errorlevel 1 (
    echo This needs torch, transformers and opencv in the venv - run local_vision.bat once first.
    goto :end
)

echo.
if "%~1"=="" (
    python siglip_tune.py check
) else (
    python siglip_tune.py %*
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
