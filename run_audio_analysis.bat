@echo off
setlocal

rem ============================================================
rem  Local audio analysis - does whatever each track is missing.
rem
rem  Double-click it. It:
rem    1. makes the virtual environment if there isn't one,
rem    2. checks and installs everything the full analysis needs
rem       (numpy, scipy, GPU torch + matching torchaudio, beat_this,
rem       Demucs, ffmpeg) - only downloads what's missing or broken,
rem    3. goes through every track in Audio-Library on the GPU:
rem         new tracks   - the full analysis (beats, energy, downbeats,
rem                        sections, drops, vocals, key) + stems
rem         older tracks - only what they're missing (extra analysis,
rem                        vocals, key, stems), keeping their beats
rem       Stems (vocals + instrumental, for mashups) are saved in
rem       scene-labeling\stems\ in your Google Drive (a folder per track).
rem
rem  From a terminal you can pass options through, e.g.:
rem      run_audio_analysis.bat --dry-run       list what would be done
rem      run_audio_analysis.bat --report        what every track has
rem      run_audio_analysis.bat --no-stems      skip the stems
rem      run_audio_analysis.bat --only "Song name"
rem ============================================================

rem Always run from the folder this .bat lives in (the scene-labeling folder)
cd /d "%~dp0"

rem Name of the virtual environment folder - change this if yours is called something else
set VENV=.venv

for %%F in (analyse_audio_local.py audio_analysis.py config.py) do (
    if not exist "%%F" (
        echo Missing required file: %%F - it must be in "%CD%"
        goto :end
    )
)

if not exist "%VENV%\Scripts\activate.bat" (
    echo No virtual environment found - creating "%CD%\%VENV%"...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo Couldn't create it - is Python installed and on PATH?
        goto :end
    )
)

call "%VENV%\Scripts\activate.bat"
set PYTHONUTF8=1
set PIP_DISABLE_PIP_VERSION_CHECK=1

rem ---- 1. Check / install everything ------------------------------------
echo.
python analyse_audio_local.py --setup
if errorlevel 1 (
    echo.
    echo Some parts aren't working - the analysis can still run, but with less
    echo detail ^(e.g. estimated downbeats, no vocals^). Scroll up for details.
    choice /c YN /m "Carry on anyway"
    if errorlevel 2 goto :end
)

rem ---- 2. Analyse the new tracks ----------------------------------------
echo.
echo Analysing your audio - new tracks, and anything older tracks are missing...
echo.
python analyse_audio_local.py %*
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" (
    echo Stopped with an error - exit code %EXITCODE%. Scroll up for details.
) else (
    echo Finished. Drive for desktop uploads the track files and stems; press Sync in the app
    echo ^(or wait for the 12-hourly sync^) and they show up in the Media Library.
)

:end
echo.
pause
endlocal
