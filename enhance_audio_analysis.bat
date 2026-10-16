@echo off
setlocal

rem ============================================================
rem  Enhance the audio analysis of tracks you already have.
rem
rem  Double-click it. It:
rem    1. checks and installs what's needed (same as run_audio_analysis.bat),
rem    2. lists every track: bar lines from the beat_this model or
rem       estimated, and whether it has vocals,
rem    3. asks, then reruns the analysis on every track with estimated
rem       bar lines, no vocals, or no extra analysis - on the GPU.
rem       Each track's beats and energy are KEPT, so plans still line up.
rem
rem  From a terminal you can pass options through, e.g.:
rem      enhance_audio_analysis.bat --only "Song name"
rem      enhance_audio_analysis.bat --no-vocals     bar lines only
rem ============================================================

cd /d "%~dp0"
set VENV=.venv

for %%F in (analyse_audio_local.py audio_analysis.py config.py) do (
    if not exist "%%F" (
        echo Missing required file: %%F - it must be in "%CD%"
        goto :end
    )
)
if not exist "%VENV%\Scripts\activate.bat" (
    echo No virtual environment found - run run_audio_analysis.bat first.
    goto :end
)

call "%VENV%\Scripts\activate.bat"
set PYTHONUTF8=1
set PIP_DISABLE_PIP_VERSION_CHECK=1

echo.
python analyse_audio_local.py --setup
if errorlevel 1 (
    echo.
    echo Some parts aren't working - scroll up for details.
    choice /c YN /m "Carry on anyway"
    if errorlevel 2 goto :end
)

echo.
python analyse_audio_local.py --report
echo.
python analyse_audio_local.py --enhance --dry-run %*
echo.
choice /c YN /m "Rerun these tracks now"
if errorlevel 2 goto :end

echo.
python analyse_audio_local.py --enhance %*
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" (
    echo Stopped with an error - exit code %EXITCODE%. Scroll up for details.
) else (
    echo Finished. Drive for desktop uploads the updated track files; press Sync in the app
    echo ^(or wait for the 12-hourly sync^) to see them on the server.
)

:end
echo.
pause
endlocal
