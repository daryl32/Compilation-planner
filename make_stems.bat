@echo off
setlocal

rem ============================================================
rem  Split your songs into stems (vocals + instrumental) on the GPU,
rem  for mashups in the Media Library's Mix tab.
rem
rem  Double-click it. It:
rem    1. checks and installs what's needed (same as run_audio_analysis.bat),
rem    2. lists the analysed tracks that have no stems yet,
rem    3. asks, then splits each one with Demucs on the GPU and saves
rem       scene-labeling\stems\<track>\vocals.m4a and instrumental.m4a
rem       in your Google Drive.
rem
rem  From a terminal you can pass options through, e.g.:
rem      make_stems.bat --only "Song name"
rem      make_stems.bat --redo
rem ============================================================

cd /d "%~dp0"
set VENV=.venv

for %%F in (make_stems_local.py analyse_audio_local.py audio_analysis.py config.py) do (
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
python make_stems_local.py --setup
if errorlevel 1 (
    echo.
    echo Some parts aren't working - Demucs is needed for stems. Scroll up for details.
    choice /c YN /m "Carry on anyway"
    if errorlevel 2 goto :end
)

echo.
python make_stems_local.py --dry-run %*
echo.
choice /c YN /m "Split these tracks now"
if errorlevel 2 goto :end

echo.
python make_stems_local.py %*
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" (
    echo Stopped with an error - exit code %EXITCODE%. Scroll up for details.
) else (
    echo Finished. Drive for desktop uploads the stems and the updated track files; press Sync
    echo in the app ^(or wait for the 12-hourly sync^) and the Mix tab can use them.
)

:end
echo.
pause
endlocal
