@echo off
setlocal

rem ============================================================
rem  Pack preview copies + catalogues for the Modal GPU trial.
rem
rem  Double-click: packs the 10-video trial into _modal_pack
rem  (next to this file). Nothing in Google Drive is changed.
rem
rem  From a terminal you can pass options through, e.g.:
rem      pack_for_modal.bat --dry-run       show which videos, copy nothing
rem      pack_for_modal.bat --trial 5       a smaller trial
rem      pack_for_modal.bat --all           every video
rem      pack_for_modal.bat --upload        pack, then upload to Modal
rem ============================================================

rem Always run from the folder this .bat lives in (the scene-labeling folder)
cd /d "%~dp0"

rem Name of the virtual environment folder - change this if yours is called something else
set VENV=.venv

if not exist "%VENV%\Scripts\activate.bat" (
    echo Could not find the virtual environment at "%CD%\%VENV%".
    echo Run setup first, or edit the VENV line at the top of this file.
    goto :end
)

for %%F in (pack_for_modal.py config.py) do (
    if not exist "%%F" (
        echo Missing required file: %%F - it must be in "%CD%"
        goto :end
    )
)

call "%VENV%\Scripts\activate.bat"
set PYTHONUTF8=1

echo.
echo Packing for Modal...
echo.
python pack_for_modal.py %*
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
