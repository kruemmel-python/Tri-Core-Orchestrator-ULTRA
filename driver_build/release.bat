@echo off
setlocal
set "ROOT=%~dp0.."

echo [STEP] Build mit RC (falls windres vorhanden)
call "%~dp0build_driver_min.bat"
if errorlevel 1 exit /b 1

echo [STEP] Post-Build (Kopieren + SHA256)
call "%~dp0post_build.bat"
if errorlevel 1 exit /b 1

echo [STEP] Exporte anzeigen (falls Tool vorhanden)
powershell -ExecutionPolicy Bypass -File "%~dp0exports_list.ps1" "%ROOT%\build\CipherCore_OpenCl.dll"

echo [DONE] Release fertig. Artefakte liegen in: "%ROOT%\dist"
exit /b 0
