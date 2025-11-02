@echo off
setlocal
REM Projektwurzel ist eine Ebene über driver_build
set "SRC_DIR=%~dp0.."

pushd "%SRC_DIR%"
if not exist build mkdir build

REM Optional: RC-Datei in RES kompilieren, wenn windres verfügbar ist
set "RC_SRC=%~dp0CipherCore_OpenCl.rc"
set "RC_OBJ=%TEMP%\CipherCore_OpenCl.res"
set "RC_FLAG="
where windres >NUL 2>&1
if %ERRORLEVEL%==0 (
  if exist "%RC_SRC%" (
    echo [RC] windres "%RC_SRC%" -O coff -o "%RC_OBJ%"
    windres "%RC_SRC%" -O coff -o "%RC_OBJ%"
    if %ERRORLEVEL%==0 (
      set "RC_FLAG=%RC_OBJ%"
    )
  )
)

REM Dein bewährter Einzeiler – mit optionalem RC-Link
g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fstrict-aliasing -DNDEBUG -DCL_TARGET_OPENCL_VERSION=120 -DCL_FAST_OPTS -shared ^
  CipherCore_OpenCl.c CipherCore_NoiseCtrl.c %RC_FLAG% ^
  -o build/CipherCore_OpenCl.dll ^
  -I"./" -I"./CL" ^
  -L"./CL" -lOpenCL ^
  -Wl,--out-implib,build/libCipherCore_OpenCl.a ^
  -static-libstdc++ -static-libgcc

set ERR=%ERRORLEVEL%
popd
exit /b %ERR%
