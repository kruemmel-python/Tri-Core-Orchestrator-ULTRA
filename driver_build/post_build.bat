@echo off
setlocal
REM Root relativ zu diesem Skript
set "ROOT=%~dp0.."
set "BUILD=%ROOT%\build"
set "DIST=%ROOT%\dist"

if not exist "%DIST%" mkdir "%DIST%"

echo [COPY] Artefakte nach dist\
set "SRC_DLL=%BUILD%\CipherCore_OpenCl.dll"
set "SRC_A=%BUILD%\libCipherCore_OpenCl.a"

if exist "%SRC_DLL%" (
  copy /Y "%SRC_DLL%" "%DIST%\CipherCore_OpenCl.dll" >nul
) else (
  echo [WARN] Quelle fehlt: %SRC_DLL%
)

if exist "%SRC_A%" (
  copy /Y "%SRC_A%" "%DIST%\libCipherCore_OpenCl.a" >nul
) else (
  echo [WARN] Quelle fehlt: %SRC_A%
)

echo [HASH] SHA256 erzeugen
> "%DIST%\checksums.txt" (
  if exist "%DIST%\CipherCore_OpenCl.dll" (
    echo CipherCore_OpenCl.dll:
    certutil -hashfile "%DIST%\CipherCore_OpenCl.dll" SHA256 | findstr /V /C:"hash of" /C:"CertUtil"
    echo.
  ) else (
    echo CipherCore_OpenCl.dll: (nicht gefunden)
    echo.
  )
  if exist "%DIST%\libCipherCore_OpenCl.a" (
    echo libCipherCore_OpenCl.a:
    certutil -hashfile "%DIST%\libCipherCore_OpenCl.a" SHA256 | findstr /V /C:"hash of" /C:"CertUtil"
    echo.
  ) else (
    echo libCipherCore_OpenCl.a: (nicht gefunden)
    echo.
  )
  echo Timestamp:
  echo %DATE% %TIME%
)

echo [OK] dist\ ist bereit: "%DIST%"
exit /b 0
