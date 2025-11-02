Param(
  [string]$DllPath = ".\build\CipherCore_OpenCl.dll"
)
Write-Host "[INFO] DLL:" (Resolve-Path $DllPath)
# Prefer dumpbin if available (VS Tools)
$dumpbin = (Get-Command dumpbin.exe -ErrorAction SilentlyContinue)
if ($dumpbin) {
  & $dumpbin /exports $DllPath
  exit $LASTEXITCODE
}

# Try llvm-objdump
$llvm = (Get-Command llvm-objdump.exe -ErrorAction SilentlyContinue)
if ($llvm) {
  & $llvm -p $DllPath
  exit $LASTEXITCODE
}

# Try objdump (binutils)
$objdump = (Get-Command objdump.exe -ErrorAction SilentlyContinue)
if ($objdump) {
  & $objdump -p $DllPath
  exit $LASTEXITCODE
}

Write-Warning "Kein Tool gefunden (dumpbin/llvm-objdump/objdump). Bitte eines davon installieren oder VS Build Tools aktivieren."
exit 0
