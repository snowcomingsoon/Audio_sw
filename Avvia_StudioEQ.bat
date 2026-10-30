@echo off
setlocal EnableExtensions

:: Imposta dinamicamente la directory di lavoro sulla root corrente del progetto
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
cd /d "%PROJECT_ROOT%"

:: Variabili runtime obbligatorie per accelerazione ROCm 7.13 (AMD Radeon RX 9070 RDNA4 / RDNA3) e HuggingFace Hub
set "TORCH_BLAS_PREFER_HIPBLASLT=0"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_IMPLICIT_TOKEN=1"

:: Arricchisce il PATH di sessione con Rust (.cargo\bin) e WinGet\Links
if exist "%USERPROFILE%\.cargo\bin" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"

:: Aggiunge automaticamente la cartella bin reale dentro WinGet\Packages\*FFmpeg* (oltre a WinGet\Links), se presente
set "FFMPEG_WINGET_BIN="
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages" (
    for /d %%P in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\*FFmpeg*") do (
        for /f "delims=" %%F in ('dir /b /s "%%~fP\ffmpeg.exe" 2^>nul') do (
            set "FFMPEG_WINGET_BIN=%%~dpF"
            goto :APPLY_FFMPEG_BIN
        )
    )
)
where ffmpeg >nul 2>&1
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages" (
        for /f "delims=" %%F in ('dir /b /s "%LOCALAPPDATA%\Microsoft\WinGet\Packages\ffmpeg.exe" 2^>nul') do (
            set "FFMPEG_WINGET_BIN=%%~dpF"
            goto :APPLY_FFMPEG_BIN
        )
    )
)
goto :FFMPEG_READY

:APPLY_FFMPEG_BIN
if defined FFMPEG_WINGET_BIN (
    if "%FFMPEG_WINGET_BIN:~-1%"=="\" set "FFMPEG_WINGET_BIN=%FFMPEG_WINGET_BIN:~0,-1%"
)
if defined FFMPEG_WINGET_BIN set "PATH=%FFMPEG_WINGET_BIN%;%PATH%"

:FFMPEG_READY
if exist "%PROJECT_ROOT%\.venv\Scripts" set "PATH=%PROJECT_ROOT%\.venv\Scripts;%PATH%"

:: Opzioni da riga di comando:
::   --check      -> esegue solo il controllo prerequisiti (GPU NVIDIA/AMD Radeon, Rust gnu/msvc, Python, FFmpeg)
::   --rebuild    -> forza la verifica ambiente + ricompilazione nativa Rust
::   --setup      -> esegue setup completo (.venv + build Rust)
::   --force-cpu  -> forza PyTorch CPU durante il setup
set "PS_EXTRA_ARGS="
if /I "%~1"=="--check" goto :CHECK_ONLY
if /I "%~1"=="--force-cpu" (
    set "PS_EXTRA_ARGS=-ForceCpu"
    goto :FIRST_RUN_SETUP
)
if /I "%~1"=="--rebuild" goto :FIRST_RUN_SETUP
if /I "%~1"=="--setup"   goto :FIRST_RUN_SETUP

:: Avvio immediato e pulito se sia l'eseguibile nativo sia l'ambiente AI (.venv) sono presenti
if exist "%PROJECT_ROOT%\StudioEQ_Master_Pro.exe" (
    if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
        start "" "%PROJECT_ROOT%\StudioEQ_Master_Pro.exe"
        exit /b 0
    )
)
goto :FIRST_RUN_SETUP

:CHECK_ONLY
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\setup_and_build.ps1" -CheckOnly
exit /b %ERRORLEVEL%

:FIRST_RUN_SETUP
chcp 65001 >nul 2>&1
title StudioEQ Master Pro - Setup AI GPU (NVIDIA / AMD Radeon) e Build Rust (GNU / MSVC)
set "SETUP_SCRIPT=%PROJECT_ROOT%\scripts\setup_and_build.ps1"

echo ==============================================================================
echo   StudioEQ Master Pro - Configurazione e Compilazione Automatica
echo ==============================================================================
echo.
echo [INFO] Rilevamento GPU (Win32_VideoController: NVIDIA / AMD Radeon) e Toolchain Rust (gnu / msvc)...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$gpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name); if ($gpus) { Write-Host ('[GPU]  Schede video rilevate: ' + ($gpus -join ' | ')) -ForegroundColor Cyan }; $rc = Get-Command rustc -ErrorAction SilentlyContinue; if ($rc) { $hostLine = (& $rc.Source -vV 2>$null | Select-String '^host:'); if ($hostLine) { Write-Host ('[RUST] Toolchain attivo    : ' + $hostLine.Line.Trim()) -ForegroundColor Cyan } }"
echo [INFO] Esecuzione automatica di "scripts\setup_and_build.ps1"...
echo.

if not exist "%SETUP_SCRIPT%" (
    echo [ERRORE] Script non trovato: "%SETUP_SCRIPT%"
    echo.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SETUP_SCRIPT%" %PS_EXTRA_ARGS%
set "BUILD_EXIT=%ERRORLEVEL%"

if not "%BUILD_EXIT%"=="0" (
    echo.
    echo [ERRORE] La configurazione o compilazione si e' interrotta ^(codice %BUILD_EXIT%^).
    echo          Controlla le indicazioni riportate sopra e riprova.
    echo.
    pause
    exit /b %BUILD_EXIT%
)

if exist "%PROJECT_ROOT%\StudioEQ_Master_Pro.exe" (
    start "" "%PROJECT_ROOT%\StudioEQ_Master_Pro.exe"
    exit /b 0
) else (
    echo.
    echo [ERRORE] Compilazione terminata ma "%PROJECT_ROOT%\StudioEQ_Master_Pro.exe" non risulta presente.
    echo.
    pause
    exit /b 1
)
