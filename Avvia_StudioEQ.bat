@echo off
setlocal EnableExtensions

:: Imposta la directory di lavoro sulla root del progetto (c:\_gestione ufficio\Audio SW)
cd /d "%~dp0"

:: Opzioni da riga di comando per forzare il setup o la ricompilazione (es. Avvia_StudioEQ.bat --rebuild)
if /I "%~1"=="--rebuild" goto :FIRST_RUN_SETUP
if /I "%~1"=="--setup"   goto :FIRST_RUN_SETUP

:: Avvio immediato e pulito senza lasciare alcuna finestra cmd.exe aperta
if exist "%~dp0StudioEQ_Master_Pro.exe" (
    start "" "%~dp0StudioEQ_Master_Pro.exe"
    exit /b 0
)

:FIRST_RUN_SETUP
chcp 65001 >nul 2>&1
title StudioEQ Master Pro - Primo Avvio e Compilazione Automatica
set "SETUP_SCRIPT=%~dp0scripts\setup_and_build.ps1"

echo ==============================================================================
echo   StudioEQ Master Pro - Configurazione e Compilazione Automatica
echo ==============================================================================
echo.
echo [INFO] Esecuzione automatica di "scripts\setup_and_build.ps1"...
echo.

if not exist "%SETUP_SCRIPT%" (
    echo [ERRORE] Script non trovato: "%SETUP_SCRIPT%"
    echo.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SETUP_SCRIPT%"
set "BUILD_EXIT=%ERRORLEVEL%"

if not "%BUILD_EXIT%"=="0" (
    echo.
    echo [ERRORE] La configurazione o compilazione si e' interrotta ^(codice %BUILD_EXIT%^).
    echo          Controlla le indicazioni riportate sopra e riprova.
    echo.
    pause
    exit /b %BUILD_EXIT%
)

if exist "%~dp0StudioEQ_Master_Pro.exe" (
    start "" "%~dp0StudioEQ_Master_Pro.exe"
    exit /b 0
) else (
    echo.
    echo [ERRORE] Compilazione terminata ma "%~dp0StudioEQ_Master_Pro.exe" non risulta presente.
    echo.
    pause
    exit /b 1
)
