#Requires -Version 5.1
<#
.SYNOPSIS
    Script automatico di configurazione e compilazione per StudioEQ Master Pro.

.DESCRIPTION
    Rende la repository StudioEQ Master Pro completamente autonoma e riproducibile
    su qualsiasi PC Windows:
      1. Aggiorna il PATH di sessione e verifica la presenza di:
         - Rust (cargo / rustc)
         - Python 3.11 - 3.13 (compatibile con PyTorch CUDA, Demucs, Pedalboard e SciPy)
         - FFmpeg (decodifica/codifica audio WAV 24-bit e MP3 320 kbps)
      2. Crea l'ambiente virtuale Python in "<root>\.venv" (se non presente o incompleto),
         rilevando automaticamente tramite "nvidia-smi" se e' disponibile una GPU NVIDIA
         per installare PyTorch CUDA (cu126) oppure il fallback CPU, insieme a:
         demucs, pedalboard, noisereduce, pyloudnorm, soundfile, matchering, scipy.
      3. Compila il motore nativo "studio-eq-rust" in modalita' Release ("cargo build --release"),
         copia "StudioEQ_Master_Pro.exe" nella root del progetto e crea/aggiorna
         opzionalmente il collegamento sul Desktop con l'icona custom "app_icon.ico".

.PARAMETER SkipVenv
    Salta la verifica/creazione dell'ambiente virtuale Python (.venv).

.PARAMETER ForceVenv
    Elimina e ricrea da zero la cartella ".venv" forzando la reinstallazione dei pacchetti.

.PARAMETER ForceCpu
    Forza l'installazione di PyTorch in versione CPU anche se viene rilevata una GPU NVIDIA.

.PARAMETER SkipBuild
    Salta la compilazione Rust ("cargo build --release").

.PARAMETER NoDesktopShortcut
    Disabilita la creazione/aggiornamento del collegamento sul Desktop.

.PARAMETER AskDesktopShortcut
    Chiede conferma interattiva [S/n] prima di creare il collegamento sul Desktop.

.PARAMETER Launch
    Avvia automaticamente "StudioEQ_Master_Pro.exe" al termine della procedura.

.PARAMETER CheckOnly
    Esegue esclusivamente il controllo dei prerequisiti di sistema e della GPU senza
    modificare file, creare ".venv" o avviare la compilazione.
#>

[CmdletBinding()]
param(
    [switch]$SkipVenv,
    [switch]$ForceVenv,
    [switch]$ForceCpu,
    [switch]$SkipBuild,
    [Alias('SkipDesktopShortcut')]
    [switch]$NoDesktopShortcut,
    [switch]$AskDesktopShortcut,
    [switch]$Launch,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ==============================================================================
# Funzioni di utilita' per log leggibili in console
# ==============================================================================
function Write-Banner {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("  " + $Title) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

function Write-Step {
    param([string]$StepNum, [string]$Message)
    Write-Host ""
    Write-Host ("[{0}] {1}" -f $StepNum, $Message) -ForegroundColor Yellow
}

function Write-Ok {
    param([string]$Message)
    Write-Host ("  [OK] " + $Message) -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host ("  [i]  " + $Message) -ForegroundColor Gray
}

function Write-WarnMsg {
    param([string]$Message)
    Write-Host ("  [!]  " + $Message) -ForegroundColor DarkYellow
}

function Write-ErrMsg {
    param([string]$Message)
    Write-Host ("  [X]  " + $Message) -ForegroundColor Red
}

# Esegue un eseguibile nativo (cargo, python, pip) senza che l'output informativo
# su stderr (es. "Compiling..." di cargo o i notice di pip) venga trattato come
# eccezione fatale da PowerShell 5.1 quando lo script viene invocato con pipe ridirette.
function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $FilePath @Arguments
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prevEap
    }
}

# ==============================================================================
# Aggiornamento PATH di sessione (utile subito dopo installazioni con winget/rustup)
# ==============================================================================
function Update-SessionEnvironmentPath {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $extraDirs   = @(
        (Join-Path $env:USERPROFILE '.cargo\bin'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'),
        'C:\ffmpeg\bin',
        'C:\ProgramData\chocolatey\bin'
    )

    $combined = New-Object System.Collections.Generic.List[string]
    foreach ($raw in @($env:Path, $machinePath, $userPath)) {
        if (-not [string]::IsNullOrWhiteSpace($raw)) {
            foreach ($part in ($raw -split ';')) {
                $trimmed = $part.Trim()
                if ($trimmed -and (-not $combined.Contains($trimmed))) {
                    $combined.Add($trimmed)
                }
            }
        }
    }

    foreach ($dir in $extraDirs) {
        if ($dir -and (Test-Path -LiteralPath $dir) -and (-not $combined.Contains($dir))) {
            $combined.Add($dir)
        }
    }

    # Cerca anche eventuali pacchetti FFmpeg installati da WinGet non ancora nel PATH
    $wingetPkgRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    if (Test-Path -LiteralPath $wingetPkgRoot) {
        $ffmpegBins = Get-ChildItem -Path $wingetPkgRoot -Filter 'ffmpeg.exe' -Recurse -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty DirectoryName -Unique
        foreach ($binDir in $ffmpegBins) {
            if ($binDir -and (-not $combined.Contains($binDir))) {
                $combined.Add($binDir)
            }
        }
    }

    $env:Path = ($combined -join ';')
}

# ==============================================================================
# Ricerca automatica di Python 3.11, 3.12 o 3.13
# (Esclude versioni non compatibili come 3.14+ o <3.11 e gli stub Microsoft Store)
# ==============================================================================
function Test-PythonInterpreter {
    param(
        [Parameter(Mandatory = $true)][string]$CommandOrPath,
        [string[]]$ExtraArgs = @()
    )

    try {
        $probeCode = "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}|{sys.executable}')"
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $rawOut = & $CommandOrPath @ExtraArgs -c $probeCode 2>$null
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEap

        if ($exitCode -ne 0 -or -not $rawOut) {
            return $null
        }

        $line = ($rawOut | Select-Object -First 1).ToString().Trim()
        if ($line -match '^(\d+)\.(\d+)\.(\d+)\|(.+)$') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            $patch = [int]$Matches[3]
            $exePath = $Matches[4].Trim()

            if (Test-Path -LiteralPath $exePath) {
                return [PSCustomObject]@{
                    Major       = $major
                    Minor       = $minor
                    Patch       = $patch
                    Version     = "$major.$minor.$patch"
                    Executable  = $exePath
                    IsSupported = ($major -eq 3 -and ($minor -ge 11 -and $minor -le 13))
                }
            }
        }
    }
    catch {
        return $null
    }
    return $null
}

function Find-CompatiblePython {
    param([string]$ExistingVenvPython)

    $detectedUnsupported = New-Object System.Collections.Generic.List[string]

    # 1. Se esiste gia' una .venv valida con Python 3.11-3.13, tienila presente come fallback
    $venvCandidate = $null
    if ($ExistingVenvPython -and (Test-Path -LiteralPath $ExistingVenvPython)) {
        $probe = Test-PythonInterpreter -CommandOrPath $ExistingVenvPython
        if ($probe -and $probe.IsSupported) {
            $venvCandidate = $probe
        }
    }

    # 2. Prova tramite il Python Launcher ufficiale di Windows (py.exe): 3.13 -> 3.12 -> 3.11
    $pyLauncher = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($verTag in @('-3.13', '-3.12', '-3.11')) {
            $probe = Test-PythonInterpreter -CommandOrPath $pyLauncher.Source -ExtraArgs @($verTag)
            if ($probe -and $probe.IsSupported) {
                return $probe
            }
        }
    }

    # 3. Prova i comandi diretti presenti nel PATH
    foreach ($cmdName in @('python3.13', 'python3.12', 'python3.11', 'python', 'python3')) {
        $cmd = Get-Command $cmdName -ErrorAction SilentlyContinue
        if ($cmd) {
            $probe = Test-PythonInterpreter -CommandOrPath $cmd.Source
            if ($probe) {
                if ($probe.IsSupported) {
                    return $probe
                }
                $desc = "Python $($probe.Version) ($($probe.Executable))"
                if (-not $detectedUnsupported.Contains($desc)) {
                    $detectedUnsupported.Add($desc)
                }
            }
        }
    }

    # 4. Cerca nei percorsi standard di installazione su Windows
    $standardPaths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:ProgramFiles 'Python313\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python311\python.exe'),
        'C:\Python313\python.exe',
        'C:\Python312\python.exe',
        'C:\Python311\python.exe'
    )

    foreach ($candidatePath in $standardPaths) {
        if ($candidatePath -and (Test-Path -LiteralPath $candidatePath)) {
            $probe = Test-PythonInterpreter -CommandOrPath $candidatePath
            if ($probe -and $probe.IsSupported) {
                return $probe
            }
        }
    }

    if ($venvCandidate -and (-not $ForceVenv)) {
        return $venvCandidate
    }

    if ($detectedUnsupported.Count -gt 0) {
        Write-WarnMsg ("Rilevate versioni Python fuori dal range 3.11-3.13: " + ($detectedUnsupported -join '; '))
    }

    return $null
}

# ==============================================================================
# Rilevamento GPU NVIDIA tramite nvidia-smi
# ==============================================================================
function Get-NvidiaGpuInfo {
    $smiCandidates = @()
    $smiCmd = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if ($smiCmd) {
        $smiCandidates += $smiCmd.Source
    }
    $smiCandidates += @(
        'C:\Windows\System32\nvidia-smi.exe',
        'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    )

    foreach ($smi in ($smiCandidates | Select-Object -Unique)) {
        if ($smi -and (Test-Path -LiteralPath $smi)) {
            try {
                $prevEap = $ErrorActionPreference
                $ErrorActionPreference = 'Continue'
                $gpuOut = & $smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
                $exitCode = $LASTEXITCODE
                $ErrorActionPreference = $prevEap

                if ($exitCode -eq 0 -and $gpuOut) {
                    $firstGpu = ($gpuOut | Select-Object -First 1).ToString().Trim()
                    if (-not [string]::IsNullOrWhiteSpace($firstGpu)) {
                        return [PSCustomObject]@{
                            Available = $true
                            Summary   = $firstGpu
                        }
                    }
                }
            }
            catch {
                # Continua con gli altri candidati
            }
        }
    }

    return [PSCustomObject]@{
        Available = $false
        Summary   = $null
    }
}

# ==============================================================================
# Verifica integrita' dei pacchetti Python all'interno di .venv
# ==============================================================================
function Test-VenvPackagesReady {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][bool]$RequireCuda
    )

    if (-not (Test-Path -LiteralPath $PythonPath)) {
        return $false
    }

    $checkScript = @"
import sys
mods = ['numpy', 'scipy', 'torch', 'torchaudio', 'demucs', 'pedalboard', 'noisereduce', 'pyloudnorm', 'soundfile', 'matchering']
for m in mods:
    __import__(m)
import torch
need_cuda = sys.argv[1] == '1'
if need_cuda and not torch.version.cuda:
    sys.exit(2)
print(f'torch={torch.__version__}|cuda={torch.cuda.is_available()}')
"@

    try {
        $cudaArg = if ($RequireCuda) { '1' } else { '0' }
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $out = & $PythonPath -c $checkScript $cudaArg 2>$null
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEap

        return ($exitCode -eq 0 -and [bool]$out)
    }
    catch {
        return $false
    }
}

# ==============================================================================
# INIZIO ESECUZIONE PRINCIPALE
# ==============================================================================
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir '..')).Path
$RustDir     = Join-Path $ProjectRoot 'studio-eq-rust'
$CargoToml   = Join-Path $RustDir 'Cargo.toml'
$VenvDir     = Join-Path $ProjectRoot '.venv'
$VenvPython  = Join-Path $VenvDir 'Scripts\python.exe'
$OutputDir   = Join-Path $ProjectRoot 'output'
$ExeBuilt    = Join-Path $RustDir 'target\release\StudioEQ_Master_Pro.exe'
$ExeRoot     = Join-Path $ProjectRoot 'StudioEQ_Master_Pro.exe'

Write-Banner 'StudioEQ Master Pro - Setup Ambiente AI & Build Nativa Rust (64-Bit)'
Write-Info "Cartella progetto : $ProjectRoot"

Update-SessionEnvironmentPath

# Assicura che la cartella di lavoro "output" esista sempre dopo un git clone pulito
if (-not $CheckOnly -and -not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# ------------------------------------------------------------------------------
# PASSO 1: Verifica prerequisiti (Rust / cargo, Python 3.11-3.13, FFmpeg, GPU)
# ------------------------------------------------------------------------------
Write-Step '1/4' 'Verifica dei prerequisiti di sistema (Rust, Python 3.11-3.13, FFmpeg)...'

$missingTools = New-Object System.Collections.Generic.List[string]

# 1A. Verifica Rust (cargo)
$cargoCmd = Get-Command 'cargo' -ErrorAction SilentlyContinue
if ($cargoCmd) {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $cargoOut = & $cargoCmd.Source --version 2>$null
    $cargoExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    $cargoVer = ($cargoOut | Select-Object -First 1)

    if ($cargoExit -eq 0 -and $cargoVer) {
        Write-Ok "Rust rilevato: $cargoVer"
    }
    else {
        $missingTools.Add("Rust (cargo) non risponde correttamente. Installa da https://rustup.rs/ o con: winget install -e --id Rustlang.Rustup")
    }
}
else {
    Write-ErrMsg 'Rust (cargo) NON trovato nel sistema.'
    $missingTools.Add("Rust (cargo): installa con 'winget install -e --id Rustlang.Rustup' (oppure da https://rustup.rs/) insieme a 'Visual Studio C++ Build Tools'.")
}

# 1B. Verifica Python 3.11 - 3.13
$pythonInfo = Find-CompatiblePython -ExistingVenvPython $VenvPython
if ($pythonInfo) {
    Write-Ok "Python compatibile rilevato: v$($pythonInfo.Version) ($($pythonInfo.Executable))"
}
else {
    Write-ErrMsg 'Python 3.11, 3.12 o 3.13 NON trovato.'
    $missingTools.Add("Python 3.11-3.13: installa ad esempio Python 3.12 o 3.13 con 'winget install -e --id Python.Python.3.12' (oppure 'winget install -e --id Python.Python.3.13').")
}

# 1C. Verifica FFmpeg
$ffmpegCmd = Get-Command 'ffmpeg' -ErrorAction SilentlyContinue
if ($ffmpegCmd) {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $ffmpegOut = & $ffmpegCmd.Source -version 2>$null
    $ffmpegExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    $ffmpegVer = ($ffmpegOut | Select-Object -First 1)

    if ($ffmpegExit -eq 0 -and $ffmpegVer) {
        Write-Ok "FFmpeg rilevato: $ffmpegVer"
    }
    else {
        $missingTools.Add("FFmpeg non risponde correttamente. Reinstalla con: winget install -e --id Gyan.FFmpeg")
    }
}
else {
    Write-ErrMsg 'FFmpeg NON trovato nel PATH.'
    $missingTools.Add("FFmpeg: necessario per decodifica/esportazione audio. Installa con 'winget install -e --id Gyan.FFmpeg'.")
}

# 1D. Rilevamento GPU NVIDIA (informativo, con fallback automatico su CPU)
$gpuStatus = Get-NvidiaGpuInfo
$useCuda = ($gpuStatus.Available -and (-not $ForceCpu))
if ($ForceCpu) {
    Write-Info 'Modalita'' CPU forzata tramite parametro -ForceCpu.'
}
elseif ($useCuda) {
    Write-Ok "GPU NVIDIA rilevata (nvidia-smi): $($gpuStatus.Summary) -> accelerazione CUDA (cu126) attiva"
}
else {
    Write-Info 'Nessuna GPU NVIDIA rilevata tramite nvidia-smi -> verra'' usato PyTorch in modalita'' CPU fallback.'
}

# Verifica presenza icona custom
$iconCandidates = @(
    (Join-Path $ProjectRoot 'app_icon.ico'),
    (Join-Path $RustDir 'app_icon.ico'),
    (Join-Path $ScriptDir 'app_icon.ico')
)
$resolvedIconPath = $iconCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($resolvedIconPath) {
    Write-Ok "Icona custom rilevata: $resolvedIconPath"
}
else {
    Write-WarnMsg 'File app_icon.ico non trovato; il collegamento Desktop usera'' l''icona incorporata nell''eseguibile.'
}

if ($missingTools.Count -gt 0) {
    Write-Host ""
    Write-Host '==============================================================================' -ForegroundColor Red
    Write-Host '  PREREQUISITI MANCANTI - IMPOSSIBILE PROSEGUIRE' -ForegroundColor Red
    Write-Host '==============================================================================' -ForegroundColor Red
    foreach ($item in $missingTools) {
        Write-Host ("  - " + $item) -ForegroundColor Red
    }
    Write-Host ""
    Write-Host '  Dopo aver installato i componenti mancanti, riavvia questo script o Avvia_StudioEQ.bat.' -ForegroundColor Yellow
    exit 1
}

if ($CheckOnly) {
    Write-Host ""
    Write-Ok 'Controllo prerequisiti (-CheckOnly) superato con successo! Nessuna modifica effettuata.'
    exit 0
}

# ------------------------------------------------------------------------------
# PASSO 2: Creazione e configurazione dell'ambiente virtuale Python (.venv)
# ------------------------------------------------------------------------------
Write-Step '2/4' 'Configurazione ambiente virtuale Python (.venv) e librerie AI/DSP...'

if ($SkipVenv) {
    Write-Info 'Configurazione .venv saltata per via del parametro -SkipVenv.'
}
else {
    if ($ForceVenv -and (Test-Path -LiteralPath $VenvDir)) {
        Write-Info "Rimozione ambiente .venv esistente (-ForceVenv): $VenvDir"
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Info "Creazione nuovo ambiente virtuale in '$VenvDir' con Python $($pythonInfo.Version)..."
        $venvExit = Invoke-NativeCommand -FilePath $pythonInfo.Executable -Arguments @('-m', 'venv', $VenvDir)
        if ($venvExit -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
            Write-ErrMsg "Creazione dell'ambiente virtuale .venv fallita."
            exit 1
        }
        Write-Ok 'Ambiente virtuale .venv creato con successo.'
    }
    else {
        Write-Ok "Ambiente virtuale .venv gia' presente: $VenvDir"
    }

    # Controlla se tutte le dipendenze (incluso il backend CUDA se disponibile) sono gia' installate
    if ((-not $ForceVenv) -and (Test-VenvPackagesReady -PythonPath $VenvPython -RequireCuda $useCuda)) {
        Write-Ok 'Tutti i pacchetti Python AI/DSP (PyTorch, Demucs, Pedalboard, ecc.) risultano gia'' installati e pronti.'
    }
    else {
        Write-Info 'Aggiornamento di pip, setuptools e wheel nell''ambiente .venv...'
        $pipUpgradeExit = Invoke-NativeCommand -FilePath $VenvPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel')
        if ($pipUpgradeExit -ne 0) {
            Write-ErrMsg 'Aggiornamento di pip fallito.'
            exit 1
        }

        $installedCuda = $false
        if ($useCuda) {
            Write-Info 'Installazione di PyTorch + TorchAudio con accelerazione NVIDIA CUDA (cu126)...'
            $torchExit = Invoke-NativeCommand -FilePath $VenvPython -Arguments @(
                '-m', 'pip', 'install', '--upgrade', 'torch', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu126'
            )

            if ($torchExit -eq 0) {
                $installedCuda = $true
                Write-Ok 'PyTorch CUDA (cu126) installato correttamente.'
            }
            else {
                Write-WarnMsg 'Installazione PyTorch cu126 non riuscita; passaggio automatico al fallback CPU...'
            }
        }

        if (-not $installedCuda) {
            Write-Info 'Installazione di PyTorch + TorchAudio in versione CPU...'
            $cpuExit = Invoke-NativeCommand -FilePath $VenvPython -Arguments @(
                '-m', 'pip', 'install', '--upgrade', 'torch', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            )
            if ($cpuExit -ne 0) {
                Write-ErrMsg 'Installazione di PyTorch (CPU) fallita.'
                exit 1
            }
            Write-Ok 'PyTorch (CPU fallback) installato correttamente.'
        }

        $audioPackages = @(
            'numpy',
            'scipy',
            'soundfile',
            'demucs',
            'pedalboard',
            'noisereduce',
            'pyloudnorm',
            'matchering'
        )

        Write-Info ("Installazione pacchetti Audio/AI: " + ($audioPackages -join ', ') + '...')
        if ($installedCuda) {
            # Mantiene il repository cu126 come extra-index per evitare che pip sostituisca torch+cu126 con la build CPU
            $pkgArgs = @('-m', 'pip', 'install', '--extra-index-url', 'https://download.pytorch.org/whl/cu126') + $audioPackages
            $pkgExit = Invoke-NativeCommand -FilePath $VenvPython -Arguments $pkgArgs
        }
        else {
            $pkgArgs = @('-m', 'pip', 'install') + $audioPackages
            $pkgExit = Invoke-NativeCommand -FilePath $VenvPython -Arguments $pkgArgs
        }

        if ($pkgExit -ne 0) {
            Write-ErrMsg 'Installazione dei pacchetti Audio/AI in .venv fallita.'
            exit 1
        }

        # Verifica finale di importazione (ignora eventuali warning informativi su stderr)
        $verifyPy = "import torch, demucs, pedalboard, noisereduce, pyloudnorm, soundfile, matchering, scipy; " +
                    "dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; " +
                    "print(f'PyTorch {torch.__version__} attivo su: {dev}')"
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $verifyOut = & $VenvPython -c $verifyPy 2>$null
        $verifyExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEap

        if ($verifyExit -ne 0) {
            Write-ErrMsg 'Verifica finale dei moduli Python fallita.'
            exit 1
        }
        $verifyMsg = ($verifyOut | Select-Object -Last 1)
        Write-Ok "Ambiente AI pronto -> $verifyMsg"
    }
}

# ------------------------------------------------------------------------------
# PASSO 3: Compilazione nativa Rust (cargo build --release) e copia eseguibile
# ------------------------------------------------------------------------------
Write-Step '3/4' 'Compilazione nativa di studio-eq-rust in modalita'' Release (64-Bit)...'

if ($SkipBuild) {
    Write-Info 'Compilazione Rust saltata per via del parametro -SkipBuild.'
}
else {
    if (-not (Test-Path -LiteralPath $CargoToml)) {
        Write-ErrMsg "File Cargo.toml non trovato in: $CargoToml"
        exit 1
    }

    Push-Location -LiteralPath $RustDir
    try {
        $cargoBuildExit = Invoke-NativeCommand -FilePath $cargoCmd.Source -Arguments @('build', '--release')
    }
    finally {
        Pop-Location
    }

    if ($cargoBuildExit -ne 0 -or -not (Test-Path -LiteralPath $ExeBuilt)) {
        Write-ErrMsg 'Compilazione Rust fallita. Verifica di avere installato i Visual Studio C++ Build Tools (workload C++).'
        exit 1
    }

    Write-Ok "Binario compilato: $ExeBuilt"

    # Copia StudioEQ_Master_Pro.exe nella root del progetto
    try {
        Copy-Item -LiteralPath $ExeBuilt -Destination $ExeRoot -Force
        Write-Ok "Eseguibile copiato nella root del progetto: $ExeRoot"
    }
    catch {
        Write-ErrMsg "Impossibile copiare StudioEQ_Master_Pro.exe in '$ExeRoot'. Chiudi l'applicazione se e' attualmente aperta e riprova."
        exit 1
    }
}

# ------------------------------------------------------------------------------
# PASSO 4: Creazione / aggiornamento opzionale del collegamento sul Desktop
# ------------------------------------------------------------------------------
Write-Step '4/4' 'Configurazione collegamento sul Desktop con icona custom...'

$shouldCreateShortcut = (-not $NoDesktopShortcut)

if ($shouldCreateShortcut -and $AskDesktopShortcut) {
    if ([Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)) {
        $ans = Read-Host '  Vuoi creare/aggiornare il collegamento "StudioEQ Master Pro" sul Desktop? [S/n]'
        if ($ans -match '^(n|no)$') {
            $shouldCreateShortcut = $false
        }
    }
}

if (-not $shouldCreateShortcut) {
    Write-Info 'Creazione collegamento Desktop disattivata (-NoDesktopShortcut o scelta utente).'
}
elseif (-not (Test-Path -LiteralPath $ExeRoot)) {
    Write-WarnMsg "Eseguibile '$ExeRoot' non presente: collegamento Desktop saltato."
}
else {
    try {
        $desktopDir = [Environment]::GetFolderPath('Desktop')
        if ([string]::IsNullOrWhiteSpace($desktopDir) -or (-not (Test-Path -LiteralPath $desktopDir))) {
            Write-WarnMsg 'Cartella Desktop non rilevata per l''utente corrente; collegamento saltato.'
        }
        else {
            # Ricalcola l'icona nel caso sia stata generata durante il build
            $finalIcon = $iconCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
            $shortcutPath = Join-Path $desktopDir 'StudioEQ Master Pro.lnk'
            $wasExisting = Test-Path -LiteralPath $shortcutPath

            $wshShell = New-Object -ComObject WScript.Shell
            $shortcut = $wshShell.CreateShortcut($shortcutPath)
            $shortcut.TargetPath       = $ExeRoot
            $shortcut.WorkingDirectory = $ProjectRoot
            $shortcut.Description      = 'StudioEQ Master Pro - Multi-Track AI Stem Mixer & 64-Bit Equalizer'
            if ($finalIcon) {
                $shortcut.IconLocation = "$finalIcon,0"
            }
            else {
                $shortcut.IconLocation = "$ExeRoot,0"
            }
            $shortcut.Save()

            if ($wasExisting) {
                Write-Ok "Collegamento Desktop aggiornato: $shortcutPath"
            }
            else {
                Write-Ok "Collegamento Desktop creato: $shortcutPath"
            }
        }
    }
    catch {
        Write-WarnMsg "Impossibile creare/aggiornare il collegamento sul Desktop: $($_.Exception.Message)"
    }
}

Write-Banner 'INSTALLAZIONE E COMPILAZIONE COMPLETATE CON SUCCESSO!'
Write-Host "  Eseguibile pronto : $ExeRoot" -ForegroundColor Green
Write-Host "  Launcher rapido   : $(Join-Path $ProjectRoot 'Avvia_StudioEQ.bat')" -ForegroundColor Green
Write-Host ""

if ($Launch -and (Test-Path -LiteralPath $ExeRoot)) {
    Write-Info 'Avvio di StudioEQ Master Pro in corso...'
    Start-Process -FilePath $ExeRoot -WorkingDirectory $ProjectRoot
}

exit 0
