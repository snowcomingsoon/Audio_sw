#Requires -Version 5.1
<#
.SYNOPSIS
    Script automatico di configurazione e compilazione per StudioEQ Master Pro.

.DESCRIPTION
    Rende la repository StudioEQ Master Pro completamente autonoma e riproducibile
    su qualsiasi PC Windows:
      1. Aggiorna il PATH di sessione e verifica la presenza di:
         - Rust (cargo / rustc) con supporto nativo sia per toolchain "x86_64-pc-windows-gnu"
           (self-contained, senza bisogno di Visual Studio C++ Build Tools) sia "x86_64-pc-windows-msvc"
         - Python 3.11 - 3.13 (preferendo Python 3.12 per piena compatibilita' con ROCm Windows / CUDA)
         - FFmpeg (decodifica/codifica audio WAV 24-bit e MP3 320 kbps)
      2. Crea l'ambiente virtuale Python in "<root>\.venv" (se non presente o incompleto),
         rilevando automaticamente GPU NVIDIA (nvidia-smi / Win32_VideoController -> CUDA cu126)
         e GPU AMD Radeon (Win32_VideoController -> ROCm/TheRock HIP nativo per RDNA4 gfx120X-all /
         RDNA3 gfx110X-all, con fallback sicuro su PyTorch CPU OpenMP), insieme a:
         demucs, pedalboard, noisereduce, pyloudnorm, soundfile, librosa, matchering,
         scipy, soxr, stftpitchshift, audiotsm.
      3. Compila il motore nativo "studio-eq-rust" in modalita' Release ("cargo build --release"),
         copia "StudioEQ_Master_Pro.exe" nella root del progetto e crea/aggiorna
         opzionalmente il collegamento sul Desktop con l'icona custom "app_icon.ico".

.PARAMETER SkipVenv
    Salta la verifica/creazione dell'ambiente virtuale Python (.venv).

.PARAMETER ForceVenv
    Elimina e ricrea da zero la cartella ".venv" forzando la reinstallazione dei pacchetti.

.PARAMETER ForceCpu
    Forza l'installazione di PyTorch in versione CPU anche se viene rilevata una GPU NVIDIA o AMD Radeon.

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

# Variabili d'ambiente obbligatorie per stabilita'/accelerazione ROCm 7.13 (AMD Radeon RX 9070 RDNA4 / RDNA3) e HuggingFace Hub
$env:TORCH_BLAS_PREFER_HIPBLASLT = '0'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'

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
# Aggiornamento PATH di sessione (WinGet Links + WinGet FFmpeg bin reale + Cargo)
# ==============================================================================
function Update-SessionEnvironmentPath {
    $machinePath   = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath      = [Environment]::GetEnvironmentVariable('Path', 'User')
    $cargoBin      = Join-Path $env:USERPROFILE '.cargo\bin'
    $wingetLinks   = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'
    $wingetPkgRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    $chocoBin      = 'C:\ProgramData\chocolatey\bin'

    $seen     = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    $combined = New-Object System.Collections.Generic.List[string]

    $addDir = {
        param([string]$CandidateDir)
        if (-not [string]::IsNullOrWhiteSpace($CandidateDir)) {
            $norm = $CandidateDir.Trim().TrimEnd('\')
            if ($norm -and (Test-Path -LiteralPath $norm) -and $seen.Add($norm)) {
                $combined.Add($norm)
            }
        }
    }

    # 1. Priorita' a .cargo\bin, cartella bin reale in WinGet\Packages\*FFmpeg* e WinGet\Links
    & $addDir $cargoBin

    if (Test-Path -LiteralPath $wingetPkgRoot) {
        $ffmpegPkgDirs = @(Get-ChildItem -Path $wingetPkgRoot -Filter '*FFmpeg*' -Directory -ErrorAction SilentlyContinue)
        foreach ($pkgDir in $ffmpegPkgDirs) {
            $ffmpegBins = @(
                Get-ChildItem -Path $pkgDir.FullName -Filter 'ffmpeg.exe' -Recurse -ErrorAction SilentlyContinue |
                    Select-Object -ExpandProperty DirectoryName -Unique
            )
            foreach ($binDir in $ffmpegBins) {
                & $addDir $binDir
            }
        }
    }

    & $addDir $wingetLinks

    # 2. Percorsi correnti di Processo, Macchina e Utente
    foreach ($raw in @($env:Path, $machinePath, $userPath)) {
        if (-not [string]::IsNullOrWhiteSpace($raw)) {
            foreach ($part in ($raw -split ';')) {
                $trimmed = $part.Trim().TrimEnd('\')
                if ($trimmed -and $seen.Add($trimmed)) {
                    $combined.Add($trimmed)
                }
            }
        }
    }

    & $addDir $chocoBin

    # 3. Eventuali pacchetti WinLibs/MinGW (windres.exe) o FFmpeg alternativi in WinGet\Packages
    if (Test-Path -LiteralPath $wingetPkgRoot) {
        $mingwPkgDirs = @(
            Get-ChildItem -Path $wingetPkgRoot -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '(?i)(WinLibs|MinGW|LLVM|MSYS)' }
        )
        foreach ($pkgDir in $mingwPkgDirs) {
            $windresBins = @(
                Get-ChildItem -Path $pkgDir.FullName -Filter 'windres.exe' -Recurse -ErrorAction SilentlyContinue |
                    Select-Object -ExpandProperty DirectoryName -Unique
            )
            foreach ($binDir in $windresBins) {
                & $addDir $binDir
            }
        }
    }

    $env:Path = ($combined -join ';')

    # Fallback se ffmpeg non e' ancora risolto e si trova in un pacchetto WinGet con nome diverso
    if ((Test-Path -LiteralPath $wingetPkgRoot) -and (-not (Get-Command 'ffmpeg' -ErrorAction SilentlyContinue))) {
        $fallbackBins = @(
            Get-ChildItem -Path $wingetPkgRoot -Filter 'ffmpeg.exe' -Recurse -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty DirectoryName -Unique
        )
        foreach ($binDir in $fallbackBins) {
            & $addDir $binDir
        }
        $env:Path = ($combined -join ';')
    }
}

# ==============================================================================
# Rilevamento Toolchain Rust (supporta sia x86_64-pc-windows-gnu sia msvc)
# ==============================================================================
function Get-RustToolchainInfo {
    $cargoCmd = Get-Command 'cargo' -ErrorAction SilentlyContinue
    $rustcCmd = Get-Command 'rustc' -ErrorAction SilentlyContinue
    if (-not $cargoCmd -or -not $rustcCmd) {
        return $null
    }

    $rustupExe = $null
    $rustupCmd = Get-Command 'rustup' -ErrorAction SilentlyContinue
    if ($rustupCmd) {
        $rustupExe = $rustupCmd.Source
    }
    else {
        $candidateRustup = Join-Path (Split-Path -Parent $cargoCmd.Source) 'rustup.exe'
        if (Test-Path -LiteralPath $candidateRustup) {
            $rustupExe = $candidateRustup
        }
    }

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $cargoOut   = & $cargoCmd.Source --version 2>$null
        $cargoExit  = $LASTEXITCODE
        $rustcVv    = & $rustcCmd.Source -vV 2>$null
        $rustcExit  = $LASTEXITCODE
        $sysrootOut = & $rustcCmd.Source --print sysroot 2>$null
    }
    finally {
        $ErrorActionPreference = $prevEap
    }

    if ($cargoExit -ne 0 -or $rustcExit -ne 0 -or -not $cargoOut) {
        return $null
    }

    $cargoVer = ($cargoOut | Select-Object -First 1).ToString().Trim()
    $rustcVer = ($rustcVv | Select-Object -First 1).ToString().Trim()
    $hostTriple = 'unknown'
    foreach ($line in $rustcVv) {
        if ($line -match '^host:\s*(.+)$') {
            $hostTriple = $Matches[1].Trim()
            break
        }
    }

    $sysroot = ''
    if ($sysrootOut) {
        $sysroot = ($sysrootOut | Select-Object -First 1).ToString().Trim()
    }

    $rustlibBin       = if ($sysroot -and $hostTriple -ne 'unknown') { Join-Path $sysroot ("lib\rustlib\{0}\bin" -f $hostTriple) } else { '' }
    $selfContainedBin = if ($rustlibBin) { Join-Path $rustlibBin 'self-contained' } else { '' }
    $llvmArExe        = if ($rustlibBin) { Join-Path $rustlibBin 'llvm-ar.exe' } else { '' }

    $abi = if ($hostTriple -like '*-gnu') {
        'gnu'
    }
    elseif ($hostTriple -like '*-msvc') {
        'msvc'
    }
    else {
        'other'
    }

    return [PSCustomObject]@{
        CargoExe         = $cargoCmd.Source
        RustcExe         = $rustcCmd.Source
        RustupExe        = $rustupExe
        CargoVersion     = $cargoVer
        RustcVersion     = $rustcVer
        HostTriple       = $hostTriple
        Abi              = $abi
        Sysroot          = $sysroot
        RustlibBin       = $rustlibBin
        SelfContainedBin = $selfContainedBin
        LlvmArExe        = $llvmArExe
    }
}

# ==============================================================================
# Configurazione automatica shim llvm-dlltool.exe per *-pc-windows-gnu senza MinGW
# (Evita il fallimento di self-contained\dlltool.exe quando manca as.exe esterno)
# ==============================================================================
function Enable-RustGnuSelfContainedShim {
    param(
        [Parameter(Mandatory = $true)][PSCustomObject]$RustInfo,
        [Parameter(Mandatory = $true)][string]$RustProjectDir
    )

    if ($RustInfo.HostTriple -notlike '*-pc-windows-gnu') {
        return
    }

    $hasExternalGcc = [bool](Get-Command 'gcc.exe' -ErrorAction SilentlyContinue)
    $hasExternalAs  = [bool](Get-Command 'as.exe' -ErrorAction SilentlyContinue)
    if ($hasExternalGcc -and $hasExternalAs) {
        Write-Info 'Toolchain MinGW esterno (gcc.exe + as.exe) rilevato nel PATH: shim llvm-dlltool non necessario.'
        return
    }

    Write-Info "Toolchain '$($RustInfo.HostTriple)' senza gcc.exe/as.exe esterni: configurazione shim nativo llvm-dlltool.exe..."

    $llvmAr = $RustInfo.LlvmArExe
    if (-not ($llvmAr -and (Test-Path -LiteralPath $llvmAr))) {
        if (-not $RustInfo.RustupExe -or (-not (Test-Path -LiteralPath $RustInfo.RustupExe))) {
            Write-ErrMsg "Componente 'llvm-tools' (llvm-ar.exe) assente e 'rustup.exe' non trovato per installarlo."
            exit 1
        }

        Write-Info "Installazione automatica componente 'llvm-tools' tramite rustup..."
        $compExit = Invoke-NativeCommand -FilePath $RustInfo.RustupExe -Arguments @('component', 'add', 'llvm-tools')
        if ($compExit -ne 0) {
            $compExit = Invoke-NativeCommand -FilePath $RustInfo.RustupExe -Arguments @('component', 'add', 'llvm-tools-preview')
        }
        if ($compExit -ne 0) {
            Write-ErrMsg "Impossibile installare 'llvm-tools' tramite rustup."
            exit 1
        }

        if (-not ($llvmAr -and (Test-Path -LiteralPath $llvmAr)) -and $RustInfo.Sysroot -and (Test-Path -LiteralPath $RustInfo.Sysroot)) {
            $foundAr = Get-ChildItem -Path $RustInfo.Sysroot -Filter 'llvm-ar.exe' -Recurse -ErrorAction SilentlyContinue |
                Select-Object -First 1 -ExpandProperty FullName
            if ($foundAr) {
                $llvmAr = $foundAr
            }
        }
    }

    if (-not ($llvmAr -and (Test-Path -LiteralPath $llvmAr))) {
        Write-ErrMsg "File 'llvm-ar.exe' non trovato nel sysroot Rust ($($RustInfo.Sysroot)) dopo 'rustup component add llvm-tools'."
        exit 1
    }

    $shimDir     = Join-Path $RustProjectDir 'target\llvm-shims'
    $shimDlltool = Join-Path $shimDir 'llvm-dlltool.exe'
    New-Item -ItemType Directory -Path $shimDir -Force | Out-Null

    $needCopyShim = $true
    if (Test-Path -LiteralPath $shimDlltool) {
        $srcLen = (Get-Item -LiteralPath $llvmAr).Length
        $dstLen = (Get-Item -LiteralPath $shimDlltool).Length
        if ($srcLen -eq $dstLen) {
            $needCopyShim = $false
        }
    }
    if ($needCopyShim) {
        Copy-Item -LiteralPath $llvmAr -Destination $shimDlltool -Force
    }

    # Copia eventuali DLL runtime di rustlib\bin (libgcc_s_seh-1.dll, libwinpthread-1.dll) accanto a llvm-dlltool.exe
    $llvmArDir = Split-Path -Parent $llvmAr
    if ($llvmArDir -and (Test-Path -LiteralPath $llvmArDir)) {
        $runtimeDlls = @(Get-ChildItem -Path $llvmArDir -Filter '*.dll' -File -ErrorAction SilentlyContinue)
        foreach ($dll in $runtimeDlls) {
            $dstDll = Join-Path $shimDir $dll.Name
            $copyDll = $true
            if (Test-Path -LiteralPath $dstDll) {
                if ((Get-Item -LiteralPath $dstDll).Length -eq $dll.Length) {
                    $copyDll = $false
                }
            }
            if ($copyDll) {
                Copy-Item -LiteralPath $dll.FullName -Destination $dstDll -Force
            }
        }
    }

    # Prepend in ordine: target\llvm-shims, rustlib\bin\self-contained, rustlib\bin, rustlib\bin\gcc-ld
    $gnuPathAdditions = New-Object System.Collections.Generic.List[string]
    foreach ($candidate in @(
        $shimDir,
        $RustInfo.SelfContainedBin,
        $RustInfo.RustlibBin,
        $(if ($RustInfo.RustlibBin) { Join-Path $RustInfo.RustlibBin 'gcc-ld' } else { '' })
    )) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            $gnuPathAdditions.Add($candidate.TrimEnd('\'))
        }
    }

    $existingParts = @($env:Path -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    $newPathList   = New-Object System.Collections.Generic.List[string]
    $pathSeen      = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in ($gnuPathAdditions + $existingParts)) {
        $normEntry = $entry.Trim().TrimEnd('\')
        if ($normEntry -and $pathSeen.Add($normEntry)) {
            $newPathList.Add($normEntry)
        }
    }
    $env:Path = ($newPathList -join ';')

    # Se il percorso contiene spazi (es. "C:\_antigravity\_audio SW\..."), usa il nome eseguibile risolto dal PATH
    # per evitare che Cargo spezzi RUSTFLAGS sugli spazi; altrimenti passa il path completo dello shim.
    $dlltoolSpec = if ($shimDlltool -match '\s') { 'llvm-dlltool.exe' } else { $shimDlltool }
    $currentRustFlags = [Environment]::GetEnvironmentVariable('RUSTFLAGS', 'Process')
    if ([string]::IsNullOrWhiteSpace($currentRustFlags)) {
        $env:RUSTFLAGS = "-C dlltool=$dlltoolSpec"
    }
    elseif ($currentRustFlags -notmatch '(^|\s)dlltool=') {
        $env:RUSTFLAGS = "$($currentRustFlags.Trim()) -C dlltool=$dlltoolSpec"
    }

    $encodedFlags = [Environment]::GetEnvironmentVariable('CARGO_ENCODED_RUSTFLAGS', 'Process')
    if (-not [string]::IsNullOrWhiteSpace($encodedFlags) -and ($encodedFlags -notmatch 'dlltool=')) {
        $sep = [char]0x1f
        $env:CARGO_ENCODED_RUSTFLAGS = "${encodedFlags}${sep}-C${sep}dlltool=${shimDlltool}"
    }

    Write-Ok "Shim GNU attivo: $shimDlltool | RUSTFLAGS=""$($env:RUSTFLAGS)"" + self-contained nel PATH"
}

# ==============================================================================
# Rilevamento GPU NVIDIA e AMD Radeon (tramite nvidia-smi + Win32_VideoController)
# ==============================================================================
function Get-GpuAccelerationInfo {
    # 1. Verifica GPU NVIDIA tramite nvidia-smi
    $smiCandidates = @()
    $smiCmd = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if ($smiCmd) {
        $smiCandidates += $smiCmd.Source
    }
    $smiCandidates += @(
        (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
        (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
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
                            Vendor         = 'NVIDIA'
                            Backend        = 'cuda'
                            Summary        = $firstGpu
                            IndexUrl       = 'https://download.pytorch.org/whl/cu126'
                            PreferPy312    = $false
                            UsePreRelease  = $false
                        }
                    }
                }
            }
            catch {
                # Continua la scansione WMI/CIM
            }
        }
    }

    # 2. Scansione Win32_VideoController per GPU AMD Radeon e NVIDIA
    $controllers = @()
    try {
        $controllers = @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue)
    }
    catch {
        $controllers = @()
    }

    foreach ($vc in $controllers) {
        $name = [string]$vc.Name
        $drv  = [string]$vc.DriverVersion
        if ([string]::IsNullOrWhiteSpace($name)) { continue }

        if ($name -match '(?i)NVIDIA') {
            return [PSCustomObject]@{
                Vendor        = 'NVIDIA'
                Backend       = 'cuda'
                Summary       = "$name (Driver $drv)"
                IndexUrl      = 'https://download.pytorch.org/whl/cu126'
                PreferPy312   = $false
                UsePreRelease = $false
            }
        }

        if ($name -match '(?i)\b(AMD|Radeon)\b') {
            # RDNA 4 (Radeon RX 9000 / RX 9070 / RX 9060 -> gfx120X-all)
            if ($name -match '(?i)RX\s*90\d{2}') {
                return [PSCustomObject]@{
                    Vendor        = 'AMD'
                    Backend       = 'rocm-rdna4'
                    Summary       = "$name (Driver $drv, RDNA 4 gfx120X)"
                    IndexUrl      = 'https://rocm.nightlies.amd.com/v2/gfx120X-all/'
                    PreferPy312   = $true
                    UsePreRelease = $true
                }
            }
            # RDNA 3 (Radeon RX 7000 -> gfx110X-all)
            if ($name -match '(?i)RX\s*7[6789]\d{2}') {
                return [PSCustomObject]@{
                    Vendor        = 'AMD'
                    Backend       = 'rocm-rdna3'
                    Summary       = "$name (Driver $drv, RDNA 3 gfx110X)"
                    IndexUrl      = 'https://rocm.nightlies.amd.com/v2/gfx110X-all/'
                    PreferPy312   = $true
                    UsePreRelease = $true
                }
            }
            # Altre GPU AMD Radeon rilevate: usa fallback sicuro CPU per evitare crash ComplexFloat STFT su DirectML
            return [PSCustomObject]@{
                Vendor        = 'AMD'
                Backend       = 'cpu-amd'
                Summary       = "$name (Driver $drv)"
                IndexUrl      = 'https://download.pytorch.org/whl/cpu'
                PreferPy312   = $true
                UsePreRelease = $false
            }
        }
    }

    return [PSCustomObject]@{
        Vendor        = 'CPU'
        Backend       = 'cpu'
        Summary       = $null
        IndexUrl      = 'https://download.pytorch.org/whl/cpu'
        PreferPy312   = $true
        UsePreRelease = $false
    }
}

# ==============================================================================
# Ricerca automatica di Python 3.11, 3.12 o 3.13
# (Preferisce Python 3.12 per compatibilita' wheel ROCm Windows / CUDA)
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
    param(
        [string]$ExistingVenvPython,
        [bool]$PreferPy312 = $true
    )

    $detectedUnsupported = New-Object System.Collections.Generic.List[string]

    # 1. Se esiste gia' una .venv valida con Python 3.11-3.13 (e Python 3.12 se richiesto da ROCm), usala
    $venvCandidate = $null
    if ($ExistingVenvPython -and (Test-Path -LiteralPath $ExistingVenvPython)) {
        $probe = Test-PythonInterpreter -CommandOrPath $ExistingVenvPython
        if ($probe -and $probe.IsSupported) {
            $venvCandidate = $probe
            if ((-not $ForceVenv) -and ((-not $PreferPy312) -or $probe.Minor -eq 12)) {
                return $venvCandidate
            }
        }
    }

    $verOrder = if ($PreferPy312) { @('-3.12', '-3.13', '-3.11') } else { @('-3.13', '-3.12', '-3.11') }
    $cmdOrder = if ($PreferPy312) {
        @('python3.12', 'python3.13', 'python3.11', 'python', 'python3')
    }
    else {
        @('python3.13', 'python3.12', 'python3.11', 'python', 'python3')
    }

    # 2. Prova tramite il Python Launcher ufficiale di Windows (py.exe)
    $pyLauncher = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($verTag in $verOrder) {
            $probe = Test-PythonInterpreter -CommandOrPath $pyLauncher.Source -ExtraArgs @($verTag)
            if ($probe -and $probe.IsSupported) {
                return $probe
            }
        }
    }

    # 3. Prova i comandi diretti presenti nel PATH
    foreach ($cmdName in $cmdOrder) {
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
    $minorList = if ($PreferPy312) { @('312', '313', '311') } else { @('313', '312', '311') }
    $standardPaths = New-Object System.Collections.Generic.List[string]
    foreach ($m in $minorList) {
        $standardPaths.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python$m\python.exe"))
        $standardPaths.Add((Join-Path $env:ProgramFiles "Python$m\python.exe"))
        $standardPaths.Add("C:\Python$m\python.exe")
    }

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
# Verifica integrita' dei pacchetti Python all'interno di .venv
# ==============================================================================
function Test-VenvPackagesReady {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][bool]$RequireGpuBuild
    )

    if (-not (Test-Path -LiteralPath $PythonPath)) {
        return $false
    }

    $checkScript = @"
import sys
mods = [
    'numpy', 'scipy', 'soundfile', 'soxr', 'torch', 'torchaudio',
    'demucs', 'pedalboard', 'noisereduce', 'pyloudnorm',
    'librosa', 'matchering', 'stftpitchshift', 'audiotsm'
]
for m in mods:
    __import__(m)
import torch
need_gpu = sys.argv[1] == '1'
has_gpu_runtime = bool(torch.version.cuda) or bool(getattr(torch.version, 'hip', None))
if need_gpu and not has_gpu_runtime:
    sys.exit(2)
print(f'torch={torch.__version__}|gpu_avail={torch.cuda.is_available()}')
"@

    try {
        $gpuArg = if ($RequireGpuBuild) { '1' } else { '0' }
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $out = & $PythonPath -c $checkScript $gpuArg 2>$null
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
$ExeRoot     = Join-Path $ProjectRoot 'StudioEQ_Master_Pro.exe'

Write-Banner 'StudioEQ Master Pro - Setup Ambiente AI GPU & Build Nativa Rust (64-Bit)'
Write-Info "Cartella progetto : $ProjectRoot"

Update-SessionEnvironmentPath

if (-not $CheckOnly -and -not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# ------------------------------------------------------------------------------
# PASSO 1: Verifica prerequisiti (Rust gnu/msvc, Python 3.11-3.13, FFmpeg, GPU)
# ------------------------------------------------------------------------------
Write-Step '1/4' 'Verifica dei prerequisiti di sistema (Rust gnu/msvc, Python 3.11-3.13, FFmpeg, GPU)...'

$missingTools = New-Object System.Collections.Generic.List[string]

# 1A. Verifica Rust (supporto sia per x86_64-pc-windows-gnu sia per x86_64-pc-windows-msvc)
$rustInfo = Get-RustToolchainInfo
if ($rustInfo) {
    $abiDesc = switch ($rustInfo.Abi) {
        'gnu'   { 'GNU MinGW self-contained (nessun Visual Studio C++ Build Tools richiesto)' }
        'msvc'  { 'MSVC ABI' }
        default { $rustInfo.HostTriple }
    }
    Write-Ok "Rust rilevato: $($rustInfo.CargoVersion) | $($rustInfo.RustcVersion) [$($rustInfo.HostTriple) - $abiDesc]"

    if ($rustInfo.HostTriple -like '*-pc-windows-gnu') {
        $hasExtGcc = [bool](Get-Command 'gcc.exe' -ErrorAction SilentlyContinue)
        $hasExtAs  = [bool](Get-Command 'as.exe' -ErrorAction SilentlyContinue)
        if ($hasExtGcc -and $hasExtAs) {
            Write-Ok 'Toolchain MinGW esterno (gcc.exe + as.exe) rilevato nel PATH.'
        }
        elseif ($rustInfo.LlvmArExe -and (Test-Path -LiteralPath $rustInfo.LlvmArExe) -and $rustInfo.SelfContainedBin -and (Test-Path -LiteralPath $rustInfo.SelfContainedBin)) {
            Write-Ok "Linker GNU self-contained pronto: llvm-tools (llvm-ar -> shim llvm-dlltool.exe) + self-contained disponibili senza MinGW esterno"
        }
        elseif ($rustInfo.RustupExe) {
            Write-Info "MinGW gcc.exe/as.exe non nel PATH: in fase di build verra' installato automaticamente 'llvm-tools' via rustup per lo shim llvm-dlltool.exe."
        }
        else {
            Write-WarnMsg "MinGW gcc.exe/as.exe e 'llvm-tools' non rilevati: esegui 'rustup component add llvm-tools' per abilitare lo shim llvm-dlltool.exe."
        }
    }
}
else {
    Write-ErrMsg 'Rust (cargo / rustc) NON trovato nel sistema.'
    $missingTools.Add("Rust (cargo): installa senza Visual Studio C++ Build Tools tramite 'rustup-init.exe -y --default-host x86_64-pc-windows-gnu --default-toolchain stable' (oppure 'winget install -e --id Rustlang.Rustup' per MSVC).")
}

# 1B. Rilevamento GPU NVIDIA / AMD Radeon (prima di Python per privilegiare Python 3.12 su ROCm)
$gpuStatus = Get-GpuAccelerationInfo
$useGpuAcceleration = (($gpuStatus.Backend -in @('cuda', 'rocm-rdna4', 'rocm-rdna3')) -and (-not $ForceCpu))

# 1C. Verifica Python 3.11 - 3.13
$pythonInfo = Find-CompatiblePython -ExistingVenvPython $VenvPython -PreferPy312 $gpuStatus.PreferPy312
if ($pythonInfo) {
    Write-Ok "Python compatibile rilevato: v$($pythonInfo.Version) ($($pythonInfo.Executable))"
}
else {
    Write-ErrMsg 'Python 3.11, 3.12 o 3.13 NON trovato.'
    $missingTools.Add("Python 3.12 (consigliato per GPU ROCm/CUDA): installa con 'winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements'.")
}

# 1D. Verifica FFmpeg
$ffmpegCmd = Get-Command 'ffmpeg' -ErrorAction SilentlyContinue
if ($ffmpegCmd) {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $ffmpegOut = & $ffmpegCmd.Source -version 2>$null
    $ffmpegExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    $ffmpegVer = ($ffmpegOut | Select-Object -First 1)

    if ($ffmpegExit -eq 0 -and $ffmpegVer) {
        Write-Ok "FFmpeg rilevato: $ffmpegVer ($($ffmpegCmd.Source))"
    }
    else {
        $missingTools.Add("FFmpeg non risponde correttamente. Reinstalla con: winget install -e --id Gyan.FFmpeg")
    }
}
else {
    Write-ErrMsg 'FFmpeg NON trovato nel PATH.'
    $missingTools.Add("FFmpeg: necessario per decodifica/esportazione audio. Installa con 'winget install -e --id Gyan.FFmpeg'.")
}

# 1E. Riepilogo accelerazione GPU rilevata
if ($ForceCpu) {
    Write-Info 'Modalita'' CPU forzata tramite parametro -ForceCpu.'
}
elseif ($gpuStatus.Backend -eq 'cuda') {
    Write-Ok "GPU NVIDIA rilevata: $($gpuStatus.Summary) -> accelerazione CUDA (cu126) attiva"
}
elseif ($gpuStatus.Backend -in @('rocm-rdna4', 'rocm-rdna3')) {
    if ($pythonInfo -and $pythonInfo.Minor -ne 12) {
        Write-WarnMsg "GPU AMD Radeon rilevata ($($gpuStatus.Summary)), ma ROCm Windows richiede Python 3.12 (trovato $($pythonInfo.Version)) -> fallback CPU attivo."
        $useGpuAcceleration = $false
    }
    else {
        Write-Ok "GPU AMD Radeon rilevata (Win32_VideoController): $($gpuStatus.Summary) -> accelerazione nativa ROCm/HIP attiva"
    }
}
elseif ($gpuStatus.Backend -eq 'cpu-amd') {
    Write-Info "GPU AMD Radeon rilevata ($($gpuStatus.Summary)) -> utilizzo PyTorch CPU multi-core (evita crash ComplexFloat STFT su DirectML)."
}
else {
    Write-Info 'Nessuna GPU dedicata NVIDIA/AMD ROCm rilevata -> verra'' usato PyTorch in modalita'' CPU multi-core.'
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
    if (Test-Path -LiteralPath $VenvPython) {
        $rocmProbe = @'
import os, torch, demucs, pedalboard, noisereduce, pyloudnorm, scipy, soundfile, librosa
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Multi-Core'
hip = getattr(torch.version, 'hip', None)
backend = f'ROCm/HIP {hip}' if hip else (f'CUDA {torch.version.cuda}' if torch.version.cuda else 'CPU')
hipblaslt = os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT', '')
print(f'PyTorch {torch.__version__} [{backend}] su {dev} | TORCH_BLAS_PREFER_HIPBLASLT={hipblaslt}')
'@
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $venvProbeOut  = & $VenvPython -c $rocmProbe 2>$null
        $venvProbeExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEap
        if ($venvProbeExit -eq 0 -and $venvProbeOut) {
            Write-Ok "Runtime .venv pronto: $(($venvProbeOut | Select-Object -Last 1).ToString().Trim())"
        }
        else {
            Write-WarnMsg 'Ambiente .venv presente ma alcuni pacchetti AI/DSP richiedono completamento (esegui senza -CheckOnly).'
        }
    }
    else {
        Write-Info 'Ambiente .venv non ancora creato (verra'' inizializzato automaticamente al primo setup).'
    }
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

    if ((-not $ForceVenv) -and (Test-VenvPackagesReady -PythonPath $VenvPython -RequireGpuBuild $useGpuAcceleration)) {
        Write-Ok 'Tutti i pacchetti Python AI/DSP (PyTorch GPU/CPU, Demucs, Pedalboard, Librosa, ecc.) risultano gia'' installati e pronti.'
    }
    else {
        Write-Info 'Aggiornamento di pip, setuptools e wheel nell''ambiente .venv...'
        $pipUpgradeExit = Invoke-NativeCommand -FilePath $VenvPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools<82', 'wheel')
        if ($pipUpgradeExit -ne 0) {
            Write-ErrMsg 'Aggiornamento di pip fallito.'
            exit 1
        }

        $installedGpuTorch = $false
        $activeExtraIndex = $null

        if ($useGpuAcceleration) {
            if ($gpuStatus.Backend -eq 'cuda') {
                Write-Info "Installazione di PyTorch + TorchAudio con accelerazione NVIDIA CUDA (cu126)..."
                $torchArgs = @('-m', 'pip', 'install', '--upgrade', 'torch', 'torchaudio', '--index-url', $gpuStatus.IndexUrl)
            }
            else {
                Write-Info "Installazione di PyTorch + TorchAudio con accelerazione AMD ROCm Windows ($($gpuStatus.Backend))..."
                $torchArgs = @('-m', 'pip', 'install', '--upgrade', '--pre', 'torch', 'torchaudio', '--index-url', $gpuStatus.IndexUrl)
            }

            $torchExit = Invoke-NativeCommand -FilePath $VenvPython -Arguments $torchArgs
            if ($torchExit -eq 0) {
                $installedGpuTorch = $true
                $activeExtraIndex = $gpuStatus.IndexUrl
                Write-Ok "PyTorch GPU ($($gpuStatus.Backend)) installato correttamente."
            }
            else {
                Write-WarnMsg "Installazione PyTorch GPU ($($gpuStatus.Backend)) non riuscita; passaggio automatico al fallback CPU..."
            }
        }

        if (-not $installedGpuTorch) {
            Write-Info 'Installazione di PyTorch + TorchAudio in versione CPU OpenMP...'
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
            'soxr',
            'librosa',
            'demucs',
            'pedalboard',
            'noisereduce',
            'pyloudnorm',
            'matchering',
            'stftpitchshift',
            'audiotsm'
        )

        Write-Info ("Installazione pacchetti Audio/AI: " + ($audioPackages -join ', ') + '...')
        if ($installedGpuTorch -and $activeExtraIndex) {
            $pkgArgs = @('-m', 'pip', 'install', '--extra-index-url', $activeExtraIndex)
            if ($gpuStatus.UsePreRelease) {
                $pkgArgs += '--pre'
            }
            $pkgArgs += $audioPackages
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
    }

    # Verifica finale di importazione + pre-cache modello htdemucs_ft
    $verifyPy = "import torch, torchaudio, demucs, pedalboard, noisereduce, pyloudnorm, soundfile, librosa, matchering, scipy, soxr, stftpitchshift, audiotsm; " +
                "from demucs.pretrained import get_model; get_model('htdemucs_ft'); " +
                "dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Multi-Core'; " +
                "print(f'PyTorch {torch.__version__} attivo su: {dev} | htdemucs_ft pronto in cache')"
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $verifyOut = & $VenvPython -c $verifyPy 2>$null
    $verifyExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap

    if ($verifyExit -ne 0) {
        Write-ErrMsg 'Verifica finale dei moduli Python o del modello htdemucs_ft fallita.'
        exit 1
    }
    $verifyMsg = ($verifyOut | Select-Object -Last 1)
    Write-Ok "Ambiente AI pronto -> $verifyMsg"
}

# ------------------------------------------------------------------------------
# PASSO 3: Compilazione nativa Rust (cargo build --release) e copia eseguibile
# ------------------------------------------------------------------------------
Write-Step '3/4' "Compilazione nativa di studio-eq-rust in modalita' Release ($($rustInfo.HostTriple))..."

if ($SkipBuild) {
    Write-Info 'Compilazione Rust saltata per via del parametro -SkipBuild.'
}
else {
    if (-not (Test-Path -LiteralPath $CargoToml)) {
        Write-ErrMsg "File Cargo.toml non trovato in: $CargoToml"
        exit 1
    }

    # Attiva automaticamente lo shim locale llvm-dlltool.exe + self-contained se la toolchain e' *-pc-windows-gnu senza MinGW esterno
    Enable-RustGnuSelfContainedShim -RustInfo $rustInfo -RustProjectDir $RustDir

    Push-Location -LiteralPath $RustDir
    try {
        $cargoBuildExit = Invoke-NativeCommand -FilePath $rustInfo.CargoExe -Arguments @('build', '--release')
    }
    finally {
        Pop-Location
    }

    $exeCandidates = @(
        (Join-Path $RustDir 'target\release\StudioEQ_Master_Pro.exe'),
        (Join-Path $RustDir ("target\{0}\release\StudioEQ_Master_Pro.exe" -f $rustInfo.HostTriple))
    )
    $ExeBuilt = $exeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

    if ($cargoBuildExit -ne 0 -or -not $ExeBuilt) {
        if ($rustInfo.Abi -eq 'msvc') {
            Write-ErrMsg 'Compilazione Rust (MSVC) fallita. Installa i Visual Studio C++ Build Tools oppure passa al toolchain self-contained GNU (rustup default stable-x86_64-pc-windows-gnu).'
        }
        else {
            Write-ErrMsg "Compilazione Rust ($($rustInfo.HostTriple)) fallita. Controlla i messaggi del compilatore sopra."
        }
        exit 1
    }

    Write-Ok "Binario compilato: $ExeBuilt"

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
