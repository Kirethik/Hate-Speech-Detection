<#
Civitas Voice - one-time setup on Windows (Intel Core Ultra + Arc GPU).

Creates three Python environments (their dependencies conflict, so they are separate):
  .venv                    API + ASR + classifiers         (PyTorch XPU build)
  .venv-tts-chatterbox     Chatterbox voice cloning (en, hi, +21)
  .venv-tts-indicf5        IndicF5 voice cloning (ta, te, ml, kn, bn, mr, ...)

Usage (from the repo root, in PowerShell):
  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Cpu          # no Intel GPU build
  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -SkipIndicF5  # English/Hindi only

Needs: Python 3.11 (py launcher), Node.js 20+, Git, Intel Arc driver (latest), Ollama.
#>
param(
    [switch]$Cpu,
    [switch]$SkipChatterbox,
    [switch]$SkipIndicF5,
    [switch]$SkipFrontend,
    [string]$Python = "3.11"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Index = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/xpu" }

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { Write-Host "Missing: $cmd. $hint" -ForegroundColor Red; exit 1 }
}

Need "py" "Install Python $Python from python.org (tick 'py launcher')."
& py -$Python --version
if ($LASTEXITCODE -ne 0) { Write-Host "Python $Python not found. Install it from python.org." -ForegroundColor Red; exit 1 }

function New-Venv($dir) {
    if (-not (Test-Path "$dir\Scripts\python.exe")) { & py -$Python -m venv $dir }
    & "$dir\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools | Out-Null
    return "$dir\Scripts\python.exe"
}

# ---------------------------------------------------------------- main env
Step "Main environment (.venv): PyTorch from $Index"
$py = New-Venv ".venv"
& $py -m pip install torch torchaudio --index-url $Index
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
Step "Main environment: Civitas requirements"
& $py -m pip install -r requirements-voice.txt
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
& $py -c "import torch; x = hasattr(torch,'xpu') and torch.xpu.is_available(); print('torch', torch.__version__, '| Intel XPU available:', x); print(torch.xpu.get_device_name(0) if x else 'running on CPU')"

# ---------------------------------------------------------------- TTS envs
if (-not $SkipChatterbox) {
    Step "Chatterbox environment (.venv-tts-chatterbox)"
    $cb = New-Venv ".venv-tts-chatterbox"
    & $cb -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url $Index
    if ($LASTEXITCODE -ne 0) {
        Write-Host "torch 2.6.0 not available from $Index; using the CPU build for Chatterbox" -ForegroundColor Yellow
        & $cb -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    }
    & $cb -m pip install -r tts_service\requirements-chatterbox.txt
    if ($LASTEXITCODE -ne 0) { throw "chatterbox install failed" }
}
if (-not $SkipIndicF5) {
    Step "IndicF5 environment (.venv-tts-indicf5)"
    $f5 = New-Venv ".venv-tts-indicf5"
    & $f5 -m pip install torch torchaudio --index-url $Index
    & $f5 -m pip install -r tts_service\requirements-indicf5.txt
    if ($LASTEXITCODE -ne 0) { throw "IndicF5 install failed" }
}

# ---------------------------------------------------------------- frontend
if (-not $SkipFrontend) {
    Need "npm" "Install Node.js 20+ from nodejs.org."
    Step "Frontend (npm install)"
    Push-Location frontend; npm install; Pop-Location
}

# ---------------------------------------------------------------- config + models
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    $tok = Read-Host "Paste your Hugging Face token (or press Enter to add it to .env later)"
    if ($tok) { (Get-Content .env) -replace '^HF_TOKEN=.*', "HF_TOKEN=$tok" | Set-Content .env -Encoding ascii }
}

Step "Ollama model"
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $model = ((Get-Content .env | Select-String '^CIVITAS_LLM_MODEL=') -replace 'CIVITAS_LLM_MODEL=', '').Trim()
    if (-not $model) { $model = "gemma4:e4b" }
    ollama pull $model
} else {
    Write-Host "Ollama not found. Install it from ollama.com, then run: ollama pull gemma4:e4b" -ForegroundColor Yellow
}

Step "Downloading models (first time only; several GB)"
& $py scripts\download_models.py
if (-not $SkipIndicF5) { & ".venv-tts-indicf5\Scripts\python.exe" scripts\download_models.py --tts indicf5 }
if (-not $SkipChatterbox) { & ".venv-tts-chatterbox\Scripts\python.exe" scripts\download_models.py --tts chatterbox }

Step "Done. Start everything with:  powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1"
