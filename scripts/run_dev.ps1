<#
Start Civitas Voice locally: TTS services, API and the web app, each in its own window.
  powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -SkipIndicF5
Then open http://localhost:5173
#>
param([switch]$SkipChatterbox, [switch]$SkipIndicF5, [switch]$SkipFrontend, [switch]$DummyTts)
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Start-Win($title, $cmd) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='$title'; Set-Location '$Root'; $cmd"
}

if ($DummyTts) {
    Start-Win "Civitas TTS (dummy)" "`$env:CIVITAS_TTS_ENGINE='dummy'; .\.venv\Scripts\python.exe -m uvicorn tts_service.app:app --host 127.0.0.1 --port 9002"
} else {
    if (-not $SkipIndicF5 -and (Test-Path ".venv-tts-indicf5")) {
        Start-Win "Civitas TTS IndicF5 :9001" "`$env:CIVITAS_TTS_ENGINE='indicf5'; .\.venv-tts-indicf5\Scripts\python.exe -m uvicorn tts_service.app:app --host 127.0.0.1 --port 9001"
    }
    if (-not $SkipChatterbox -and (Test-Path ".venv-tts-chatterbox")) {
        Start-Win "Civitas TTS Chatterbox :9002" "`$env:CIVITAS_TTS_ENGINE='chatterbox'; .\.venv-tts-chatterbox\Scripts\python.exe -m uvicorn tts_service.app:app --host 127.0.0.1 --port 9002"
    }
}
Start-Win "Civitas API :8000" ".\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000"
if (-not $SkipFrontend) { Start-Win "Civitas Web :5173" "cd frontend; npm run dev" }

Write-Host "Starting... the API window shows when models are loaded. Open http://localhost:5173"
Write-Host "Check everything with: .\.venv\Scripts\python.exe scripts\check_system.py"
