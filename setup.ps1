# Setup script for local-tts-studio (Windows / PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "Setting up Local TTS Studio..."

# Check Python version (chatterbox-tts requires 3.10-3.12)
$pythonVersion = python --version
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info < (3, 13) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.10-3.12 required, found $pythonVersion"
    exit 1
}
Write-Host "Found $pythonVersion"

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host "Virtual environment created"
} else {
    Write-Host "Virtual environment already exists"
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
. .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip | Out-Null

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt

# Pre-download Chatterbox model weights so the first app run doesn't stall
Write-Host "Pre-downloading Chatterbox model weights..."
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
$variant = if ($env:CHATTERBOX_VARIANT) { $env:CHATTERBOX_VARIANT } else { "turbo" }
if ($variant -eq "standard") {
    python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"
} else {
    python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained(device='cpu')"
}
Write-Host "Model weights cached"

Write-Host ""
Write-Host "Setup complete!"
Write-Host ""
Write-Host "To start the application:"
Write-Host "  1. Activate the virtual environment: .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run the app: streamlit run streamlit_app.py"
