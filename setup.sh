#!/bin/bash
# Setup script for local-tts-studio

set -e

echo "🎙️ Setting up Local TTS Studio..."

# Check Python version (chatterbox-tts requires >=3.10)
python_version=$(python3 --version 2>&1 | awk '{print $2}')
if ! python3 -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info < (3, 13) else 1)'; then
    echo "❌ Python 3.10-3.12 required, found $python_version"
    exit 1
fi
echo "✓ Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the app: streamlit run streamlit_app.py"
echo ""
echo "Note: The TTS model (~2GB) will be downloaded on first run."
