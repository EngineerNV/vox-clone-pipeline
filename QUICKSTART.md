# Quick Start Guide

## Installation

### Option 1: Using the setup script (Recommended)

```bash
# Clone the repository
git clone https://github.com/EngineerNV/vox-clone-pipeline.git
cd vox-clone-pipeline

# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Start the app
streamlit run streamlit_app.py
```

### Option 2: Manual setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## First Run

1. The app will open in your browser at `http://localhost:8501`
2. On first use, the XTTS v2 model (~2GB) will download automatically
3. This may take several minutes depending on your internet connection

## Basic Usage

### Voice Cloning

1. Go to the "Voice Cloning" tab
2. Upload a reference audio file (5-30 seconds recommended)
3. Enter the text you want to synthesize
4. Adjust settings if needed (optional)
5. Click "Generate Speech"
6. Wait for processing (10-30 seconds)
7. Play and download your generated audio

### Audio Cleaning

1. Go to the "Audio Cleaning" tab
2. Upload an audio file
3. Select cleaning operations (noise reduction, normalization, trim silence)
4. Click "Clean Audio"
5. Play and download the cleaned audio

## Tips

- **Reference Audio Quality**: Better quality reference = better cloning results
- **Reference Audio Length**: 5-30 seconds is ideal, clear speech works best
- **Text Length**: Shorter texts generate faster (aim for < 500 characters)
- **Languages**: Select the correct language for best results
- **Temperature**: Lower (0.5-0.7) = more consistent, Higher (0.8-1.0) = more varied

## Troubleshooting

**Model download fails**:
- Check your internet connection
- Ensure ~2GB of free disk space
- Try running again

**Generation is slow**:
- This is normal on CPU-only systems
- M1/M2 Macs perform better than Intel Macs
- Consider shorter text for faster results

**Audio quality issues**:
- Try cleaning the reference audio first
- Ensure reference audio is clear and noise-free
- Use a longer/better quality reference sample

## Next Steps

- Read the full [README.md](README.md) for detailed information
- Explore the codebase to understand the architecture
- Check out the tests for usage examples
- Customize settings in `core/config.py`

## Getting Help

- Check the README for detailed documentation
- Review the code comments and docstrings
- Open an issue on GitHub if you encounter problems
