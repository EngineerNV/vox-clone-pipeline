#!/usr/bin/env python3
"""
CLI Test Script for TTS Pipeline

This script validates the full TTS pipeline works end-to-end without
requiring the Streamlit UI. It creates a synthetic audio file and tests:
1. Import verification for all modules
2. Audio I/O operations
3. Audio processing (cleaning)
4. TTS engine initialization
5. Voice cloning synthesis

Usage:
    python scripts/test_tts_pipeline.py [--full]
    
    --full: Run full TTS synthesis test (downloads ~2GB model on first run)
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


def print_status(message: str, success: bool = True) -> None:
    """Print a status message with emoji indicator."""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {message}")


def test_imports() -> bool:
    """Test that all modules can be imported."""
    print("\n=== Testing Module Imports ===")
    
    modules = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("soundfile", "soundfile"),
        ("scipy", "scipy"),
        ("noisereduce", "noisereduce"),
        ("TTS.api", "TTS (coqui-tts)"),
        ("transformers", "transformers"),
        ("streamlit", "streamlit"),
    ]
    
    all_passed = True
    for module_path, display_name in modules:
        try:
            __import__(module_path)
            print_status(f"{display_name} imported successfully")
        except ImportError as e:
            print_status(f"{display_name} import failed: {e}", success=False)
            all_passed = False
    
    return all_passed


def test_project_imports() -> bool:
    """Test that project modules can be imported."""
    print("\n=== Testing Project Module Imports ===")
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    modules = [
        "core.config",
        "core.utils",
        "audio.io_handler",
        "audio.processor",
        "tts.engine",
        "tts.voice_cloner",
        "orchestration.tts_orchestrator",
        "app.ui_components",
        "app.pages",
    ]
    
    all_passed = True
    for module in modules:
        try:
            __import__(module)
            print_status(f"{module} imported successfully")
        except Exception as e:
            print_status(f"{module} import failed: {e}", success=False)
            all_passed = False
    
    return all_passed


def create_test_audio(duration: float = 3.0, sample_rate: int = 22050) -> tuple[np.ndarray, Path]:
    """Create a synthetic test audio file (sine wave with noise)."""
    print("\n=== Creating Test Audio ===")
    
    # Generate a simple sine wave with some noise
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    
    # 440 Hz sine wave (A4 note) + harmonics + noise
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.15 * np.sin(2 * np.pi * 880 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)
    
    # Normalize
    audio = audio / np.abs(audio).max() * 0.8
    
    # Save to temp file
    import soundfile as sf
    temp_file = Path(tempfile.mktemp(suffix=".wav"))
    sf.write(str(temp_file), audio, sample_rate)
    
    print_status(f"Created test audio: {temp_file} ({duration}s, {sample_rate}Hz)")
    
    return audio, temp_file


def test_audio_io(test_audio_path: Path) -> bool:
    """Test audio I/O operations."""
    print("\n=== Testing Audio I/O ===")
    
    try:
        from audio.io_handler import AudioIOHandler
        
        handler = AudioIOHandler()
        
        # Test loading
        audio_data, sample_rate = handler.load_audio(test_audio_path)
        print_status(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
        
        # Test duration calculation
        duration = handler.get_audio_duration(audio_data, sample_rate)
        print_status(f"Duration: {duration:.2f}s")
        
        # Test saving
        output_path = Path(tempfile.mktemp(suffix="_saved.wav"))
        handler.save_audio(audio_data, output_path, sample_rate)
        print_status(f"Saved audio to: {output_path}")
        
        # Clean up
        output_path.unlink()
        
        return True
    except Exception as e:
        print_status(f"Audio I/O test failed: {e}", success=False)
        return False


def test_audio_processing(test_audio_path: Path) -> bool:
    """Test audio processing operations."""
    print("\n=== Testing Audio Processing ===")
    
    try:
        from audio.io_handler import AudioIOHandler
        from audio.processor import AudioProcessor
        
        handler = AudioIOHandler()
        processor = AudioProcessor()
        
        # Load test audio
        audio_data, sample_rate = handler.load_audio(test_audio_path)
        
        # Test normalization
        normalized = processor.normalize_audio(audio_data)
        print_status(f"Normalized audio: max={np.abs(normalized).max():.3f}")
        
        # Test noise reduction
        denoised = processor.reduce_noise(audio_data, sample_rate)
        print_status(f"Noise reduction applied: {len(denoised)} samples")
        
        # Test resampling
        resampled, new_sr = processor.resample_audio(audio_data, sample_rate, 16000)
        print_status(f"Resampled: {sample_rate}Hz -> {new_sr}Hz ({len(resampled)} samples)")
        
        # Test full cleaning pipeline
        cleaned = processor.clean_audio(
            audio_data, sample_rate,
            reduce_noise=True, normalize=True, trim_silence=True
        )
        print_status(f"Full cleaning pipeline: {len(audio_data)} -> {len(cleaned)} samples")
        
        return True
    except Exception as e:
        print_status(f"Audio processing test failed: {e}", success=False)
        import traceback
        traceback.print_exc()
        return False


def test_tts_engine_init() -> bool:
    """Test TTS engine initialization (does NOT load model)."""
    print("\n=== Testing TTS Engine Initialization ===")
    
    try:
        from tts.engine import TTSEngine
        
        # Just test that we can create the engine object
        engine = TTSEngine(use_cpu=True)
        print_status(f"TTSEngine created with model: {engine.model_name}")
        print_status(f"Device: {engine._device}")
        
        # Don't load the model (it's ~2GB)
        print_status("Model loading deferred (use --full to test synthesis)")
        
        return True
    except Exception as e:
        print_status(f"TTS engine init failed: {e}", success=False)
        import traceback
        traceback.print_exc()
        return False


def test_full_synthesis(test_audio_path: Path) -> bool:
    """Test full TTS synthesis (requires model download)."""
    print("\n=== Testing Full TTS Synthesis ===")
    print("⚠️  This will download the XTTS v2 model (~2GB) on first run...")
    
    try:
        from orchestration.tts_orchestrator import TTSOrchestrator
        
        orchestrator = TTSOrchestrator()
        
        # Generate speech
        output_path = orchestrator.process_and_clone(
            text="Hello, this is a test of the voice cloning pipeline.",
            reference_audio_path=test_audio_path,
            language="en",
            clean_reference=True,
        )
        
        print_status(f"Generated audio saved to: {output_path}")
        
        # Verify output exists and has content
        if output_path.exists() and output_path.stat().st_size > 0:
            from audio.io_handler import AudioIOHandler
            handler = AudioIOHandler()
            audio, sr = handler.load_audio(output_path)
            duration = len(audio) / sr
            print_status(f"Output audio: {duration:.2f}s at {sr}Hz")
            return True
        else:
            print_status("Output file is empty or missing", success=False)
            return False
            
    except Exception as e:
        print_status(f"Full synthesis test failed: {e}", success=False)
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test TTS Pipeline")
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Run full synthesis test (downloads ~2GB model)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙️  Vox Clone Pipeline - Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test imports
    results["imports"] = test_imports()
    results["project_imports"] = test_project_imports()
    
    if not results["imports"] or not results["project_imports"]:
        print("\n❌ Critical import failures. Cannot continue.")
        return 1
    
    # Create test audio
    _, test_audio_path = create_test_audio()
    
    try:
        # Test audio I/O
        results["audio_io"] = test_audio_io(test_audio_path)
        
        # Test audio processing
        results["audio_processing"] = test_audio_processing(test_audio_path)
        
        # Test TTS engine init
        results["tts_engine_init"] = test_tts_engine_init()
        
        # Optional: full synthesis test
        if args.full:
            results["full_synthesis"] = test_full_synthesis(test_audio_path)
        else:
            print("\n=== Skipping Full Synthesis Test ===")
            print("ℹ️  Use --full flag to run complete TTS synthesis test")
    
    finally:
        # Clean up test audio
        if test_audio_path.exists():
            test_audio_path.unlink()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
