"""
TTS module for vox-clone-pipeline.

Provides text-to-speech and voice cloning functionality.
"""

from tts.engine import BaseTTSEngine, ChatterboxEngine, XTTSEngine, create_engine
from tts.voice_cloner import VoiceCloner

__all__ = [
    "BaseTTSEngine",
    "ChatterboxEngine",
    "XTTSEngine",
    "VoiceCloner",
    "create_engine",
]
