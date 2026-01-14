"""
TTS module for vox-clone-pipeline.

Provides text-to-speech and voice cloning functionality.
"""

from tts.engine import TTSEngine
from tts.voice_cloner import VoiceCloner

__all__ = ["TTSEngine", "VoiceCloner"]
