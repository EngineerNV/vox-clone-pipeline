"""Configuration settings for the TTS studio."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    sample_rate: int = 22050
    max_duration_seconds: int = 30
    min_duration_seconds: float = 0.5
    supported_formats: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    noise_reduce_stationary: bool = True
    noise_reduce_prop_decrease: float = 0.8


@dataclass
class TTSConfig:
    """Configuration for TTS model."""

    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    language: str = "en"
    temperature: float = 0.75
    top_k: int = 50
    top_p: float = 0.85
    speed: float = 1.0
    use_cpu: bool = True  # macOS CPU-only


@dataclass
class AppConfig:
    """Configuration for the Streamlit application."""

    title: str = "Vox Clone Pipeline"
    page_icon: str = "🎙️"
    layout: str = "wide"
    max_file_size_mb: int = 25
    output_dir: Path = Path("outputs")
    temp_dir: Path = Path("temp")

    def __post_init__(self) -> None:
        """Ensure output directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class."""

    def __init__(self) -> None:
        """Initialize configuration."""
        self.audio = AudioConfig()
        self.tts = TTSConfig()
        self.app = AppConfig()

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment or defaults."""
        config = cls()

        # Override from environment variables if present
        if sample_rate := os.getenv("AUDIO_SAMPLE_RATE"):
            config.audio.sample_rate = int(sample_rate)

        if model_name := os.getenv("TTS_MODEL_NAME"):
            config.tts.model_name = model_name

        if language := os.getenv("TTS_LANGUAGE"):
            config.tts.language = language

        return config


# Global configuration instance
config = Config.load()
