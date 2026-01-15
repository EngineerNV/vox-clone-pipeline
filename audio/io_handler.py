"""Audio I/O operations for loading and saving audio files."""

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from core.config import config


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


class AudioIOHandler:
    """Handles reading and writing audio files."""

    @staticmethod
    def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
        """
        Load an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            ValueError: If the file format is not supported
            FileNotFoundError: If the file does not exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in config.audio.supported_formats:
            raise ValueError(f"Unsupported audio format: {suffix}")

        # For WAV and FLAC, use soundfile directly
        if suffix in (".wav", ".flac"):
            audio_data, sample_rate = sf.read(str(file_path))
        else:
            # For M4A, MP3, OGG - use pydub (requires ffmpeg)
            if not _check_ffmpeg_available():
                raise RuntimeError(
                    f"ffmpeg is required to load {suffix} files. "
                    "Install it with: brew install ffmpeg (macOS) or "
                    "apt install ffmpeg (Linux)"
                )
            try:
                audio_segment = AudioSegment.from_file(str(file_path))
            except CouldntDecodeError as e:
                raise ValueError(
                    f"Could not decode {suffix} file. Ensure ffmpeg supports "
                    f"this format and the file is not corrupted: {e}"
                ) from e
            # Convert to mono if stereo
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            # Set sample rate
            audio_segment = audio_segment.set_frame_rate(config.audio.sample_rate)
            # Convert to numpy array
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / (2**15)  # Normalize to [-1, 1]
            sample_rate = config.audio.sample_rate

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, sample_rate

    @staticmethod
    def save_audio(
        audio_data: np.ndarray,
        file_path: Path,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        Save audio data to a file.

        Args:
            audio_data: Audio data as numpy array
            file_path: Path where to save the audio
            sample_rate: Sample rate (uses config default if not provided)
        """
        if sample_rate is None:
            sample_rate = config.audio.sample_rate

        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save using soundfile
        sf.write(str(file_path), audio_data, sample_rate)

    @staticmethod
    def get_audio_duration(audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Calculate the duration of audio data.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate

        Returns:
            Duration in seconds
        """
        return len(audio_data) / sample_rate

    @staticmethod
    def validate_audio_duration(audio_data: np.ndarray, sample_rate: int) -> bool:
        """
        Validate that audio duration is within acceptable range.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate

        Returns:
            True if duration is valid, False otherwise
        """
        duration = AudioIOHandler.get_audio_duration(audio_data, sample_rate)
        return (
            config.audio.min_duration_seconds <= duration <= config.audio.max_duration_seconds
        )
