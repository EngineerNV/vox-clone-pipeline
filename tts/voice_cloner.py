"""Voice cloning functionality for TTS."""

from pathlib import Path
from typing import Optional

import numpy as np

from audio import AudioIOHandler, AudioProcessor
from core.config import config
from tts.engine import TTSEngine


class VoiceCloner:
    """Handles voice cloning operations."""

    def __init__(self, tts_engine: Optional[TTSEngine] = None) -> None:
        """
        Initialize the voice cloner.

        Args:
            tts_engine: TTS engine instance (creates new one if not provided)
        """
        self.tts_engine = tts_engine or TTSEngine()
        self.audio_io = AudioIOHandler()
        self.audio_processor = AudioProcessor()

    def prepare_reference_audio(
        self,
        audio_path: Path,
        clean: bool = True,
    ) -> Path:
        """
        Prepare reference audio for voice cloning.

        Args:
            audio_path: Path to the reference audio file
            clean: Whether to apply audio cleaning

        Returns:
            Path to the prepared audio file

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio duration is invalid
        """
        # Load the audio
        audio_data, sample_rate = self.audio_io.load_audio(audio_path)

        # Validate duration
        if not self.audio_io.validate_audio_duration(audio_data, sample_rate):
            duration = self.audio_io.get_audio_duration(audio_data, sample_rate)
            raise ValueError(
                f"Audio duration ({duration:.2f}s) must be between "
                f"{config.audio.min_duration_seconds}s and "
                f"{config.audio.max_duration_seconds}s"
            )

        # Resample if needed
        if sample_rate != config.audio.sample_rate:
            audio_data, sample_rate = self.audio_processor.resample_audio(
                audio_data, sample_rate
            )

        # Apply cleaning if requested
        if clean:
            audio_data = self.audio_processor.clean_audio(
                audio_data,
                sample_rate,
                reduce_noise=True,
                normalize=True,
                trim_silence=True,
            )

        # Save the prepared audio to a temporary file (always WAV for compatibility)
        prepared_path = config.app.temp_dir / f"prepared_{audio_path.stem}.wav"
        self.audio_io.save_audio(audio_data, prepared_path, sample_rate)

        return prepared_path

    def clone_voice(
        self,
        text: str,
        reference_audio: Path,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        clean_reference: bool = True,
    ) -> np.ndarray:
        """
        Clone a voice and synthesize text.

        Args:
            text: Text to synthesize
            reference_audio: Path to reference speaker audio
            language: Language code
            temperature: Sampling temperature
            speed: Speech speed multiplier
            clean_reference: Whether to clean the reference audio

        Returns:
            Generated audio as numpy array

        Raises:
            ValueError: If inputs are invalid
        """
        if not self.tts_engine.is_multi_speaker():
            raise ValueError(
                f"Model {self.tts_engine.model_name} does not support voice cloning"
            )

        # Prepare the reference audio
        prepared_audio = self.prepare_reference_audio(reference_audio, clean=clean_reference)

        # Synthesize with the cloned voice
        audio = self.tts_engine.synthesize(
            text=text,
            speaker_wav=prepared_audio,
            language=language,
            temperature=temperature,
            speed=speed,
        )

        return audio
