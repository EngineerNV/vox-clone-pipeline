"""TTS orchestrator coordinating audio processing and TTS operations."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from audio import AudioIOHandler, AudioProcessor
from core.config import config
from core.utils import sanitize_filename
from tts import TTSEngine, VoiceCloner


class TTSOrchestrator:
    """Orchestrates TTS operations, coordinating audio processing and voice cloning."""

    def __init__(self) -> None:
        """Initialize the TTS orchestrator with required components."""
        self.tts_engine = TTSEngine()
        self.voice_cloner = VoiceCloner(self.tts_engine)
        self.audio_io = AudioIOHandler()
        self.audio_processor = AudioProcessor()

    def process_and_clone(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Optional[Path] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        clean_reference: bool = True,
    ) -> Path:
        """
        Process reference audio and generate cloned speech.

        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference speaker audio
            output_path: Path to save output audio (auto-generated if not provided)
            language: Language code
            temperature: Sampling temperature
            speed: Speech speed multiplier
            clean_reference: Whether to clean the reference audio

        Returns:
            Path to the generated audio file

        Raises:
            ValueError: If inputs are invalid
        """
        # Generate cloned audio
        audio_data = self.voice_cloner.clone_voice(
            text=text,
            reference_audio=reference_audio_path,
            language=language,
            temperature=temperature,
            speed=speed,
            clean_reference=clean_reference,
        )

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_output_{timestamp}.wav"
            output_path = config.app.output_dir / filename

        # Save the generated audio
        self.audio_io.save_audio(audio_data, output_path)

        return output_path

    def clean_audio_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        reduce_noise: bool = True,
        normalize: bool = True,
        trim_silence: bool = True,
    ) -> Path:
        """
        Clean an audio file.

        Args:
            input_path: Path to input audio file
            output_path: Path to save cleaned audio (auto-generated if not provided)
            reduce_noise: Whether to apply noise reduction
            normalize: Whether to normalize audio
            trim_silence: Whether to trim silence

        Returns:
            Path to the cleaned audio file
        """
        # Load audio
        audio_data, sample_rate = self.audio_io.load_audio(input_path)

        # Clean audio
        cleaned = self.audio_processor.clean_audio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            reduce_noise=reduce_noise,
            normalize=normalize,
            trim_silence=trim_silence,
        )

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name = input_path.stem
            safe_name = sanitize_filename(original_name)
            filename = f"{safe_name}_cleaned_{timestamp}.wav"
            output_path = config.app.output_dir / filename

        # Save cleaned audio
        self.audio_io.save_audio(cleaned, output_path, sample_rate)

        return output_path

    def get_audio_info(self, audio_path: Path) -> dict[str, any]:
        """
        Get information about an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio information
        """
        audio_data, sample_rate = self.audio_io.load_audio(audio_path)
        duration = self.audio_io.get_audio_duration(audio_data, sample_rate)

        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "samples": len(audio_data),
            "is_valid_duration": self.audio_io.validate_audio_duration(audio_data, sample_rate),
        }
