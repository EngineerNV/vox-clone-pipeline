"""TTS engine for text-to-speech synthesis."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from TTS.api import TTS

from core.config import config


class TTSEngine:
    """Manages TTS model and synthesis operations."""

    def __init__(self, model_name: Optional[str] = None, use_cpu: bool = True) -> None:
        """
        Initialize the TTS engine.

        Args:
            model_name: Name of the TTS model to use
            use_cpu: Whether to use CPU (True for macOS compatibility)
        """
        self.model_name = model_name or config.tts.model_name
        self.use_cpu = use_cpu
        self._model: Optional[TTS] = None
        self._device = "cpu" if use_cpu else "cuda"

    @property
    def model(self) -> TTS:
        """
        Lazy-load and return the TTS model.

        Returns:
            Initialized TTS model
        """
        if self._model is None:
            self._model = TTS(model_name=self.model_name, progress_bar=True).to(self._device)
        return self._model

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[Path] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker_wav: Path to reference speaker audio for voice cloning
            language: Language code (e.g., 'en', 'es')
            temperature: Sampling temperature
            speed: Speech speed multiplier

        Returns:
            Generated audio as numpy array

        Raises:
            ValueError: If text is empty or speaker_wav is required but not provided
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Use config defaults if not provided
        language = language or config.tts.language
        temperature = temperature if temperature is not None else config.tts.temperature
        speed = speed if speed is not None else config.tts.speed

        # This is a voice cloning app - speaker reference is always required
        if speaker_wav is None:
            raise ValueError(
                "Speaker reference audio is required for voice cloning. "
                "This application only supports voice cloning synthesis."
            )

        # Synthesize with voice cloning
        audio = self.model.tts(
            text=text,
            speaker_wav=str(speaker_wav),
            language=language,
            temperature=temperature,
            speed=speed,
        )

        # Convert to numpy array if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        return np.array(audio, dtype=np.float32)

    def get_available_languages(self) -> list[str]:
        """
        Get list of supported languages for the current model.

        Returns:
            List of language codes
        """
        if hasattr(self.model, "languages"):
            return self.model.languages
        return ["en"]  # Default fallback

    def is_multi_speaker(self) -> bool:
        """
        Check if the model supports multiple speakers/voice cloning.

        Returns:
            True if the model supports voice cloning
        """
        return "xtts" in self.model_name.lower() or "vits" in self.model_name.lower()
