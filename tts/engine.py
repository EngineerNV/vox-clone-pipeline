"""TTS engines for text-to-speech synthesis.

Two engines are supported:

- ChatterboxEngine (default): Resemble AI's Chatterbox, MIT-licensed, strong
  zero-shot accent preservation. English only. Runs on CUDA, Apple Silicon
  (MPS), or CPU. Install via requirements.txt.
- XTTSEngine (legacy): Coqui XTTS v2, multilingual. Its dependency stack
  (transformers 4.x) conflicts with Chatterbox (transformers 5.x), so it
  needs its own virtualenv. Install via requirements-xtts.txt.

Select with the TTS_ENGINE environment variable ("chatterbox" or "xtts").
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Some MPS kernels are missing on older torch builds; fall back to CPU for
# those ops instead of crashing. Must be set before the first MPS op runs.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from core.config import config


def resolve_device(preference: str = "auto") -> str:
    """
    Resolve the torch device to run inference on.

    Args:
        preference: "auto", "cuda", "mps", or "cpu". "auto" picks the best
            available device (CUDA > Apple Silicon GPU via MPS > CPU).

    Returns:
        Device string usable with torch.
    """
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaseTTSEngine(ABC):
    """Common interface for TTS engines."""

    #: Sample rate the reference audio should be resampled to before cloning,
    #: or None to pass the reference at its original rate.
    reference_sample_rate: Optional[int] = None

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[Path] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        **kwargs: object,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker_wav: Path to reference speaker audio for voice cloning
            language: Language code (e.g., 'en', 'es')
            temperature: Sampling temperature
            speed: Speech speed multiplier
            **kwargs: Engine-specific parameters

        Returns:
            Tuple of (audio as float32 numpy array, sample rate in Hz)

        Raises:
            ValueError: If text is empty or speaker_wav is required but not provided
        """

    @abstractmethod
    def get_available_languages(self) -> list[str]:
        """Return supported language codes."""

    @abstractmethod
    def supports_voice_cloning(self) -> bool:
        """Return True if the engine can clone a voice from reference audio."""

    @staticmethod
    def _validate_inputs(text: str, speaker_wav: Optional[Path]) -> None:
        if not text.strip():
            raise ValueError("Text cannot be empty")
        if speaker_wav is None:
            raise ValueError(
                "Speaker reference audio is required for voice cloning. "
                "This application only supports voice cloning synthesis."
            )

    @staticmethod
    def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """Time-stretch audio for engines without a native speed control."""
        if abs(speed - 1.0) < 1e-3:
            return audio
        import librosa

        return librosa.effects.time_stretch(audio, rate=speed)


class ChatterboxEngine(BaseTTSEngine):
    """Resemble AI Chatterbox engine (zero-shot voice cloning, English only).

    Variants:
        - "turbo" (default): 350M params, ~2-3 GB peak memory. Ignores
          exaggeration/cfg_weight.
        - "standard": ~0.5B params, ~4-5 GB peak memory. Supports
          exaggeration and cfg_weight for expressiveness control.
    """

    # Chatterbox resamples references internally; pass the original audio.
    reference_sample_rate = None

    def __init__(self, variant: Optional[str] = None, device: Optional[str] = None) -> None:
        self.variant = variant or config.tts.chatterbox_variant
        if self.variant not in ("turbo", "standard"):
            raise ValueError(
                f"Unknown Chatterbox variant: {self.variant!r} (expected 'turbo' or 'standard')"
            )
        self._device = resolve_device(device or config.tts.device)
        self._model = None

    @property
    def model(self):
        """Lazy-load and return the Chatterbox model."""
        if self._model is None:
            try:
                if self.variant == "turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS as model_cls
                else:
                    from chatterbox.tts import ChatterboxTTS as model_cls
            except ImportError as e:
                raise RuntimeError(
                    "chatterbox-tts is not installed. Install the default engine "
                    "dependencies with: pip install -r requirements.txt"
                ) from e
            self._model = model_cls.from_pretrained(device=self._device)
        return self._model

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[Path] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        **kwargs: object,
    ) -> tuple[np.ndarray, int]:
        self._validate_inputs(text, speaker_wav)

        language = language or config.tts.language
        if language not in self.get_available_languages():
            raise ValueError(
                f"Chatterbox ({self.variant}) only supports English ('en'), "
                f"got language={language!r}. Use the XTTS engine for other languages."
            )

        temperature = temperature if temperature is not None else config.tts.temperature
        speed = speed if speed is not None else config.tts.speed

        generate_kwargs: dict = {"temperature": temperature}
        if self.variant == "standard":
            generate_kwargs["exaggeration"] = (
                exaggeration if exaggeration is not None else config.tts.exaggeration
            )
            generate_kwargs["cfg_weight"] = (
                cfg_weight if cfg_weight is not None else config.tts.cfg_weight
            )

        wav = self.model.generate(
            text,
            audio_prompt_path=str(speaker_wav),
            **generate_kwargs,
        )

        audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        audio = self._apply_speed(audio, speed)
        return audio, self.model.sr

    def get_available_languages(self) -> list[str]:
        return ["en"]

    def supports_voice_cloning(self) -> bool:
        return True


class XTTSEngine(BaseTTSEngine):
    """Legacy Coqui XTTS v2 engine (multilingual voice cloning).

    Requires the coqui-tts package (requirements-xtts.txt), which cannot be
    installed alongside chatterbox-tts in the same environment. XTTS is kept
    on CPU/CUDA only; its MPS support is unreliable.
    """

    # XTTS expects the reference at the app's processing rate (legacy behavior).
    reference_sample_rate = config.audio.sample_rate

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_name = model_name or config.tts.model_name
        device = resolve_device(device or config.tts.device)
        self._device = device if device == "cuda" else "cpu"
        self._model = None

    @property
    def model(self):
        """Lazy-load and return the Coqui TTS model."""
        if self._model is None:
            try:
                from TTS.api import TTS
            except ImportError as e:
                raise RuntimeError(
                    "coqui-tts is not installed. The XTTS engine needs its own "
                    "virtualenv (its dependencies conflict with chatterbox-tts): "
                    "pip install -r requirements-xtts.txt"
                ) from e
            self._model = TTS(model_name=self.model_name, progress_bar=True).to(self._device)
        return self._model

    @property
    def output_sample_rate(self) -> int:
        """Sample rate of the audio the loaded model generates."""
        sr = getattr(getattr(self.model, "synthesizer", None), "output_sample_rate", None)
        return int(sr) if sr else 24000  # XTTS v2 generates 24 kHz audio

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[Path] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        **kwargs: object,
    ) -> tuple[np.ndarray, int]:
        self._validate_inputs(text, speaker_wav)

        language = language or config.tts.language
        temperature = temperature if temperature is not None else config.tts.temperature
        speed = speed if speed is not None else config.tts.speed

        audio = self.model.tts(
            text=text,
            speaker_wav=str(speaker_wav),
            language=language,
            temperature=temperature,
            speed=speed,
        )

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        return np.array(audio, dtype=np.float32), self.output_sample_rate

    def get_available_languages(self) -> list[str]:
        if hasattr(self.model, "languages"):
            return self.model.languages
        return ["en"]

    def supports_voice_cloning(self) -> bool:
        return "xtts" in self.model_name.lower() or "vits" in self.model_name.lower()


_ENGINES = {
    "chatterbox": ChatterboxEngine,
    "xtts": XTTSEngine,
}


def create_engine(engine_type: Optional[str] = None, **engine_kwargs: object) -> BaseTTSEngine:
    """
    Create a TTS engine instance.

    Args:
        engine_type: "chatterbox" or "xtts" (defaults to config.tts.engine)
        **engine_kwargs: Passed to the engine constructor

    Returns:
        Engine instance (model weights load lazily on first synthesis)

    Raises:
        ValueError: If engine_type is unknown
    """
    engine_type = engine_type or config.tts.engine
    engine_cls = _ENGINES.get(engine_type)
    if engine_cls is None:
        raise ValueError(
            f"Unknown TTS engine: {engine_type!r}. Available: {sorted(_ENGINES)}"
        )
    return engine_cls(**engine_kwargs)
