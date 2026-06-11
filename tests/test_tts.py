"""Tests for the TTS engine abstraction (no model downloads)."""

import numpy as np
import pytest
import torch

from tts.engine import (
    ChatterboxEngine,
    XTTSEngine,
    create_engine,
    resolve_device,
)


class FakeChatterboxModel:
    """Stands in for ChatterboxTurboTTS/ChatterboxTTS."""

    sr = 24000

    def __init__(self):
        self.last_kwargs = None

    def generate(self, text, audio_prompt_path=None, **kwargs):
        self.last_kwargs = {"text": text, "audio_prompt_path": audio_prompt_path, **kwargs}
        return torch.zeros(1, self.sr)  # 1 second of silence


@pytest.fixture
def reference_wav(tmp_path):
    path = tmp_path / "ref.wav"
    path.write_bytes(b"fake")
    return path


class TestEngineFactory:
    def test_creates_chatterbox_by_default(self):
        engine = create_engine("chatterbox")
        assert isinstance(engine, ChatterboxEngine)
        assert engine.variant == "turbo"

    def test_creates_xtts(self):
        engine = create_engine("xtts")
        assert isinstance(engine, XTTSEngine)

    def test_rejects_unknown_engine(self):
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            create_engine("espeak")

    def test_rejects_unknown_chatterbox_variant(self):
        with pytest.raises(ValueError, match="variant"):
            ChatterboxEngine(variant="mega")


class TestResolveDevice:
    def test_explicit_preference_wins(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("mps") == "mps"

    def test_auto_returns_available_device(self):
        assert resolve_device("auto") in ("cuda", "mps", "cpu")


class TestChatterboxEngine:
    def test_synthesize_returns_audio_and_native_sample_rate(self, reference_wav):
        engine = ChatterboxEngine(variant="turbo", device="cpu")
        engine._model = FakeChatterboxModel()

        audio, sample_rate = engine.synthesize("Hello world", speaker_wav=reference_wav)

        assert sample_rate == 24000
        assert audio.dtype == np.float32
        assert audio.ndim == 1

    def test_turbo_omits_unsupported_params(self, reference_wav):
        engine = ChatterboxEngine(variant="turbo", device="cpu")
        engine._model = FakeChatterboxModel()

        engine.synthesize("Hi", speaker_wav=reference_wav, exaggeration=0.9, cfg_weight=0.3)

        assert "exaggeration" not in engine._model.last_kwargs
        assert "cfg_weight" not in engine._model.last_kwargs

    def test_standard_passes_expressiveness_params(self, reference_wav):
        engine = ChatterboxEngine(variant="standard", device="cpu")
        engine._model = FakeChatterboxModel()

        engine.synthesize("Hi", speaker_wav=reference_wav, exaggeration=0.9, cfg_weight=0.3)

        assert engine._model.last_kwargs["exaggeration"] == 0.9
        assert engine._model.last_kwargs["cfg_weight"] == 0.3

    def test_rejects_non_english_language(self, reference_wav):
        engine = ChatterboxEngine(variant="turbo", device="cpu")
        engine._model = FakeChatterboxModel()

        with pytest.raises(ValueError, match="only supports English"):
            engine.synthesize("Bonjour", speaker_wav=reference_wav, language="fr")

    def test_speed_changes_duration(self, reference_wav):
        engine = ChatterboxEngine(variant="turbo", device="cpu")
        engine._model = FakeChatterboxModel()

        audio_normal, _ = engine.synthesize("Hi", speaker_wav=reference_wav, speed=1.0)
        audio_fast, _ = engine.synthesize("Hi", speaker_wav=reference_wav, speed=2.0)

        assert len(audio_fast) < len(audio_normal)

    def test_requires_speaker_wav(self):
        engine = ChatterboxEngine(variant="turbo", device="cpu")
        with pytest.raises(ValueError, match="reference audio is required"):
            engine.synthesize("Hello")

    def test_rejects_empty_text(self, reference_wav):
        engine = ChatterboxEngine(variant="turbo", device="cpu")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            engine.synthesize("   ", speaker_wav=reference_wav)

    def test_reference_passed_at_original_rate(self):
        assert ChatterboxEngine(device="cpu").reference_sample_rate is None

    def test_supports_voice_cloning(self):
        assert ChatterboxEngine(device="cpu").supports_voice_cloning() is True

    def test_english_only(self):
        assert ChatterboxEngine(device="cpu").get_available_languages() == ["en"]

    def test_missing_package_raises_helpful_error(self, reference_wav, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def block_chatterbox(name, *args, **kwargs):
            if name.startswith("chatterbox"):
                raise ImportError("No module named 'chatterbox'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_chatterbox)
        engine = ChatterboxEngine(variant="turbo", device="cpu")

        with pytest.raises(RuntimeError, match="requirements.txt"):
            engine.synthesize("Hello", speaker_wav=reference_wav)


class TestXTTSEngine:
    def test_supports_voice_cloning_for_xtts_models(self):
        engine = XTTSEngine(device="cpu")
        assert engine.supports_voice_cloning() is True

    def test_reference_resampled_to_processing_rate(self):
        from core.config import config

        assert XTTSEngine(device="cpu").reference_sample_rate == config.audio.sample_rate

    def test_never_selects_mps(self):
        # Coqui XTTS is unreliable on MPS; it must stay on CPU unless CUDA exists
        engine = XTTSEngine(device="mps")
        assert engine._device == "cpu"


class TestOrchestratorSampleRate:
    def test_output_saved_at_engine_sample_rate(self, reference_wav, tmp_path, monkeypatch):
        from orchestration.tts_orchestrator import TTSOrchestrator

        orchestrator = TTSOrchestrator()

        engine_sr = 24000
        monkeypatch.setattr(
            orchestrator.voice_cloner,
            "clone_voice",
            lambda **kwargs: (np.zeros(engine_sr, dtype=np.float32), engine_sr),
        )

        saved = {}
        monkeypatch.setattr(
            orchestrator.audio_io,
            "save_audio",
            lambda audio, path, sample_rate=None: saved.update(sample_rate=sample_rate),
        )

        orchestrator.process_and_clone(
            text="Hello",
            reference_audio_path=reference_wav,
            output_path=tmp_path / "out.wav",
        )

        assert saved["sample_rate"] == engine_sr
