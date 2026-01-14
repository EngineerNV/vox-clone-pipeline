"""Tests for audio module."""


import numpy as np
import pytest

from audio import AudioIOHandler, AudioProcessor


class TestAudioIOHandler:
    """Tests for AudioIOHandler class."""

    def test_get_audio_duration(self):
        """Test audio duration calculation."""
        # Create 1 second of audio at 22050 Hz
        audio_data = np.zeros(22050, dtype=np.float32)
        duration = AudioIOHandler.get_audio_duration(audio_data, 22050)
        assert duration == 1.0

    def test_validate_audio_duration(self):
        """Test audio duration validation."""
        # Valid duration (5 seconds)
        audio_data = np.zeros(int(22050 * 5), dtype=np.float32)
        assert AudioIOHandler.validate_audio_duration(audio_data, 22050) is True

        # Too short (0.1 seconds)
        audio_data = np.zeros(int(22050 * 0.1), dtype=np.float32)
        assert AudioIOHandler.validate_audio_duration(audio_data, 22050) is False

        # Too long (40 seconds)
        audio_data = np.zeros(int(22050 * 40), dtype=np.float32)
        assert AudioIOHandler.validate_audio_duration(audio_data, 22050) is False

    def test_save_and_load_audio(self, tmp_path):
        """Test saving and loading audio files."""
        # Create test audio
        audio_data = np.random.randn(22050).astype(np.float32) * 0.1
        file_path = tmp_path / "test.wav"

        # Save audio
        AudioIOHandler.save_audio(audio_data, file_path, 22050)
        assert file_path.exists()

        # Load audio
        loaded_data, loaded_sr = AudioIOHandler.load_audio(file_path)
        assert loaded_sr == 22050
        assert len(loaded_data) == len(audio_data)


class TestAudioProcessor:
    """Tests for AudioProcessor class."""

    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create audio with known amplitude
        audio_data = np.ones(1000, dtype=np.float32) * 0.1

        normalized = AudioProcessor.normalize_audio(audio_data)

        # Normalized audio should have higher amplitude
        assert np.abs(normalized).max() > np.abs(audio_data).max()
        # But not exceed 1.0
        assert np.abs(normalized).max() <= 1.0

    def test_normalize_silent_audio(self):
        """Test normalization of silent audio."""
        # Silent audio should remain unchanged
        audio_data = np.zeros(1000, dtype=np.float32)
        normalized = AudioProcessor.normalize_audio(audio_data)
        np.testing.assert_array_equal(audio_data, normalized)

    def test_resample_audio(self):
        """Test audio resampling."""
        # Create 1 second of audio at 44100 Hz
        audio_data = np.random.randn(44100).astype(np.float32)

        # Resample to 22050 Hz
        resampled, new_sr = AudioProcessor.resample_audio(audio_data, 44100, 22050)

        assert new_sr == 22050
        # Should have approximately half the samples
        assert len(resampled) == pytest.approx(22050, rel=0.01)

    def test_resample_same_rate(self):
        """Test resampling with same rate returns original."""
        audio_data = np.random.randn(22050).astype(np.float32)

        resampled, new_sr = AudioProcessor.resample_audio(audio_data, 22050, 22050)

        assert new_sr == 22050
        np.testing.assert_array_equal(resampled, audio_data)

    def test_trim_silence(self):
        """Test silence trimming."""
        # Create audio with silence at start and end
        silence = np.zeros(1000, dtype=np.float32)
        sound = np.ones(1000, dtype=np.float32) * 0.5
        audio_data = np.concatenate([silence, sound, silence])

        trimmed = AudioProcessor.trim_silence(audio_data, 22050)

        # Trimmed audio should be shorter
        assert len(trimmed) < len(audio_data)
        # Should be approximately the length of the sound portion
        assert len(trimmed) <= len(sound) * 1.5  # Allow some margin

    def test_clean_audio(self):
        """Test full audio cleaning pipeline."""
        processor = AudioProcessor()

        # Create noisy audio
        audio_data = np.random.randn(22050).astype(np.float32) * 0.01
        audio_data += np.sin(2 * np.pi * 440 * np.arange(22050) / 22050) * 0.1

        cleaned = processor.clean_audio(audio_data, 22050)

        # Cleaned audio should have similar length (may be trimmed)
        assert len(cleaned) > 0
        assert len(cleaned) <= len(audio_data)
