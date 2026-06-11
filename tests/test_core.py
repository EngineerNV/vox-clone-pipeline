"""Tests for core module."""



from core.config import AppConfig, AudioConfig, Config, TTSConfig
from core.utils import format_duration, get_file_hash, sanitize_filename


class TestConfig:
    """Tests for configuration classes."""

    def test_audio_config_defaults(self):
        """Test AudioConfig default values."""
        config = AudioConfig()
        assert config.sample_rate == 22050
        assert config.max_duration_seconds == 30
        assert config.min_duration_seconds == 0.5
        assert ".wav" in config.supported_formats

    def test_tts_config_defaults(self):
        """Test TTSConfig default values."""
        config = TTSConfig()
        assert config.language == "en"
        assert config.engine == "chatterbox"
        assert config.chatterbox_variant == "turbo"
        assert config.device == "auto"
        assert 0 < config.temperature <= 1.0

    def test_app_config_creates_directories(self, tmp_path):
        """Test AppConfig creates output directories."""
        config = AppConfig()
        config.output_dir = tmp_path / "outputs"
        config.temp_dir = tmp_path / "temp"
        config.__post_init__()

        assert config.output_dir.exists()
        assert config.temp_dir.exists()

    def test_config_load(self):
        """Test Config loading."""
        config = Config.load()
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.tts, TTSConfig)
        assert isinstance(config.app, AppConfig)


class TestUtils:
    """Tests for utility functions."""

    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(0) == "0:00"
        assert format_duration(65) == "1:05"
        assert format_duration(125.7) == "2:05"

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert sanitize_filename("test.wav") == "test.wav"
        assert sanitize_filename("test/file.wav") == "test_file.wav"
        assert sanitize_filename("test:file*.wav") == "test_file_.wav"
        assert sanitize_filename("  spaced  ") == "spaced"

    def test_get_file_hash(self, tmp_path):
        """Test file hashing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        hash1 = get_file_hash(test_file)
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length

        # Same content should produce same hash
        hash2 = get_file_hash(test_file)
        assert hash1 == hash2
