"""Audio processing and cleaning operations."""

from typing import Optional

import noisereduce as nr
import numpy as np
from scipy import signal

from core.config import config


class AudioProcessor:
    """Handles audio processing operations including noise reduction and normalization."""

    @staticmethod
    def reduce_noise(
        audio_data: np.ndarray,
        sample_rate: int,
        stationary: Optional[bool] = None,
        prop_decrease: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply noise reduction to audio data.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate
            stationary: Whether to use stationary noise reduction
            prop_decrease: Proportion of noise to reduce (0-1)

        Returns:
            Noise-reduced audio data
        """
        if stationary is None:
            stationary = config.audio.noise_reduce_stationary
        if prop_decrease is None:
            prop_decrease = config.audio.noise_reduce_prop_decrease

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=prop_decrease,
        )

        return reduced_noise

    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio to a target level in dB.

        Args:
            audio_data: Audio data as numpy array
            target_level: Target level in dB

        Returns:
            Normalized audio data
        """
        # Calculate current RMS level
        rms = np.sqrt(np.mean(audio_data**2))

        if rms == 0:
            return audio_data

        # Calculate current level in dB
        current_db = 20 * np.log10(rms)

        # Calculate gain needed
        gain_db = target_level - current_db
        gain = 10 ** (gain_db / 20)

        # Apply gain
        normalized = audio_data * gain

        # Prevent clipping
        max_val = np.abs(normalized).max()
        if max_val > 1.0:
            normalized = normalized / max_val * 0.99

        return normalized

    @staticmethod
    def resample_audio(
        audio_data: np.ndarray,
        original_sr: int,
        target_sr: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Resample audio to a target sample rate.

        Args:
            audio_data: Audio data as numpy array
            original_sr: Original sample rate
            target_sr: Target sample rate (uses config default if not provided)

        Returns:
            Tuple of (resampled_audio, target_sample_rate)
        """
        if target_sr is None:
            target_sr = config.audio.sample_rate

        if original_sr == target_sr:
            return audio_data, target_sr

        # Calculate resampling ratio
        num_samples = int(len(audio_data) * target_sr / original_sr)

        # Resample using scipy
        resampled = signal.resample(audio_data, num_samples)

        return resampled, target_sr

    @staticmethod
    def trim_silence(
        audio_data: np.ndarray,
        sample_rate: int,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.1,
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate
            threshold_db: Silence threshold in dB
            min_silence_duration: Minimum silence duration to trim (seconds)

        Returns:
            Trimmed audio data
        """
        # Convert threshold to linear scale
        threshold = 10 ** (threshold_db / 20)

        # Calculate frame size
        frame_size = int(min_silence_duration * sample_rate)

        # Find first non-silent frame
        start_idx = 0
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i : i + frame_size]
            if np.abs(frame).max() > threshold:
                start_idx = i
                break

        # Find last non-silent frame
        end_idx = len(audio_data)
        for i in range(len(audio_data) - frame_size, 0, -frame_size):
            frame = audio_data[i : i + frame_size]
            if np.abs(frame).max() > threshold:
                end_idx = i + frame_size
                break

        return audio_data[start_idx:end_idx]

    def clean_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        reduce_noise: bool = True,
        normalize: bool = True,
        trim_silence: bool = True,
    ) -> np.ndarray:
        """
        Apply a full cleaning pipeline to audio data.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate
            reduce_noise: Whether to apply noise reduction
            normalize: Whether to normalize audio
            trim_silence: Whether to trim silence

        Returns:
            Cleaned audio data
        """
        cleaned = audio_data.copy()

        if trim_silence:
            cleaned = self.trim_silence(cleaned, sample_rate)

        if reduce_noise:
            cleaned = self.reduce_noise(cleaned, sample_rate)

        if normalize:
            cleaned = self.normalize_audio(cleaned)

        return cleaned
