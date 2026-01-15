"""UI components for the Streamlit application."""

from pathlib import Path
from typing import Optional

import streamlit as st

from core.config import config
from core.utils import format_duration


class UIComponents:
    """Reusable UI components for the Streamlit app."""

    @staticmethod
    def render_header() -> None:
        """Render the application header with futuristic styling."""
        st.markdown(
            """
            <div style='text-align: center; padding: 20px 0 30px 0;'>
                <div style='display: flex; align-items: center; justify-content: center; gap: 15px;'>
                    <span style='
                        font-size: 48px;
                        filter: drop-shadow(0 0 10px rgba(0, 255, 242, 0.4));
                    '>🎙️</span>
                    <div style='
                        font-family: Orbitron, sans-serif;
                        font-size: 48px;
                        font-weight: 900;
                        background: linear-gradient(135deg, #00fff2 0%, #00b4d8 50%, #a886d6 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        text-shadow: none;
                        letter-spacing: 3px;
                    '>VOX CLONE PIPELINE</div>
                </div>
                <div style='
                    font-family: Orbitron, sans-serif;
                    font-size: 12px;
                    color: #a886d6;
                    letter-spacing: 4px;
                    text-shadow: 0 0 10px rgba(168, 134, 214, 0.5);
                    margin-bottom: 15px;
                '>
                    by NICK VAUGHN
                </div>
                <div style='
                    font-family: Exo 2, sans-serif;
                    font-size: 16px;
                    color: #b8c5d6;
                    max-width: 600px;
                    margin: 0 auto;
                    line-height: 1.6;
                '>
                    Generate natural-sounding speech with voice cloning using local TTS models.<br>
                    Upload a reference audio sample and enter your text to get started.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

    @staticmethod
    def render_ditto_loading(message: str = "Generating voice clone...") -> None:
        """Render a Ditto-themed loading animation."""
        st.markdown(
            f"""
            <div class="ditto-container">
                <div class="ditto"></div>
                <div class="loading-text">◈ {message} ◈</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_audio_uploader(
        label: str = "Upload Reference Audio",
        key: str = "audio_upload",
    ) -> Optional[Path]:
        """
        Render an audio file uploader.

        Args:
            label: Label for the uploader
            key: Unique key for the widget

        Returns:
            Path to uploaded audio file or None
        """
        uploaded_file = st.file_uploader(
            label,
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key=key,
            help=f"Max file size: {config.app.max_file_size_mb}MB. "
            f"Duration: {config.audio.min_duration_seconds}-{config.audio.max_duration_seconds}s",
        )

        if uploaded_file is not None:
            # Save uploaded file to temp directory
            temp_path = config.app.temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return temp_path

        return None

    @staticmethod
    def render_audio_player(
        audio_path: Path,
        label: str = "Audio",
    ) -> None:
        """
        Render an audio player.

        Args:
            audio_path: Path to audio file
            label: Label for the audio player
        """
        if audio_path.exists():
            st.audio(str(audio_path), format=f"audio/{audio_path.suffix[1:]}")
            st.caption(label)

    @staticmethod
    def render_text_input(
        label: str = "Enter Text to Synthesize",
        key: str = "text_input",
        default: str = "",
    ) -> str:
        """
        Render a text input area.

        Args:
            label: Label for the input
            key: Unique key for the widget
            default: Default text value

        Returns:
            User input text
        """
        return st.text_area(
            label,
            value=default,
            height=150,
            key=key,
            help="Enter the text you want to convert to speech",
        )

    @staticmethod
    def render_tts_settings() -> dict[str, any]:
        """
        Render TTS settings controls.

        Returns:
            Dictionary of TTS settings
        """
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                language = st.selectbox(
                    "Language",
                    options=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"],
                    index=0,
                    help="Select the language for speech synthesis",
                )

                temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.0,
                    value=config.tts.temperature,
                    step=0.05,
                    help="Controls randomness in generation. Lower = more consistent, Higher = more varied",
                )

            with col2:
                speed = st.slider(
                    "Speed",
                    min_value=0.5,
                    max_value=2.0,
                    value=config.tts.speed,
                    step=0.1,
                    help="Speech speed multiplier",
                )

        return {
            "language": language,
            "temperature": temperature,
            "speed": speed,
        }

    @staticmethod
    def render_audio_cleaning_settings() -> dict[str, bool]:
        """
        Render audio cleaning settings controls.

        Returns:
            Dictionary of cleaning settings
        """
        with st.expander("🧹 Audio Cleaning Options", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                reduce_noise = st.checkbox("Reduce Noise", value=True)

            with col2:
                normalize = st.checkbox("Normalize", value=True)

            with col3:
                trim_silence = st.checkbox("Trim Silence", value=True)

        return {
            "reduce_noise": reduce_noise,
            "normalize": normalize,
            "trim_silence": trim_silence,
        }

    @staticmethod
    def render_audio_info(info: dict[str, any]) -> None:
        """
        Render audio information.

        Args:
            info: Dictionary with audio information
        """
        st.markdown("**Audio Information:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Duration", format_duration(info["duration"]))

        with col2:
            st.metric("Sample Rate", f"{info['sample_rate']} Hz")

        with col3:
            status = "✅ Valid" if info["is_valid_duration"] else "❌ Invalid"
            st.metric("Status", status)

    @staticmethod
    def render_download_button(
        file_path: Path,
        label: str = "Download Audio",
        key: str = "download",
    ) -> None:
        """
        Render a download button for an audio file.

        Args:
            file_path: Path to the file to download
            label: Button label
            key: Unique key for the widget
        """
        if file_path.exists():
            with open(file_path, "rb") as f:
                st.download_button(
                    label=label,
                    data=f,
                    file_name=file_path.name,
                    mime=f"audio/{file_path.suffix[1:]}",
                    key=key,
                )
