"""Main page for the Streamlit application."""


import streamlit as st

from app.ui_components import UIComponents
from orchestration import TTSOrchestrator


@st.cache_resource
def _get_orchestrator() -> TTSOrchestrator:
    """Build the orchestrator once per process so the loaded TTS model
    survives Streamlit reruns instead of being reloaded on every interaction."""
    return TTSOrchestrator()


class MainPage:
    """Main page of the TTS Studio application."""

    def __init__(self) -> None:
        """Initialize the main page."""
        self.ui = UIComponents()
        self.orchestrator = _get_orchestrator()

    def render(self) -> None:
        """Render the main page."""
        # Header
        self.ui.render_header()

        # Create tabs
        tab1, tab2 = st.tabs(["🎙️ Voice Cloning", "🧹 Audio Cleaning"])

        with tab1:
            self._render_voice_cloning_tab()

        with tab2:
            self._render_audio_cleaning_tab()

    def _render_voice_cloning_tab(self) -> None:
        """Render the voice cloning tab."""
        st.header("Voice Cloning & TTS")

        # File upload section
        st.subheader("1. Upload Reference Audio")
        reference_audio = self.ui.render_audio_uploader(
            label="Upload a voice sample for cloning (5-30 seconds recommended)",
            key="reference_audio",
        )

        if reference_audio:
            # Display reference audio info
            try:
                audio_info = self.orchestrator.get_audio_info(reference_audio)
                self.ui.render_audio_info(audio_info)

                # Play reference audio
                st.markdown("**Reference Audio:**")
                self.ui.render_audio_player(reference_audio, "Uploaded reference")

                if not audio_info["is_valid_duration"]:
                    st.warning(
                        "⚠️ Audio duration is outside the recommended range. "
                        "This may affect cloning quality."
                    )

            except Exception as e:
                st.error(f"Error loading audio: {str(e)}")
                return

            st.divider()

            # Text input section
            st.subheader("2. Enter Text to Synthesize")
            text = self.ui.render_text_input(
                default="Hello! This is a test of voice cloning technology.",
            )

            # Settings
            st.subheader("3. Configure Settings")
            settings = self.ui.render_tts_settings()

            st.divider()

            # Generate button
            st.subheader("4. Generate Speech")
            if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
                if not text.strip():
                    st.error("Please enter some text to synthesize.")
                    return

                # Show Ditto loading animation
                loading_placeholder = st.empty()
                loading_placeholder.markdown(
                    """
                    <div class="ditto-container">
                        <div class="ditto"></div>
                        <div class="loading-text">◈ Transforming your voice... ◈</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                try:
                    # Generate speech
                    output_path = self.orchestrator.process_and_clone(
                        text=text,
                        reference_audio_path=reference_audio,
                        clean_reference=False,
                        **settings,
                    )
                    
                    # Clear loading animation
                    loading_placeholder.empty()

                    # Display success message
                    st.success("✅ Speech generated successfully!")

                    # Play generated audio
                    st.markdown("**Generated Audio:**")
                    self.ui.render_audio_player(output_path, "Generated speech")

                    # Download button
                    self.ui.render_download_button(
                        output_path,
                        label="📥 Download Generated Audio",
                        key="download_generated",
                    )

                except Exception as e:
                    loading_placeholder.empty()
                    st.error(f"Error generating speech: {str(e)}")
                    st.exception(e)

        else:
            st.info("👆 Please upload a reference audio file to get started.")

    def _render_audio_cleaning_tab(self) -> None:
        """Render the audio cleaning tab."""
        st.header("Audio Cleaning")

        # File upload section
        st.subheader("1. Upload Audio to Clean")
        audio_to_clean = self.ui.render_audio_uploader(
            label="Upload an audio file to clean",
            key="audio_to_clean",
        )

        if audio_to_clean:
            # Display original audio info
            try:
                audio_info = self.orchestrator.get_audio_info(audio_to_clean)

                st.markdown("**Original Audio:**")
                self.ui.render_audio_player(audio_to_clean, "Original audio")
                self.ui.render_audio_info(audio_info)

            except Exception as e:
                st.error(f"Error loading audio: {str(e)}")
                return

            st.divider()

            # Cleaning settings
            st.subheader("2. Select Cleaning Operations")
            cleaning_settings = self.ui.render_audio_cleaning_settings()

            st.divider()

            # Clean button
            st.subheader("3. Clean Audio")
            if st.button("🧹 Clean Audio", type="primary", use_container_width=True):
                # Show Ditto loading animation
                loading_placeholder = st.empty()
                loading_placeholder.markdown(
                    """
                    <div class="ditto-container">
                        <div class="ditto"></div>
                        <div class="loading-text">◈ Cleaning your audio... ◈</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                try:
                    # Clean audio
                    cleaned_path = self.orchestrator.clean_audio_file(
                        input_path=audio_to_clean,
                        reduce_noise=cleaning_settings["reduce_noise"],
                        normalize=cleaning_settings["normalize"],
                        trim_silence=cleaning_settings["trim_silence"],
                    )
                    
                    # Clear loading animation
                    loading_placeholder.empty()

                    # Display success message
                    st.success("✅ Audio cleaned successfully!")

                    # Play cleaned audio
                    st.markdown("**Cleaned Audio:**")
                    self.ui.render_audio_player(cleaned_path, "Cleaned audio")

                    # Download button
                    self.ui.render_download_button(
                        cleaned_path,
                        label="📥 Download Cleaned Audio",
                        key="download_cleaned",
                    )

                except Exception as e:
                    loading_placeholder.empty()
                    st.error(f"Error cleaning audio: {str(e)}")
                    st.exception(e)

        else:
            st.info("👆 Please upload an audio file to clean.")
