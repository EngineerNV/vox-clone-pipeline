"""
Local TTS Studio - Main Application Entry Point

A clean, modular Streamlit app for local zero-shot TTS voice cloning.
Optimized for macOS CPU-only execution.
"""

import streamlit as st

from app import MainPage
from core.config import config


def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=config.app.title,
        page_icon=config.app.page_icon,
        layout=config.app.layout,
        initial_sidebar_state="collapsed",
    )


def main() -> None:
    """Main application entry point."""
    # Setup page configuration
    setup_page_config()

    # Initialize and render main page
    page = MainPage()
    page.render()

    # Add footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <small>
                Local TTS Studio v0.1.0 |
                Powered by Coqui TTS |
                CPU-only for macOS
            </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
