"""
Vox Clone Pipeline - Main Application Entry Point

A clean, modular Streamlit app for local zero-shot TTS voice cloning.
Runs on Apple Silicon GPU (MPS), CUDA, or CPU.
"""

import streamlit as st

from app import MainPage
from core.config import config

DARK_THEME_CSS = """
<style>
    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Exo+2:wght@300;400;500;600&display=swap');
    
    /* Main background - dark gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d15 0%, #1a1a2e 100%);
        border-right: 1px solid #00fff2;
    }
    
    /* Headers with glow effect */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #ffffff !important; /* White core for better pop */
        text-shadow: 
            0 0 5px #00fff2,
            0 0 15px rgba(0, 255, 242, 0.8),
            0 0 30px rgba(0, 255, 242, 0.4);
        letter-spacing: 1px !important;
        text-transform: uppercase;
    }
    
    /* Body text - but exclude expander internals */
    p, label, .stMarkdown {
        font-family: 'Exo 2', sans-serif !important;
        color: #b8c5d6 !important;
    }
    
    /* Only style spans that are NOT inside expanders or file uploaders */
    span:not([data-testid="stExpander"] *):not([data-testid*="stFileUploader"] *) {
        font-family: 'Exo 2', sans-serif !important;
        color: #FFFFFF !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 5px;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid rgba(0, 255, 242, 0.4);
        border-radius: 8px;
        color: #b8c5d6 !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 242, 0.15) !important;
        border: 1px solid #00fff2 !important;
        color: #00fff2 !important;
    }
    
    /* Primary buttons with neon effect */
    .stButton > button[kind="primary"], .stButton > button {
        background: linear-gradient(135deg, #00fff2 0%, #00b4d8 100%) !important;
        color: #000000 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(0, 255, 242, 0.4), inset 0 0 20px rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease !important;
        text-shadow: none !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(0, 255, 242, 0.7), inset 0 0 30px rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px);
    }
    
    .stButton > button p, .stButton > button span {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ff006e 0%, #8338ec 100%) !important;
        color: white !important;
        font-family: 'Orbitron', sans-serif !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(255, 0, 110, 0.4);
    }
    
    /* Text inputs and areas - Updated selectors for reliability */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background-color: rgba(0, 0, 0, 0.9) !important; /* Darker background */
        border: 1px solid #00fff2 !important;
        border-radius: 10px !important;
        color: #00fff2 !important; /* Neon text */
        font-family: 'Exo 2', sans-serif !important;
        caret-color: #00fff2 !important;
    }
    
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        box-shadow: 0 0 15px rgba(0, 255, 242, 0.3) !important;
        border-color: #00fff2 !important;
    }
    
    /* Placeholder styling */
    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: rgba(0, 255, 242, 0.5) !important;
    }
    
    /* Selectbox styling to match inputs */
    [data-testid="stSelectbox"] > div > div {
        background-color: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid #00fff2 !important;
        border-radius: 10px !important;
        color: #e0e0e0 !important;
        font-family: 'Exo 2', sans-serif !important;
    }
    
    /* Remove default internal borders in selectbox */
    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background-color: transparent !important;
        border: none !important;
        color: #e0e0e0 !important; 
    }
    
    /* Icons in selectbox */
    [data-testid="stSelectbox"] svg {
        fill: #00fff2 !important;
    }
    
    /* Selected value text */
    [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {
        color: #00fff2 !important;
        font-family: 'Exo 2', sans-serif !important;
    }

    /* Dropdown menu styling */
    [data-baseweb="popover"], [data-baseweb="menu"] {
        background-color: #1a1a2e !important;
        border: 1px solid #00fff2 !important;
    }
    
    [role="option"] {
        background-color: #1a1a2e !important;
        color: #b8c5d6 !important;
    }
    
    [role="option"]:hover, [role="option"][aria-selected="true"] {
        background-color: rgba(0, 255, 242, 0.2) !important;
        color: #00fff2 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 2px dashed #00fff2 !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #ff006e !important;
        box-shadow: 0 0 20px rgba(255, 0, 110, 0.3);
    }
    
    /* Make file uploader text visible */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p {
        color: #b8c5d6 !important;
        font-family: 'Exo 2', sans-serif !important;
    }
    
    /* Direct styling for the uploaded file item container and text - covering multiple Streamlit versions */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderUploadedFile"] {
        background-color: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid #00fff2 !important;
        color: #00fff2 !important;
    }
    
    /* Target the filename text specifically */
    [data-testid="stFileUploaderFile"] div,
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderUploadedFile"] div,
    [data-testid="stFileUploaderUploadedFile"] span {
        color: #00fff2 !important;
        font-family: 'Exo 2', sans-serif !important;
    }

    [data-testid="stFileUploader"] small {
        color: #ffffff !important; /* White for file size */
        font-family: 'Exo 2', sans-serif !important;
        opacity: 0.8;
    }
    
    /* Delete (X) button styling */
    [data-testid="stFileUploader"] button {
        border: none !important;
        background: transparent !important;
    }

    /* Force the X icon to be cyan */
    [data-testid="stFileUploader"] button svg {
        fill: #00fff2 !important;
        color: #00fff2 !important;
    }

    /* Turn red on hover for danger indication */
    [data-testid="stFileUploader"] button:hover {
        border-color: transparent !important;
        background-color: transparent !important;
        color: #ff006e !important;
    }

    [data-testid="stFileUploader"] button:hover svg {
        fill: #ff006e !important;
        color: #ff006e !important;
    }
    
    /* Expanders - comprehensive fix for arrow text leak */
    .streamlit-expanderHeader {
        font-family: 'Orbitron', sans-serif !important;
    }
    
    /* Target the expander summary and hide SVG text content */
    [data-testid="stExpander"] details summary {
        color: #00fff2 !important;
    }
    
    /* Aggressively hide any element containing arrow class names */
    [data-testid="stExpander"] details summary div[class*="arrow"],
    [data-testid="stExpander"] details summary span[class*="arrow"],
    [data-testid="stExpander"] details summary [class*="StyledIcon"] {
        display: none !important;
        visibility: hidden !important;
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
    }
    
    /* Hide SVG siblings that might contain text */
    [data-testid="stExpander"] summary svg ~ *:not(p):not(span:last-child) {
        display: none !important;
    }
    
    /* Override any text content in expander that isn't the label */
    [data-testid="stExpander"] summary > div:first-child {
        display: none !important;
    }
    
    /* Keep only the text label visible */
    [data-testid="stExpander"] summary p {
        color: #00fff2 !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #b8c5d6 !important;
        font-family: 'Exo 2', sans-serif !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background: #00fff2 !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid #00fff2;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0, 255, 242, 0.2);
    }
    
    [data-testid="stMetricValue"] {
        color: #00fff2 !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(0, 255, 242, 0.3) !important;
        box-shadow: 0 0 10px rgba(0, 255, 242, 0.2);
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: rgba(0, 0, 0, 0.4) !important;
        border-radius: 10px !important;
    }
    
    /* Audio player */
    audio {
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 255, 242, 0.3);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid #00fff2 !important;
        border-radius: 10px !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        color: #00fff2 !important; /* Bright cyan for selected language */
    }
    
    /* Checkbox */
    .stCheckbox > label > span {
        color: #b8c5d6 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00fff2, #00b4d8);
        border-radius: 4px;
    }
    
    /* Ditto Loading Animation */
    @keyframes dittoMorph {
        0%, 100% { 
            border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
            transform: rotate(0deg) scale(1);
        }
        25% { 
            border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%;
            transform: rotate(90deg) scale(1.1);
        }
        50% { 
            border-radius: 50% 60% 30% 60% / 30% 60% 70% 40%;
            transform: rotate(180deg) scale(1);
        }
        75% { 
            border-radius: 60% 40% 60% 30% / 70% 30% 50% 60%;
            transform: rotate(270deg) scale(1.1);
        }
    }
    
    @keyframes dittoFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    
    @keyframes dittoGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(168, 134, 214, 0.6), 0 0 40px rgba(168, 134, 214, 0.4); }
        50% { box-shadow: 0 0 30px rgba(168, 134, 214, 0.8), 0 0 60px rgba(168, 134, 214, 0.5); }
    }
    
    .ditto-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 30px;
    }
    
    .ditto {
        width: 100px;
        height: 100px;
        background: linear-gradient(135deg, #d4a5e8 0%, #a886d6 50%, #9370db 100%);
        animation: dittoMorph 3s ease-in-out infinite, dittoFloat 2s ease-in-out infinite, dittoGlow 2s ease-in-out infinite;
        position: relative;
    }
    
    .ditto::before {
        content: '';
        position: absolute;
        top: 30%;
        left: 20%;
        width: 15px;
        height: 8px;
        background: #2a2a3a;
        border-radius: 50%;
        box-shadow: 35px 0 0 #2a2a3a;
    }
    
    .ditto::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 35%;
        width: 25px;
        height: 8px;
        background: #2a2a3a;
        border-radius: 0 0 50% 50%;
    }
    
    .loading-text {
        margin-top: 20px;
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
        color: #a886d6;
        text-shadow: 0 0 10px rgba(168, 134, 214, 0.5);
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    /* Glowing border animation for cards */
    .cyber-card {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid #00fff2;
        border-radius: 15px;
        padding: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .cyber-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0, 255, 242, 0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
</style>
"""

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
    
    # Inject dark theme CSS
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

    # Initialize and render main page
    page = MainPage()
    page.render()

    # Add footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; padding: 30px;'>
            <div style='
                font-family: Orbitron, sans-serif;
                font-size: 12px;
                color: #00fff2;
                text-shadow: 0 0 10px rgba(0, 255, 242, 0.5);
                letter-spacing: 2px;
            '>
                VOX CLONE PIPELINE v0.1.0
            </div>
            <div style='
                margin-top: 8px;
                font-family: Exo 2, sans-serif;
                font-size: 11px;
                color: #666;
                letter-spacing: 1px;
            '>
                Powered by Chatterbox TTS | Apple Silicon GPU + CPU
            </div>
            <div style='
                margin-top: 15px;
                font-family: Orbitron, sans-serif;
                font-size: 10px;
                color: #a886d6;
                text-shadow: 0 0 8px rgba(168, 134, 214, 0.4);
                letter-spacing: 3px;
            '>
                ◈ CRAFTED WITH 💜 BY NICK VAUGHN ◈
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
