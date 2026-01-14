"""Shared utility functions for the TTS studio."""

import hashlib
import tempfile
from pathlib import Path
from typing import Optional


def get_file_hash(file_path: Path) -> str:
    """
    Generate SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex digest of the file hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def create_temp_file(suffix: Optional[str] = None, prefix: Optional[str] = None) -> Path:
    """
    Create a temporary file and return its path.

    Args:
        suffix: File suffix (e.g., '.wav')
        prefix: File prefix

    Returns:
        Path to the temporary file
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    # Close the file descriptor but keep the file
    import os

    os.close(fd)
    return Path(path)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., '1:23')
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing unsafe characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    return filename.strip()
