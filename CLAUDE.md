# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A Streamlit application for local, zero-shot TTS voice cloning. Upload a short
reference audio clip, enter text, and generate speech in the cloned voice.
Runs on CUDA, Apple Silicon (MPS), or CPU.

## Commands

```bash
# First-time setup: creates a venv, installs deps, and pre-downloads
# Chatterbox model weights (avoids a multi-GB stall on first run)
./setup.sh        # macOS/Linux
.\setup.ps1       # Windows PowerShell

# Run the app
streamlit run streamlit_app.py

# Run all tests
pytest

# Run a single test file / test
pytest tests/test_tts.py
pytest tests/test_tts.py::TestChatterboxEngine::test_speed_changes_duration

# Run with coverage
pytest --cov=. --cov-report=html

# Lint and format
ruff check .
ruff format .

# Type check
mypy .
```

`tests/test_tts.py` swaps in a `FakeChatterboxModel` so the test suite never
downloads model weights — keep that pattern when adding engine tests.

## Architecture

Layered, dependency-injected modules under `core` → `audio`/`tts` →
`orchestration` → `app`:

```
streamlit_app.py
  └─ app.MainPage (app/pages.py)
       ├─ app.UIComponents (app/ui_components.py)  — pure Streamlit widgets
       └─ orchestration.TTSOrchestrator (orchestration/tts_orchestrator.py)
            ├─ tts.VoiceCloner (tts/voice_cloner.py)
            │    └─ tts.BaseTTSEngine (tts/engine.py)
            ├─ audio.AudioIOHandler (audio/io_handler.py)
            └─ audio.AudioProcessor (audio/processor.py)
```

- **`core/config.py`**: single global `config` instance (`AudioConfig`,
  `TTSConfig`, `AppConfig`), loaded once at import time via `Config.load()`
  with environment-variable overrides (`TTS_ENGINE`, `CHATTERBOX_VARIANT`,
  `TTS_DEVICE`, `TTS_MODEL_NAME`, `TTS_LANGUAGE`, `AUDIO_SAMPLE_RATE`). Most
  modules import `config` directly rather than receiving it via DI.
- **`core/logging_config.py`**: `setup_logging()` attaches a console handler
  plus a rotating file handler (`logs/app.log`, 5MB x3 backups) to the root
  logger; called once from `streamlit_app.py`. Safe to call repeatedly (no-op
  if handlers already attached, e.g. on Streamlit reruns). Engines, the
  orchestrator, `AudioIOHandler`, and `MainPage` log lifecycle events (model
  load, synthesis params/duration, file I/O, exceptions) for debugging.
- **`tts/engine.py`**: `BaseTTSEngine` ABC with two implementations:
  - `ChatterboxEngine` (default) — Resemble AI Chatterbox, English only,
    MIT-licensed. `turbo` variant (default) requires reference clips >5s
    (`min_reference_seconds`); `standard` variant additionally supports
    `exaggeration`/`cfg_weight` via `supports_expressiveness`. Model weights
    lazy-load on first `.model` access via `from_pretrained`.
  - `XTTSEngine` (legacy) — Coqui XTTS v2, multilingual, but its dependency
    stack (transformers 4.x) conflicts with chatterbox-tts (transformers 5.x)
    and **must run in a separate virtualenv** using `requirements-xtts.txt`.
    Forced to CPU/CUDA only (MPS is unreliable for this engine).
  - `create_engine()` is the factory; `_ENGINES` maps `"chatterbox"`/`"xtts"`
    to classes. `resolve_device("auto")` picks `cuda` > `mps` > `cpu`.
  - Engines declare capabilities via properties (`supports_expressiveness`,
    `reference_sample_rate`, `min_reference_seconds`,
    `supports_voice_cloning()`) so `VoiceCloner` and the UI can branch on
    engine type without isinstance checks.
- **`tts/voice_cloner.py`**: `VoiceCloner.prepare_reference_audio()` loads,
  validates duration, optionally resamples to `engine.reference_sample_rate`,
  and optionally cleans the reference clip before handing it to the engine.
  `clone_voice()` ties preparation + `engine.synthesize()` together.
- **`audio/io_handler.py`**: WAV/FLAC via `soundfile`; MP3/OGG/M4A via `pydub`,
  with `ffmpeg`/`ffprobe` bundled via `static-ffmpeg` (added to `PATH` at
  import time — no system ffmpeg install needed). Always converts to mono.
  Logs every load/save (path, duration, sample rate).
- **`audio/processor.py`**: noise reduction (`noisereduce`), RMS-based
  normalization, silence trimming, resampling (`scipy.signal.resample`), and
  speed change via `librosa.effects.time_stretch` (pitch-preserving).
- **`orchestration/tts_orchestrator.py`**: top-level facade used by the UI —
  `process_and_clone()` (full clone pipeline, saves output at the *engine's*
  native sample rate — a mismatch here pitch-shifts the result) and
  `clean_audio_file()` (standalone cleaning pipeline).
- **`app/pages.py`**: `MainPage` is built once per Streamlit process via
  `@st.cache_resource` (`_get_orchestrator()`) so the loaded TTS model
  survives reruns. Two tabs: Voice Cloning and Audio Cleaning.
- **`app/ui_components.py`**: stateless `UIComponents` static methods.
  `render_tts_settings(engine)` reads the engine's capability properties to
  decide which controls (language selector, exaggeration/cfg_weight sliders)
  to show.

## Configuration

Key env vars (see `core/config.py` for the full set and defaults):

| Var | Values | Default |
|---|---|---|
| `TTS_ENGINE` | `chatterbox`, `xtts` | `chatterbox` |
| `CHATTERBOX_VARIANT` | `turbo`, `standard` | `turbo` |
| `TTS_DEVICE` | `auto`, `cuda`, `mps`, `cpu` | `auto` |
| `TTS_LANGUAGE` | language code | `en` |

To use the XTTS engine, install it in its own venv:
```bash
python3 -m venv .venv-xtts && source .venv-xtts/bin/activate
pip install -r requirements-xtts.txt
TTS_ENGINE=xtts streamlit run streamlit_app.py
```

## Notes

- `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` and
  `os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")` in `tts/engine.py`
  must run before any MPS torch op / before `huggingface_hub` is imported
  (transitively, via chatterbox) — both are set at module import time.
- Chatterbox Turbo's `prepare_conditionals` asserts reference clips are
  strictly >5s; `ChatterboxEngine.synthesize` checks this up front (via
  `soundfile.info`) to raise a clear `ValueError` instead of an
  `AssertionError`.
- Model weights lazy-load on first `.model` access via `from_pretrained`, but
  `setup.sh`/`setup.ps1` pre-fetch them (`device="cpu"`, just to download —
  the app picks the real device at runtime) so the first generation doesn't
  stall on a multi-GB download. Respects `CHATTERBOX_VARIANT` if set.
- ruff config: line length 100, target py39, first-party packages are
  `audio`, `tts`, `services`, `app`, `core`.
