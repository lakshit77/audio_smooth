# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Uttam Voice is a Python + browser toolkit for cleaning up recorded audio (e.g. screen recordings or lectures). The pipeline:

1. **Extract audio** from a video file → WAV
2. **Denoise** the WAV using one of two strategies
3. **Remove filler words** from the denoised WAV
4. **Review** the results side-by-side in a browser player

## Running the Pipeline

Each script is run independently in sequence. All scripts hardcode the filenames at the top — change `INPUT_FILE` / `OUTPUT_FILE` constants before running if the source file differs.

```bash
# Step 1: Extract audio from MKV → WAV (requires ffmpeg on PATH)
python extract_audio.py

# Step 2a: Denoise using Demucs neural source separation (slow, high quality)
python denoise_audio.py

# Step 2b: Denoise using spectral subtraction / phase inversion (fast, no ML)
python denoise_phase_inversion.py

# Step 3a: Remove filler words using a keyword list (Whisper + pydub)
python remove_filler.py

# Step 3b: Remove filler words using Claude via OpenRouter (context-aware)
python remove_filler_llm.py

# Launch the browser player (serves files + /tracks API on port 8765)
python server.py
# Then open: http://localhost:8765/player.html
```

## Dependencies

Python packages: `whisper`, `pydub`, `soundfile`, `numpy`, `torch`, `torchaudio`, `demucs`, `requests`

System: `ffmpeg` (must be on PATH for `extract_audio.py`)

## Architecture

### Python scripts (no shared modules — all standalone)

| File | Purpose |
|---|---|
| `extract_audio.py` | Shells out to `ffmpeg` to strip audio from video |
| `denoise_audio.py` | Loads WAV via `torchaudio`, runs Demucs `htdemucs` model, keeps only the `vocals` stem |
| `denoise_phase_inversion.py` | Pure NumPy spectral subtraction: samples noise from first 1s, subtracts its power spectrum from the rest |
| `remove_filler.py` | Transcribes with Whisper (`word_timestamps=True`), matches against a hardcoded keyword set, splices audio with `pydub` |
| `remove_filler_llm.py` | Same transcription step, but sends the numbered word list to Claude (via OpenRouter) to identify fillers contextually, then splices |
| `server.py` | `http.server`-based dev server; serves static files and a `/tracks` endpoint that returns a JSON list of audio files in the directory |

### Browser player (`player.html`)

Single self-contained HTML file with inline CSS and JS. No build step.

**State model**: a single `state` object holds all app state (tracks array, single/compare mode, per-slot peaks, animation frame IDs, AudioContext).

**Two modes**:
- *Single*: one waveform, standard transport controls, speed/volume
- *Compare*: two waveforms side-by-side (slots A and B) + an "Overlap & Diff" canvas that overlays both waveforms and highlights regions where they differ above a configurable threshold

**Key rendering functions**:
- `extractPeaks(url, numCols)` — fetches audio, decodes via Web Audio API, computes RMS per column
- `drawWaveform(canvas, peaks, progress, ...)` — bar-graph waveform with playhead
- `computeDiff(peaksA, peaksB)` / `drawDiff(progress)` — difference canvas

**Server integration**: on init, `player.html` fetches `/tracks` from `server.py` to populate the sidebar. Falls back gracefully if the server isn't running (user can drag-and-drop files instead).

## Known Issues / Gotchas

- `remove_filler.py` has a bug: line 33 references `FILLER_WORD` (singular, undefined) instead of iterating over the `FILLER_WORDS` set.
- The OpenRouter API key in `remove_filler_llm.py` is hardcoded in plaintext — move it to an environment variable before sharing or committing.
- `denoise_phase_inversion.py` assumes the first second of audio is silence/noise. Adjust `NOISE_START_SEC` / `NOISE_END_SEC` if the recording doesn't start with a clean noise sample.
- All Python scripts hardcode the filenames at the top of the file. There is no CLI argument parsing.
