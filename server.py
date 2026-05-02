#!/usr/bin/env python3
"""
Simple dev server for Uttam Voice Player.
Serves static files and exposes:
  GET  /tracks   -> JSON list of audio files
  POST /process  -> multipart upload + process, streams NDJSON log lines
"""
import cgi
import io
import json
import os
import queue
import sys
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

AUDIO_EXTS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.opus'}
PORT = 8765
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────
# Processing functions (logic inlined from the original scripts;
# originals are NOT modified)
# ─────────────────────────────────────────────────────────────────

def _run_extract_audio(input_path, output_path, log_callback=None):
    """Extract audio track from a video file using ffmpeg."""
    import subprocess
    if log_callback:
        log_callback(f"Extracting audio from '{os.path.basename(input_path)}'...")
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_path,
        "-y",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")
    if log_callback:
        log_callback(f"Audio saved to '{os.path.basename(output_path)}'")


def _run_remove_fillers(input_path, output_path, log_callback=None):
    """Remove filler words using Whisper + Claude via OpenRouter."""
    import whisper
    import json as _json
    import requests
    import numpy as np
    import wave
    from pydub import AudioSegment

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before using LLM filler removal."
        )
    MODEL = "anthropic/claude-sonnet-4-5"

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log("Loading Whisper model (small)...")
    model = whisper.load_model("small")

    log("Transcribing audio with word-level timestamps...")
    result = model.transcribe(input_path, word_timestamps=True)

    words = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            words.append({
                "index": len(words),
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })

    log(f"Full transcript: {result['text']}")
    log(f"Total words detected: {len(words)}")

    # Detect unrecognized vocalizations in gaps
    with wave.open(input_path, 'r') as f:
        framerate = f.getframerate()
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    GAP_ENERGY_THRESHOLD = 300
    MIN_GAP_DURATION = 0.08

    gap_segments = []

    if words and words[0]["start"] > MIN_GAP_DURATION:
        gap_start = 0.0
        gap_end = words[0]["start"]
        s = int(gap_start * framerate)
        e = int(gap_end * framerate)
        energy = np.sqrt(np.mean(samples[s:e] ** 2)) if e > s else 0
        if energy > GAP_ENERGY_THRESHOLD:
            gap_segments.append({"start": gap_start, "end": gap_end, "label": "[vocalization gap before word 0]"})

    for i in range(len(words) - 1):
        gap_start = words[i]["end"]
        gap_end = words[i + 1]["start"]
        if gap_end - gap_start > MIN_GAP_DURATION:
            s = int(gap_start * framerate)
            e = int(gap_end * framerate)
            energy = np.sqrt(np.mean(samples[s:e] ** 2)) if e > s else 0
            if energy > GAP_ENERGY_THRESHOLD:
                gap_segments.append({
                    "start": gap_start,
                    "end": gap_end,
                    "label": f"[vocalization gap between word {i} and {i+1}]"
                })

    if gap_segments:
        log(f"Detected {len(gap_segments)} unrecognized vocalization(s) in gaps.")
    else:
        log("No unrecognized vocalizations detected in gaps.")

    # Ask Claude to identify filler words
    numbered_words = "\n".join(
        f"{w['index']}: \"{w['word']}\" ({w['start']:.2f}s - {w['end']:.2f}s)"
        for w in words
    )

    prompt = f"""You are an expert audio editor. Below is a list of words (with their index and timestamps) transcribed from a spoken audio recording.

Your job is to identify which words are FILLER words — words that add no meaning and are used as verbal pauses or habits (e.g., "um", "uh", "like" when used as a filler, "you know", "I mean", "basically", "so" at the start of a sentence as a filler, "right" used as a filler, etc.).

Be context-aware: only flag a word as a filler if it truly adds no semantic meaning in context.

Word list:
{numbered_words}

Full transcript for context:
{result['text']}

Respond with ONLY a JSON array of indexes of filler words. Example: [2, 5, 11]
If there are no filler words, respond with: []"""

    log("Asking Claude to identify filler words...")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"].strip()
    log(f"Claude's response: {reply}")

    filler_indexes = _json.loads(reply)
    filler_words = [words[i] for i in filler_indexes]

    segments_to_remove = []
    for w in filler_words:
        segments_to_remove.append({"start": w["start"], "end": w["end"], "label": f"\"{w['word']}\""})
    for g in gap_segments:
        segments_to_remove.append({"start": g["start"], "end": g["end"], "label": g["label"]})

    segments_to_remove.sort(key=lambda x: x["start"])
    merged = []
    for seg in segments_to_remove:
        if merged and seg["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merged[-1]["label"] += " + " + seg["label"]
        else:
            merged.append(dict(seg))

    log(f"Total segments to remove: {len(merged)}")
    for seg in merged:
        log(f"  {seg['label']}: {seg['start']:.2f}s - {seg['end']:.2f}s")

    audio = AudioSegment.from_wav(input_path)
    cleaned = AudioSegment.empty()
    prev_end_ms = 0

    for seg in merged:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        cleaned += audio[prev_end_ms:start_ms]
        prev_end_ms = end_ms

    cleaned += audio[prev_end_ms:]

    log(f"Original duration: {len(audio) / 1000:.2f}s")
    log(f"Cleaned duration:  {len(cleaned) / 1000:.2f}s")
    log(f"Removed:           {(len(audio) - len(cleaned)) / 1000:.2f}s")

    cleaned.export(output_path, format="wav")
    log(f"Saved cleaned audio to: '{os.path.basename(output_path)}'")


def _run_denoise(input_path, output_path, log_callback=None):
    """Denoise audio using the Demucs htdemucs model (vocals stem)."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    TARGET_STEM = "vocals"

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def load_wav_torch(path):
        waveform, rate = torchaudio.load(path)
        return waveform, rate

    def save_wav(path, waveform, rate):
        data = waveform.numpy().T
        data = np.clip(data, -1.0, 1.0).astype(np.float32)
        sf.write(path, data, rate, subtype="PCM_16")

    log(f"Loading '{os.path.basename(input_path)}'...")
    waveform, sample_rate = load_wav_torch(input_path)
    log(f"  {waveform.shape[1] / sample_rate:.2f}s, {sample_rate}Hz, {waveform.shape[0]}ch")

    log("Loading demucs model (htdemucs)...")
    model = get_model("htdemucs")
    model.eval()

    audio = waveform.unsqueeze(0)
    if sample_rate != model.samplerate:
        audio = torchaudio.functional.resample(audio, sample_rate, model.samplerate)

    log("Separating vocals from background (this may take a minute)...")
    with torch.no_grad():
        sources = apply_model(model, audio, device="cpu", progress=False)

    stem_idx = model.sources.index(TARGET_STEM)
    vocals = sources[0, stem_idx]

    if sample_rate != model.samplerate:
        vocals = torchaudio.functional.resample(vocals, model.samplerate, sample_rate)

    log(f"Saving '{os.path.basename(output_path)}'...")
    save_wav(output_path, vocals, sample_rate)
    log("Done!")


def _run_deverb(input_path, output_path, log_callback=None):
    """Remove room reverb via spectral subtraction."""
    import numpy as np
    import soundfile as sf

    ROOM_START_SEC = 0.0
    ROOM_END_SEC = 1.0
    SUBTRACTION_FACTOR = 1.5
    SPECTRAL_FLOOR = 0.05
    N_FFT = 4096

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def load_wav(path):
        data, rate = sf.read(path, dtype="float32")
        return data, rate

    def save_wav(path, data, rate):
        sf.write(path, np.clip(data, -1.0, 1.0), rate, subtype="PCM_16")

    def estimate_room_spectrum(room_sample):
        hop = N_FFT // 4
        frames = []
        for start in range(0, len(room_sample) - N_FFT, hop):
            frame = room_sample[start:start + N_FFT] * np.hanning(N_FFT)
            frames.append(np.abs(np.fft.rfft(frame)) ** 2)
        if not frames:
            frame = np.zeros(N_FFT)
            frame[:len(room_sample)] = room_sample * np.hanning(len(room_sample))
            frames.append(np.abs(np.fft.rfft(frame[:N_FFT])) ** 2)
        return np.mean(frames, axis=0)

    def deverb_channel(signal, room_power):
        hop = N_FFT // 4
        window = np.hanning(N_FFT)
        output = np.zeros(len(signal))
        window_sum = np.zeros(len(signal))
        for start in range(0, len(signal) - N_FFT, hop):
            frame = signal[start:start + N_FFT] * window
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            signal_power = magnitude ** 2
            clean_power = signal_power - SUBTRACTION_FACTOR * room_power
            clean_power = np.maximum(clean_power, SPECTRAL_FLOOR * signal_power)
            clean_magnitude = np.sqrt(clean_power)
            clean_frame = np.fft.irfft(clean_magnitude * np.exp(1j * phase)) * window
            output[start:start + N_FFT] += clean_frame
            window_sum[start:start + N_FFT] += window ** 2
        window_sum = np.where(window_sum < 1e-8, 1.0, window_sum)
        return (output / window_sum).astype(np.float32)

    def deverb(data, rate):
        start_samp = int(ROOM_START_SEC * rate)
        end_samp = int(ROOM_END_SEC * rate)
        mono = data if data.ndim == 1 else data.mean(axis=1)
        log(f"  Estimating room tone from {ROOM_START_SEC}s–{ROOM_END_SEC}s...")
        room_power = estimate_room_spectrum(mono[start_samp:end_samp])
        if data.ndim == 1:
            log("  Processing mono channel...")
            return deverb_channel(data, room_power)
        channels = []
        for ch in range(data.shape[1]):
            log(f"  Processing channel {ch + 1}/{data.shape[1]}...")
            channels.append(deverb_channel(data[:, ch], room_power))
        return np.stack(channels, axis=1)

    log(f"Loading '{os.path.basename(input_path)}'...")
    audio, rate = load_wav(input_path)
    shape_str = f"{audio.shape[0] / rate:.2f}s, {rate}Hz"
    shape_str += f", {'stereo' if audio.ndim > 1 and audio.shape[1] == 2 else 'mono'}"
    log(f"  {shape_str}")

    log("Applying deverb (room tone spectral subtraction)...")
    result = deverb(audio, rate)

    log(f"Saving '{os.path.basename(output_path)}'...")
    save_wav(output_path, result, rate)
    log("Done.")


# ─────────────────────────────────────────────────────────────────
# HTTP Handler
# ─────────────────────────────────────────────────────────────────

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/tracks':
            files = sorted(
                f for f in os.listdir(SERVE_DIR)
                if os.path.splitext(f)[1].lower() in AUDIO_EXTS
                and os.path.isfile(os.path.join(SERVE_DIR, f))
            )
            body = json.dumps(files).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != '/process':
            self.send_error(404, 'Not Found')
            return

        # Parse multipart/form-data
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_error(400, 'Expected multipart/form-data')
            return

        try:
            length = int(self.headers.get('Content-Length', 0))
            environ = {
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': content_type,
                'CONTENT_LENGTH': str(length),
            }
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ=environ,
            )
        except Exception as e:
            self.send_error(400, f'Could not parse form data: {e}')
            return

        # Extract fields
        operation = form.getvalue('operation', '').strip()
        filename = form.getvalue('filename', '').strip()
        output_name = form.getvalue('output_name', '').strip()

        file_field = form['file'] if 'file' in form else None
        file_data = file_field.file.read() if file_field and hasattr(file_field, 'file') else None

        # Validate operation
        valid_ops = {'extract_audio', 'remove_fillers', 'denoise', 'deverb'}
        if operation not in valid_ops:
            self.send_error(400, f'Invalid operation. Must be one of: {", ".join(valid_ops)}')
            return

        # If a file was uploaded, save it; otherwise assume filename already exists
        if file_data and filename:
            safe_name = os.path.basename(filename)
            input_path = os.path.join(SERVE_DIR, safe_name)
            with open(input_path, 'wb') as f:
                f.write(file_data)
        elif filename:
            safe_name = os.path.basename(filename)
            input_path = os.path.join(SERVE_DIR, safe_name)
            if not os.path.isfile(input_path):
                self.send_error(400, f"File not found on server: {safe_name}")
                return
        else:
            self.send_error(400, 'No file uploaded and no filename provided')
            return

        # Determine output filename
        base, _ = os.path.splitext(os.path.basename(input_path))
        if output_name:
            out_filename = os.path.basename(output_name)
        elif operation == 'extract_audio':
            out_filename = base + '.wav'
        elif operation == 'remove_fillers':
            out_filename = base + '_cleaned.wav'
        elif operation == 'denoise':
            out_filename = base + '_denoised.wav'
        elif operation == 'deverb':
            out_filename = base + '_deverbed.wav'
        else:
            out_filename = base + '_output.wav'

        output_path = os.path.join(SERVE_DIR, out_filename)

        # Set up streaming response
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        log_queue = queue.Queue()

        def log_callback(msg):
            log_queue.put(('log', msg))

        def worker():
            try:
                if operation == 'extract_audio':
                    _run_extract_audio(input_path, output_path, log_callback)
                elif operation == 'remove_fillers':
                    _run_remove_fillers(input_path, output_path, log_callback)
                elif operation == 'denoise':
                    _run_denoise(input_path, output_path, log_callback)
                elif operation == 'deverb':
                    _run_deverb(input_path, output_path, log_callback)
                log_queue.put(('done', out_filename))
            except Exception as e:
                log_queue.put(('error', str(e)))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        def send_line(data: dict):
            line = (json.dumps(data) + '\n').encode('utf-8')
            # Chunked transfer encoding: hex length + CRLF + data + CRLF
            chunk = f'{len(line):X}\r\n'.encode() + line + b'\r\n'
            self.wfile.write(chunk)
            self.wfile.flush()

        try:
            while True:
                try:
                    item = log_queue.get(timeout=300)  # 5 min max
                except queue.Empty:
                    send_line({'error': 'Processing timed out'})
                    break

                kind, payload = item
                if kind == 'log':
                    send_line({'log': payload})
                elif kind == 'done':
                    send_line({'done': True, 'output': payload})
                    break
                elif kind == 'error':
                    send_line({'error': payload})
                    break
        except Exception:
            pass
        finally:
            # Terminate chunked transfer
            try:
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            except Exception:
                pass
            t.join(timeout=5)

    def log_message(self, fmt, *args):
        pass  # silence request logs


if __name__ == '__main__':
    os.chdir(SERVE_DIR)
    HTTPServer.allow_reuse_address = True
    httpd = HTTPServer(('', PORT), Handler)
    print(f'Uttam Voice server running at http://localhost:{PORT}/player.html')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
