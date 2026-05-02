import os
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

INPUT_FILE  = "2026-05-02 17-06-34.wav"
OUTPUT_FILE = "2026-05-02 17-06-34_denoised.wav"

# htdemucs separates: drums, bass, other, vocals
# We keep only "vocals" and discard everything else (background noise)
TARGET_STEM = "vocals"


def load_wav_torch(path):
    waveform, rate = torchaudio.load(path)  # shape: (channels, samples)
    return waveform, rate


def save_wav(path, waveform, rate):
    # waveform: (channels, samples) float tensor
    data = waveform.numpy().T  # → (samples, channels)
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    sf.write(path, data, rate, subtype="PCM_16")


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found.")
        sys.exit(1)

    print(f"Loading '{INPUT_FILE}'...")
    waveform, sample_rate = load_wav_torch(INPUT_FILE)
    print(f"  {waveform.shape[1] / sample_rate:.2f}s, {sample_rate}Hz, {waveform.shape[0]}ch")

    print("Loading demucs model (htdemucs)...")
    model = get_model("htdemucs")
    model.eval()

    # demucs expects shape (batch, channels, samples) and its own sample rate (44100)
    audio = waveform.unsqueeze(0)  # (1, channels, samples)
    if sample_rate != model.samplerate:
        audio = torchaudio.functional.resample(audio, sample_rate, model.samplerate)

    print("Separating vocals from background (this may take a minute)...")
    with torch.no_grad():
        sources = apply_model(model, audio, device="cpu", progress=True)
    # sources shape: (batch, stems, channels, samples)
    # stem order matches model.sources list

    stem_idx = model.sources.index(TARGET_STEM)
    vocals = sources[0, stem_idx]  # (channels, samples)

    # Resample back if needed
    if sample_rate != model.samplerate:
        vocals = torchaudio.functional.resample(vocals, model.samplerate, sample_rate)

    print(f"Saving '{OUTPUT_FILE}'...")
    save_wav(OUTPUT_FILE, vocals, sample_rate)
    print("Done!")
