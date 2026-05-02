import os
import sys
import numpy as np
import soundfile as sf

INPUT_FILE  = "2026-05-02 17-06-34_cleaned.wav"
OUTPUT_FILE = "2026-05-02 17-06-34_deverbed.wav"

# Section of the file that contains only room tone (no voice) — used to
# estimate the reverb/room signature. Adjust if speech starts before 1s.
ROOM_START_SEC = 0.0
ROOM_END_SEC   = 1.0

# How aggressively to subtract the room tone spectrum.
# Raise toward 2.0 if room reverb is still audible; lower toward 0.8 if
# voice sounds hollow or tinny after processing.
SUBTRACTION_FACTOR = 1.5

# Spectral floor: fraction of original magnitude to keep as minimum.
# Prevents over-subtraction artifacts ("musical noise").
SPECTRAL_FLOOR = 0.05

# Larger FFT window captures longer reverb tails better than the 2048-sample
# window used in denoise_phase_inversion.py.
N_FFT = 4096


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
        # fallback: use whatever we have
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

    print(f"  Estimating room tone from {ROOM_START_SEC}s–{ROOM_END_SEC}s...")
    room_power = estimate_room_spectrum(mono[start_samp:end_samp])

    if data.ndim == 1:
        print("  Processing mono channel...")
        return deverb_channel(data, room_power)

    channels = []
    for ch in range(data.shape[1]):
        print(f"  Processing channel {ch + 1}/{data.shape[1]}...")
        channels.append(deverb_channel(data[:, ch], room_power))
    return np.stack(channels, axis=1)


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found.")
        sys.exit(1)

    print(f"Loading '{INPUT_FILE}'...")
    audio, rate = load_wav(INPUT_FILE)
    shape_str = f"{audio.shape[0] / rate:.2f}s, {rate}Hz"
    shape_str += f", {'stereo' if audio.ndim > 1 and audio.shape[1] == 2 else 'mono'}"
    print(f"  {shape_str}")

    print("Applying deverb (room tone spectral subtraction)...")
    result = deverb(audio, rate)

    print(f"Saving '{OUTPUT_FILE}'...")
    save_wav(OUTPUT_FILE, result, rate)
    print("Done.")
