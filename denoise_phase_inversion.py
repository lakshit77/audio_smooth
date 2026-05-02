import os
import sys
import numpy as np
import soundfile as sf

INPUT_FILE       = "2026-05-02 17-06-34.wav"
OUTPUT_FILE      = "2026-05-02 17-06-34_phase_inversion.wav"

# Noise profile: section assumed to contain only background noise (no voice)
NOISE_START_SEC  = 0.0
NOISE_END_SEC    = 1.0

# How much of the inverted noise to subtract (1.0 = full subtraction)
# Lower if voice sounds distorted; raise if background is still audible
SUBTRACTION_FACTOR = 1.0

# Spectral floor: prevents over-subtraction from creating musical noise artifacts
# Value between 0.0 and 1.0 — keeps at least this fraction of original magnitude
SPECTRAL_FLOOR   = 0.02


def load_wav(path):
    data, rate = sf.read(path, dtype="float32")
    return data, rate


def spectral_subtract_channel(signal, noise_profile, rate):
    n_fft = 2048
    hop   = n_fft // 4

    # Estimate average noise power spectrum from the noise profile
    noise_frames = []
    for start in range(0, len(noise_profile) - n_fft, hop):
        frame = noise_profile[start:start + n_fft] * np.hanning(n_fft)
        noise_frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    noise_power = np.mean(noise_frames, axis=0)

    # Process the full signal frame by frame
    output = np.zeros(len(signal))
    window_sum = np.zeros(len(signal))
    window = np.hanning(n_fft)

    for start in range(0, len(signal) - n_fft, hop):
        frame = signal[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase     = np.angle(spectrum)

        # Subtract estimated noise power from signal power
        signal_power    = magnitude ** 2
        clean_power     = signal_power - SUBTRACTION_FACTOR * noise_power
        # Apply spectral floor to prevent artifacts from over-subtraction
        floor           = SPECTRAL_FLOOR * signal_power
        clean_power     = np.maximum(clean_power, floor)
        clean_magnitude = np.sqrt(clean_power)

        # Reconstruct with original phase (phase inversion step)
        clean_spectrum = clean_magnitude * np.exp(1j * phase)
        clean_frame    = np.fft.irfft(clean_spectrum) * window

        output[start:start + n_fft]      += clean_frame
        window_sum[start:start + n_fft]  += window ** 2

    # Normalize by window overlap
    window_sum = np.where(window_sum < 1e-8, 1.0, window_sum)
    output /= window_sum

    return output.astype(np.float32)


def denoise_stereo(data, rate):
    noise_start = int(NOISE_START_SEC * rate)
    noise_end   = int(NOISE_END_SEC * rate)

    left  = data[:, 0]
    right = data[:, 1]

    print("  Processing left channel...")
    denoised_left  = spectral_subtract_channel(left,  left[noise_start:noise_end],  rate)
    print("  Processing right channel...")
    denoised_right = spectral_subtract_channel(right, right[noise_start:noise_end], rate)

    return np.stack([denoised_left, denoised_right], axis=1)


def save_wav(path, data, rate):
    clipped = np.clip(data, -1.0, 1.0)
    sf.write(path, clipped, rate, subtype="PCM_16")


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found.")
        sys.exit(1)

    print(f"Loading '{INPUT_FILE}'...")
    audio, sample_rate = load_wav(INPUT_FILE)
    print(f"  {audio.shape[0] / sample_rate:.2f}s, {sample_rate}Hz, {audio.shape[1]}ch")

    print("Applying spectral subtraction (phase inversion)...")
    denoised = denoise_stereo(audio, sample_rate)

    print(f"Saving '{OUTPUT_FILE}'...")
    save_wav(OUTPUT_FILE, denoised, sample_rate)
    print("Done!")
