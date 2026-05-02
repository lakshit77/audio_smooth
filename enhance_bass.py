import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

INPUT_FILE  = "2026-05-02 17-06-34_cleaned.wav"
OUTPUT_FILE = "2026-05-02 17-06-34_bass_enhanced.wav"

# Low-shelf boost parameters
SHELF_FREQ_HZ = 150   # frequencies below this get boosted
GAIN_DB       = 14.0  # how much to boost (dB)

# Extra sub-bass shelf (deep voice body)
SUB_FREQ_HZ   = 80    # second shelf cutoff
SUB_GAIN_DB   = 8.0   # additional boost below SUB_FREQ_HZ

def low_shelf_sos(cutoff, gain_db, fs):
    """2nd-order low-shelf filter via bilinear transform."""
    A  = 10 ** (gain_db / 40.0)   # amplitude ratio (half-power)
    w0 = 2 * np.pi * cutoff / fs
    cosw0 = np.cos(w0)
    sinw0 = np.sin(w0)
    alpha = sinw0 / 2 * np.sqrt((A + 1/A) * (1/1 - 1) + 2)  # S=1

    # Standard Audio EQ Cookbook low-shelf coefficients
    b0 =      A * ((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 =  2 * A * ((A-1) - (A+1)*cosw0)
    b2 =      A * ((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =           (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =     -2 * ((A-1) + (A+1)*cosw0)
    a2 =           (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def main():
    audio, sr = sf.read(INPUT_FILE, dtype="float32")
    print(f"Loaded: {INPUT_FILE}  ({sr} Hz, {audio.shape})")

    sos_main = low_shelf_sos(SHELF_FREQ_HZ, GAIN_DB, sr)
    sos_sub  = low_shelf_sos(SUB_FREQ_HZ,   SUB_GAIN_DB, sr)

    if audio.ndim == 1:
        enhanced = sosfilt(sos_sub, sosfilt(sos_main, audio))
    else:
        enhanced = np.stack(
            [sosfilt(sos_sub, sosfilt(sos_main, audio[:, ch])) for ch in range(audio.shape[1])],
            axis=1
        )

    # Prevent clipping
    peak = np.max(np.abs(enhanced))
    if peak > 0.99:
        enhanced = enhanced * (0.99 / peak)
        print(f"Peak clipping prevented (scale={0.99/peak:.4f})")

    sf.write(OUTPUT_FILE, enhanced, sr)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
