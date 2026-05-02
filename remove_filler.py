import whisper
import json
from pydub import AudioSegment

INPUT_FILE = "2026-05-02 17-06-34.wav"
OUTPUT_FILE = "2026-05-02 17-06-34_cleaned.wav"
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "hmm", "hm",
    "like", "so", "basically", "literally",
    "you know", "i mean", "right", "okay", "ok",
    "actually", "honestly", "well", "anyway",
    "kind of", "sort of", "you see"
}

print("Loading Whisper model...")
model = whisper.load_model("base")

print("Transcribing audio with word-level timestamps...")
result = model.transcribe(INPUT_FILE, word_timestamps=True)

# Collect all word segments
words = []
for segment in result["segments"]:
    for word_info in segment.get("words", []):
        words.append(word_info)

print(f"\nFull transcript:\n{result['text']}\n")

# Find filler word occurrences
fillers = []
for w in words:
    text = w["word"].strip().lower().strip(".,!?;:\"'")
    if text == FILLER_WORD:
        fillers.append((w["start"], w["end"], w["word"]))

print(f"Found {len(fillers)} instance(s) of '{FILLER_WORD}':")
for start, end, word in fillers:
    print(f"  '{word}' at {start:.2f}s - {end:.2f}s")

if not fillers:
    print("No filler words found. Exiting.")
    exit(0)

print("\nLoading audio...")
audio = AudioSegment.from_wav(INPUT_FILE)

# Sort by start time descending isn't needed — we build by slicing
# Build cleaned audio by keeping everything except filler segments
cleaned = AudioSegment.empty()
prev_end_ms = 0

for start, end, word in fillers:
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    cleaned += audio[prev_end_ms:start_ms]
    prev_end_ms = end_ms

# Append remaining audio after last filler
cleaned += audio[prev_end_ms:]

print(f"\nOriginal duration: {len(audio) / 1000:.2f}s")
print(f"Cleaned duration:  {len(cleaned) / 1000:.2f}s")
print(f"Removed:           {(len(audio) - len(cleaned)) / 1000:.2f}s")

cleaned.export(OUTPUT_FILE, format="wav")
print(f"\nSaved cleaned audio to: {OUTPUT_FILE}")
