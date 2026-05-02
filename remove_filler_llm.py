import os
import whisper
import json
import requests
import numpy as np
import wave
from pydub import AudioSegment

INPUT_FILE = "2026-05-02 17-06-34.wav"
OUTPUT_FILE = "2026-05-02 17-06-34_cleaned.wav"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "anthropic/claude-sonnet-4-5"

if not OPENROUTER_API_KEY:
    raise SystemExit(
        "OPENROUTER_API_KEY is not set. Export it in your shell or use a .env file "
        "(never commit API keys to git)."
    )

# --- Step 1: Transcribe with Whisper ---
print("Loading Whisper model...")
model = whisper.load_model("small")

print("Transcribing audio with word-level timestamps...")
result = model.transcribe(INPUT_FILE, word_timestamps=True)

words = []
for segment in result["segments"]:
    for word_info in segment.get("words", []):
        words.append({
            "index": len(words),
            "word": word_info["word"].strip(),
            "start": word_info["start"],
            "end": word_info["end"]
        })

print(f"\nFull transcript:\n{result['text']}\n")
print(f"Total words detected: {len(words)}")

# --- Step 2: Detect unrecognized vocalizations (e.g. "aaaa", "mmm") in gaps ---
with wave.open(INPUT_FILE, 'r') as f:
    framerate = f.getframerate()
    n_frames = f.getnframes()
    raw = f.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

total_duration = n_frames / framerate

# Find gaps between words and check if audio energy is high (vocalization present)
GAP_ENERGY_THRESHOLD = 300  # RMS energy above this = vocalization in gap
MIN_GAP_DURATION = 0.08     # ignore gaps shorter than 80ms

gap_segments = []

# Check gap before first word
if words and words[0]["start"] > MIN_GAP_DURATION:
    gap_start = 0.0
    gap_end = words[0]["start"]
    s = int(gap_start * framerate)
    e = int(gap_end * framerate)
    energy = np.sqrt(np.mean(samples[s:e] ** 2)) if e > s else 0
    if energy > GAP_ENERGY_THRESHOLD:
        gap_segments.append({"start": gap_start, "end": gap_end, "label": f"[vocalization gap before word 0]"})

# Check gaps between consecutive words
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
    print(f"\nDetected {len(gap_segments)} unrecognized vocalization(s) in gaps:")
    for g in gap_segments:
        print(f"  {g['label']}: {g['start']:.2f}s - {g['end']:.2f}s")
else:
    print("\nNo unrecognized vocalizations detected in gaps.")

# --- Step 3: Ask Claude to identify filler words from transcript ---
numbered_words = "\n".join(
    f"{w['index']}: \"{w['word']}\" ({w['start']:.2f}s - {w['end']:.2f}s)"
    for w in words
)

prompt = f"""You are an expert audio editor. Below is a list of words (with their index and timestamps) transcribed from a spoken audio recording.

Your job is to identify which words are FILLER words — words that add no meaning and are used as verbal pauses or habits (e.g., "um", "uh", "like" when used as a filler, "you know", "I mean", "basically", "so" at the start of a sentence as a filler, "right" used as a filler, etc.).

Be context-aware: only flag a word as a filler if it truly adds no semantic meaning in context. For example:
- "so" in "So today we are going to learn..." — this is a filler (sentence opener with no meaning)
- "like" in "I like this" — NOT a filler
- "right" in "turn right" — NOT a filler
- "right" at the end of a sentence seeking agreement — IS a filler

Word list:
{numbered_words}

Full transcript for context:
{result['text']}

Respond with ONLY a JSON array of indexes of filler words. Example: [2, 5, 11]
If there are no filler words, respond with: []"""

print("\nAsking Claude to identify filler words...")
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
)

response.raise_for_status()
reply = response.json()["choices"][0]["message"]["content"].strip()
print(f"Claude's response: {reply}")

filler_indexes = json.loads(reply)
filler_words = [words[i] for i in filler_indexes]

# --- Step 4: Merge filler word segments + gap vocalizations ---
segments_to_remove = []

for w in filler_words:
    segments_to_remove.append({"start": w["start"], "end": w["end"], "label": f"\"{w['word']}\""})

for g in gap_segments:
    segments_to_remove.append({"start": g["start"], "end": g["end"], "label": g["label"]})

# Sort and merge overlapping segments
segments_to_remove.sort(key=lambda x: x["start"])
merged = []
for seg in segments_to_remove:
    if merged and seg["start"] <= merged[-1]["end"]:
        merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        merged[-1]["label"] += " + " + seg["label"]
    else:
        merged.append(dict(seg))

# --- Step 5: Cut segments and stitch audio ---
print(f"\nTotal segments to remove: {len(merged)}")
for seg in merged:
    print(f"  {seg['label']}: {seg['start']:.2f}s - {seg['end']:.2f}s")

audio = AudioSegment.from_wav(INPUT_FILE)
cleaned = AudioSegment.empty()
prev_end_ms = 0

for seg in merged:
    start_ms = int(seg["start"] * 1000)
    end_ms = int(seg["end"] * 1000)
    cleaned += audio[prev_end_ms:start_ms]
    prev_end_ms = end_ms

cleaned += audio[prev_end_ms:]

print(f"\nOriginal duration: {len(audio) / 1000:.2f}s")
print(f"Cleaned duration:  {len(cleaned) / 1000:.2f}s")
print(f"Removed:           {(len(audio) - len(cleaned)) / 1000:.2f}s")

cleaned.export(OUTPUT_FILE, format="wav")
print(f"\nSaved cleaned audio to: {OUTPUT_FILE}")
