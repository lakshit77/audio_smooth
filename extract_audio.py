import subprocess
import sys
import os


def extract_audio_ffmpeg(video_path: str, output_path: str) -> None:
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_path,
        "-y",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:", result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    video_file = "2026-05-02 17-06-34.mkv"
    output_file = "2026-05-02 17-06-34.wav"

    if not os.path.exists(video_file):
        print(f"Error: '{video_file}' not found.")
        sys.exit(1)

    print(f"Extracting audio from '{video_file}'...")
    extract_audio_ffmpeg(video_file, output_file)
    print(f"Done! Audio saved to '{output_file}'")
