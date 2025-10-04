# asr_test.py
# Test Whisper models locally (faster-whisper + distil-whisper)
import time

from faster_whisper import WhisperModel
from transformers import pipeline

AUDIO_FILE = "test.mp3"  # put your 60s test wav file here


def run_faster_whisper(model_size="small", compute_type="float16"):
    """
    Run faster-whisper (small, medium, etc.)
    """
    model_download_start = time.time()
    print(f"\n--- Running faster-whisper-{model_size} ---")
    model = WhisperModel(model_size, device="cuda", compute_type=compute_type)

    transcription_start = time.time()
    segments, info = model.transcribe(AUDIO_FILE)
    print("Detected language:", info.language)

    transcript = []
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        transcript.append(segment.text)

    print("\nFull transcript:\n", " ".join(transcript))
    print(
        f"====Full-time-{model_size}====\nTime taken: {time.time() - model_download_start}s"
    )
    print(
        f"====Transcription-time-{model_size}====\nTime taken: {time.time() - transcription_start}s"
    )


def run_distil_whisper(model_name="distil-whisper/distil-small.en"):
    """
    Run Hugging Face distil-whisper
    """
    # Couldn't get this working, The faster whisper models are doing pretty good (1 second transcription time, for a 60second .mp3 file)
    model_download_start = time.time()
    print(f"\n--- Running {model_name} ---")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=0,
        return_timestamps=True,
    )
    transcription_start = time.time()
    result = pipe(AUDIO_FILE)
    print("\nFull transcript:\n", result["text"])
    print(
        f"====Full-time-{model_name}====\nTime taken: {time.time() - model_download_start}s"
    )
    print(
        f"====Transcription-time-{model_name}====\nTime taken: {time.time() - transcription_start}s"
    )


if __name__ == "__main__":
    # Test faster-whisper-small
    run_faster_whisper("small", compute_type="float16")  # good balance

    # Test faster-whisper-medium (might need int8 on 4GB VRAM)
    run_faster_whisper("medium", compute_type="int8")

    # Test distil-whisper-small.en (fastest)
    run_distil_whisper("distil-whisper/distil-small.en")
