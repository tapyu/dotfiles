import os
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 8
MODEL = "large-v3"

print("🎤 Speak...", file=sys.stderr)

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

sd.wait()

print("Transcribing...", file=sys.stderr)

model = WhisperModel(
    MODEL,
    device="cuda",
    compute_type="int8_float32",
)

segments, info = model.transcribe(
    audio.flatten(),
    language="en",
    beam_size=5,
    vad_filter=True,
    initial_prompt=(
        "Linux terminal commands, Git, GitHub, "
        "Zsh, Python, Ubuntu, directories, files."
    ),
)

text = " ".join(segment.text.strip() for segment in segments)
print(text)

