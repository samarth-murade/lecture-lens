"""
Lecture Notes Helper
Audio -> cleaned audio -> transcript (Whisper) -> summary + questions (Gemma)

Usage:
    python lecture_notes.py lec1.m4a 3 20
(Process 3 minutes of audio, starting at minute 20. Good for testing a middle part.
 If you leave out the last number, it starts from minute 0.)
"""
import subprocess
import sys
import wave

import numpy as np
import ollama
from faster_whisper import WhisperModel

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "recording.mp3"
MINUTES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
START = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # start minute

WHISPER_SIZE = "medium"  # "small" is faster but was too inaccurate for this audio
LANG = "hi"              # lecture is Hindi/Hinglish. Use None to auto-detect
# A hint that tells Whisper what the lecture is about (helps with technical words)
SCRIPT = "roman"         # "roman" = Hinglish in English letters, "devanagari" = Hindi script
PROMPT_ROMAN = "Yeh C programming ki class hai. Isme printf, scanf, int, float, variable, loop, array aur function jaise words aate hain."
PROMPT_DEVANAGARI = "यह C programming की क्लास है। printf, scanf, int, float, variable, loop, array और function जैसे शब्द आते हैं।"
PROMPT = PROMPT_ROMAN if SCRIPT == "roman" else PROMPT_DEVANAGARI

# A Whisper model fine-tuned on noisy Indian-accented Hindi that writes Hinglish directly.
# Community CTranslate2 conversion (about 1.5 GB, downloaded once). Set False to use plain Whisper.
USE_HINGLISH_MODEL = True
HINGLISH_MODEL = "Hub84/faster-whisper-hinglish-prime"
LLM = "gemma4:e4b"       # must match the name shown by `ollama list`
CHUNK_CHARS = 3000       # transcript is split into pieces of this many characters


def clean_audio():
    """Make the audio louder and cleaner using ffmpeg."""
    print("1/3: Cleaning and boosting the audio...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(START * 60), "-i", AUDIO,
            "-t", str(MINUTES * 60),
            "-af", "highpass=f=100,afftdn=nf=-25,loudnorm",  # remove rumble, reduce noise, normalize volume
            "-ar", "16000", "-ac", "1",        # 16kHz mono, what Whisper expects
            "clean.wav",
        ],
        check=True,
    )


def load_wav(path):
    """Read clean.wav (16kHz, mono, 16-bit) directly into a numpy array."""
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe():
    """Convert speech to text with faster-whisper."""
    print("2/3: Transcribing (this takes a while on CPU, please wait)...")
    model_name = HINGLISH_MODEL if USE_HINGLISH_MODEL else WHISPER_SIZE
    lang = "en" if USE_HINGLISH_MODEL else LANG      # Hinglish model writes Roman letters
    prompt = None if USE_HINGLISH_MODEL else PROMPT
    print("   Using model:", model_name)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    audio = load_wav("clean.wav")
    segments, info = model.transcribe(
        audio,
        language=lang,
        beam_size=5,
        vad_filter=True,                    # skip silence (stops "aap aap aap" loops)
        condition_on_previous_text=False,   # stops repeated-word loops
        initial_prompt=prompt,
    )
    print("Detected language:", info.language)
    text = " ".join(seg.text.strip() for seg in segments)
    if SCRIPT == "roman" and has_devanagari(text):
        print("   Converting Hindi script to Hinglish with Gemma...")
        text = romanize(text)
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved transcript.txt. Characters:", len(text))
    return text


def ask(prompt):
    """Send a prompt to Gemma running locally in Ollama."""
    reply = ollama.chat(
        model=LLM,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192},  # larger context window than the default
    )
    return reply["message"]["content"]


def has_devanagari(text):
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def romanize(text):
    """Ask Gemma to rewrite Devanagari text in Roman letters (Hinglish), piece by piece."""
    pieces = [text[i:i + 1500] for i in range(0, len(text), 1500)]
    out = []
    for piece in pieces:
        out.append(
            ask(
                "Rewrite the following Hindi text in Roman script (Hinglish), the way "
                "people type on WhatsApp. Keep English words as they are. Do not "
                "translate, summarize, correct or add anything. Output only the "
                "rewritten text.\n\n" + piece
            )
        )
    return " ".join(out)


def summarize(text):
    """Summarize the transcript piece by piece, then combine the results."""
    print("3/3: Gemma is writing the summary...")
    parts = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
    partial = []
    for n, part in enumerate(parts, 1):
        print(f"   part {n}/{len(parts)}")
        partial.append(
            ask(
                "Below is one part of a college lecture transcript. It may contain "
                "speech-recognition mistakes. Explain it as short bullet points "
                "in simple Hinglish (Hindi written in English letters).\n\n" + part
            )
        )
    combined = "\n\n".join(partial)
    final = ask(
        "Below are notes from different parts of one lecture. Combine them into "
        "one clean output with:\n"
        "1) A short summary (5-6 lines)\n"
        "2) Important points (bullet list)\n"
        "3) 5 revision questions\n"
        "Write in simple Hinglish (Hindi written in English letters).\n\n" + combined
    )
    with open("summary.md", "w", encoding="utf-8") as f:
        f.write(final)
    print("Saved summary.md. Done!")


if __name__ == "__main__":
    clean_audio()
    summarize(transcribe())
