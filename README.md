# 🎧 Lecture Lens

**Clear audio. Easy notes. No cloud.**

Lecture Lens turns a quiet, noisy class recording into louder, clearer audio and
easy-to-revise notes. Everything runs on your own laptop using open-source and
open-weight AI, so a recording never has to be uploaded to an online service.

> Built for my sister, who sat at the front of class and recorded every lecture
> just to follow it, because the teacher's voice was too quiet to hear from anywhere else.

## What it does

- **Enhance audio:** makes a quiet recording louder and reduces background noise
  (ffmpeg), then lets you play and download the result as MP3.
- **Make notes:** converts speech to a transcript, then writes a summary, key points
  and 5 revision questions, in Hinglish (Hindi in English letters) or English.
- **Pick the part you need:** choose a start minute and how many minutes to process.
- **Private by design:** all processing happens on your laptop.

## How the AI is used

| Step | Tool | Role |
|---|---|---|
| Clean the audio | ffmpeg | Not AI: high-pass filter, noise reduction, loudness normalization |
| Speech to text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with a Hinglish fine-tuned Whisper model (`Hub84/faster-whisper-hinglish-prime`, a community CTranslate2 conversion) | Writes the transcript. Also supports the original Whisper `medium` and `small` models. |
| Summary and questions | Gemma 4 (`gemma4:e4b`) through [Ollama](https://ollama.com) | Writes the summary, key points and revision questions |
| Interface | Streamlit | Web page that runs locally |

Please check the Hugging Face model card of the Hinglish model for the original
author and its license.

## Install (Windows)

1. Install [Python 3](https://www.python.org/downloads/) (tick "Add Python to PATH"),
   [Ollama](https://ollama.com/download) and ffmpeg (`winget install ffmpeg`).
2. Download Gemma:
   ```
   ollama pull gemma4:e4b
   ```
3. Install the Python libraries:
   ```
   pip install faster-whisper ollama streamlit numpy
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```
   The first run downloads the speech model (about 1.5 GB). After that it works offline.

There is also a command-line version:
```
python lecture_notes.py lecture.m4a 3 20
```
(process 3 minutes, starting at minute 20).

## Limitations (honest notes)

- Everything runs on a laptop CPU, so long recordings are slow. Process a few
  minutes at a time.
- Transcript quality depends heavily on the recording. Very quiet, distant or noisy
  audio gives more mistakes, and the first minutes of a class are often noise.
- Notes are written by AI and can contain mistakes. Check important points against
  the original audio.
- The app has no login. If you open it to other devices on your Wi-Fi, only do so on
  a network you trust.
- Tested on Windows 10 with Python 3.14.

## Open source

Released under the [MIT License](LICENSE). Built for Hacktoberfest 2026.
Written with help from an AI coding assistant (Claude) and tested on a real
classroom recording.
