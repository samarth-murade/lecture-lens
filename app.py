"""
Lecture Lens - web page
1) Enhance a quiet, noisy class recording (louder + less noise) and download it.
2) Turn the recording into a Hinglish transcript, summary, key points and revision questions.
Everything runs locally: ffmpeg, Whisper (speech-to-text) and Gemma via Ollama (summary).

Run with:
    streamlit run app.py
"""
import os
import subprocess
import tempfile
import wave

import numpy as np
import ollama
import streamlit as st
from faster_whisper import WhisperModel

LLM = "gemma4:e4b"      # must match the name shown by `ollama list`
CHUNK_CHARS = 3000      # transcript is split into pieces of this many characters

# Speech models you can pick in the page.
# The Hinglish model writes Hindi/English speech in English letters.
MODELS = {
    "Best quality (Hinglish, slower)": "Hub84/faster-whisper-hinglish-prime",
    "Hindi script (Devanagari)": "medium",
    "Quick draft (fastest, less accurate)": "small",
}
HINGLISH_MODEL = MODELS["Best quality (Hinglish, slower)"]

st.set_page_config(page_title="Lecture Lens", page_icon="🎧")
st.markdown(
    """
    <style>
    #MainMenu, footer, .stDeployButton {visibility: hidden;}
    .block-container {max-width: 760px; padding-top: 2rem;}
    .hero {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #EC4899 100%);
        border-radius: 20px; padding: 28px 30px; margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.28);
    }
    .hero h1 {color: #fff; font-size: 2.1rem; margin: 0 0 6px 0; padding: 0;}
    .hero p {color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 0 0 16px 0;}
    .pill {
        display: inline-block; background: rgba(255,255,255,0.20); color: #fff;
        border-radius: 999px; padding: 5px 13px; margin: 0 8px 6px 0;
        font-size: 0.85rem; font-weight: 600;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px; font-weight: 600; padding: 0.55rem 1.3rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5, #7C3AED); border: none; color: #fff;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }
    .stButton > button[kind="primary"]:hover {filter: brightness(1.08);}
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #7C3AED; border-radius: 16px;
    }
    .stTabs [data-baseweb="tab"] {font-weight: 600; font-size: 1rem;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
    .stTabs [aria-selected="true"] {color: #7C3AED;}
    .stTabs [data-baseweb="tab-highlight"] {background-color: #7C3AED;}
    [data-baseweb="slider"] [role="slider"] {background-color: #7C3AED;}
    [data-testid="stSliderThumbValue"], [data-testid="stThumbValue"] {color: #7C3AED;}
    </style>
    <div class="hero">
        <h1>🎧 Lecture Lens</h1>
        <p>Clear audio. Easy notes. No cloud.</p>
        <span class="pill">🔊 Louder, clearer audio</span>
        <span class="pill">📝 Easy notes in Hinglish</span>
        <span class="pill">🔒 Private, runs on your laptop</span>
        <span class="pill">🧠 Open-source AI: Whisper + Gemma</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_whisper(name):
    return WhisperModel(name, device="cpu", compute_type="int8")


def save_upload(uploaded, folder):
    path = os.path.join(folder, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def enhance_audio_file(src_path, out_path, denoise=True, start=0, minutes=None):
    """Make a quiet recording louder and cleaner for listening (MP3).
    start = minute to begin from, minutes = how long (None = until the end)."""
    filters = ["highpass=f=100"]                 # remove low rumble
    if denoise:
        filters.append("afftdn=nf=-25")          # reduce steady background noise
    filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")  # boost and even out the volume
    cmd = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", str(start * 60)]
    cmd += ["-i", src_path]
    if minutes:
        cmd += ["-t", str(minutes * 60)]
    cmd += ["-af", ",".join(filters), "-ac", "1", "-ar", "44100", "-b:a", "96k", out_path]
    subprocess.run(cmd, check=True, capture_output=True)


def clean_audio(src_path, out_path, minutes, start=0):
    """Prepare a 16kHz mono WAV for Whisper (only the chosen part of the recording)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start * 60), "-i", src_path,
            "-t", str(minutes * 60),
            "-af", "highpass=f=100,afftdn=nf=-25,loudnorm",
            "-ar", "16000", "-ac", "1",
            out_path,
        ],
        check=True,
        capture_output=True,
    )


def load_wav(path):
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def ask(prompt):
    reply = ollama.chat(
        model=LLM,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192},
    )
    return reply["message"]["content"]


def summarize(text, status, style="simple Hinglish (Hindi written in English letters)"):
    parts = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
    partial = []
    for n, part in enumerate(parts, 1):
        status.write(f"Summarizing part {n} of {len(parts)}...")
        partial.append(
            ask(
                "Below is one part of a college lecture transcript. It may contain "
                "speech-recognition mistakes. Explain it as short bullet points "
                "in " + style + ".\n\n" + part
            )
        )
    status.write("Combining everything into final notes...")
    return ask(
        "Below are notes from different parts of one lecture. Combine them into "
        "one clean output with:\n"
        "1) A short summary (5-6 lines)\n"
        "2) Important points (bullet list)\n"
        "3) 5 revision questions\n"
        "Write in " + style + ".\n\n"
        + "\n\n".join(partial)
    )


uploaded = st.file_uploader(
    "Upload recording", type=["m4a", "mp3", "wav", "aac", "ogg", "mp4"]
)

tab_audio, tab_notes = st.tabs(["🔊 Enhance audio", "📝 Make notes"])

# ---------------------------------------------------------------- Enhance audio
with tab_audio:
    st.write(
        "Makes a quiet recording louder and reduces background noise. "
        "This is fast and does not use AI. Choose which part of the recording you want."
    )
    whole = st.checkbox("Enhance the whole recording", value=False)
    if not whole:
        e_start = st.number_input(
            "Start from minute", min_value=0, max_value=300, value=0, key="e_start"
        )
        e_minutes = st.slider("How many minutes?", 1, 90, 5, key="e_minutes")
    denoise = st.checkbox("Reduce background noise", value=True)
    if uploaded and st.button("Enhance audio", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            src = save_upload(uploaded, tmp)
            out = os.path.join(tmp, "enhanced.mp3")
            with st.spinner("Enhancing the recording..."):
                try:
                    enhance_audio_file(
                        src, out, denoise,
                        start=0 if whole else int(e_start),
                        minutes=None if whole else int(e_minutes),
                    )
                    with open(out, "rb") as f:
                        st.session_state["enhanced"] = f.read()
                except Exception as e:
                    st.error(f"Error: {e}")

    if "enhanced" in st.session_state:
        st.success("Enhanced audio is ready.")
        st.audio(st.session_state["enhanced"], format="audio/mp3")
        st.download_button(
            "Download enhanced audio",
            st.session_state["enhanced"],
            file_name="enhanced_lecture.mp3",
            mime="audio/mpeg",
        )

# ------------------------------------------------------------------ Make notes
with tab_notes:
    minutes = st.slider("How many minutes to process?", 1, 90, 3)
    start = st.number_input(
        "Start from minute", min_value=0, max_value=300, value=0,
        help="The first minutes of a class are often noise or roll-call. Start where the teaching begins.",
    )
    choice = st.selectbox("Transcript quality", list(MODELS.keys()), index=0)
    st.caption("Best quality is slower and warms up your laptop. Quick draft is fast but makes more mistakes.")
    notes_lang = st.radio(
        "Notes language", ["Hinglish (Hindi in English letters)", "English"], horizontal=True
    )
    style = (
        "simple Hinglish (Hindi written in English letters)"
        if notes_lang.startswith("Hinglish") else "simple English"
    )
    also_enhance = st.checkbox(
        "Also make the enhanced audio for this part (one click, both results)", value=True
    )

    if uploaded and st.button("Make my notes", type="primary"):
        model_name = MODELS[choice]
        use_hinglish = model_name == HINGLISH_MODEL

        with tempfile.TemporaryDirectory() as tmp:
            src = save_upload(uploaded, tmp)
            clean = os.path.join(tmp, "clean.wav")

            with st.status("Working... this can take several minutes on a laptop CPU.", expanded=True) as status:
                try:
                    if also_enhance:
                        status.write("Making the enhanced audio for listening...")
                        out_mp3 = os.path.join(tmp, "enhanced_part.mp3")
                        enhance_audio_file(src, out_mp3, True, int(start), int(minutes))
                        with open(out_mp3, "rb") as f:
                            st.session_state["notes_enhanced"] = f.read()
                    else:
                        st.session_state.pop("notes_enhanced", None)

                    status.write("Step 1/3: Cleaning and boosting the audio...")
                    clean_audio(src, clean, minutes, int(start))

                    status.write("Step 2/3: Converting speech to text (first run downloads the model)...")
                    segments, info = get_whisper(model_name).transcribe(
                        load_wav(clean),
                        language="en" if use_hinglish else "hi",
                        beam_size=5,
                        vad_filter=True,                   # skip silence
                        condition_on_previous_text=False,  # stops repeated-word loops
                    )
                    transcript = " ".join(seg.text.strip() for seg in segments)

                    status.write("Step 3/3: Writing the summary with Gemma...")
                    notes = summarize(transcript, status, style)
                    status.update(label="Done!", state="complete")
                    st.session_state["notes"] = notes
                    st.session_state["transcript"] = transcript
                    st.session_state["short"] = len(transcript) < 250 * minutes
                except Exception as e:
                    status.update(label="Something went wrong", state="error")
                    st.error(f"Error: {e}")

    # Results are kept in session_state so they stay on screen after any click
    if "notes" in st.session_state:
        st.subheader("Your notes")
        with st.container(border=True):
            st.markdown(st.session_state["notes"])
        if "notes_enhanced" in st.session_state:
            st.write("Enhanced audio for this part")
            st.audio(st.session_state["notes_enhanced"], format="audio/mp3")
            st.download_button(
                "Download this enhanced audio",
                st.session_state["notes_enhanced"],
                file_name="enhanced_part.mp3",
                mime="audio/mpeg",
                key="dl_notes_audio",
            )
        if st.session_state.get("short"):
            st.warning(
                "Very little speech was detected in this part. Try a different "
                "'Start from minute', where the teacher is clearly talking."
            )
        st.caption(
            "These notes are made by AI and can contain mistakes. "
            "Check important points against the original audio."
        )
        st.download_button("Download notes", st.session_state["notes"], file_name="notes.md")
        st.download_button(
            "Download transcript", st.session_state["transcript"], file_name="transcript.txt"
        )
        with st.expander("Show full transcript"):
            st.write(st.session_state["transcript"])
