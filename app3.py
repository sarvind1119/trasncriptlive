import sys
import types
import warnings
import os
import streamlit as st
import tempfile
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import google.generativeai as genai
from dotenv import load_dotenv
import io
import wave
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av

# Patch PyTorch class bug (if Whisper is ever re-added)
import torch
torch.classes = types.SimpleNamespace()
sys.modules["torch.classes"] = torch.classes
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load API key
env_loaded = load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Transcription function
def transcribe(audio_file, format_type):
    try:
        your_file = genai.upload_file(path=audio_file)
        prompt = f"Act as a speech recognizer expert. Listen carefully to the following audio file. Provide a complete transcript in {format_type} format."
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content([prompt, your_file])
        return response.text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# Translation function
def translate(audio_file, format_type):
    try:
        your_file = genai.upload_file(path=audio_file)
        prompt = f"Listen carefully to the following audio file. Translate it to English in {format_type} format."
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content([prompt, your_file])
        return response.text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return None

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral" if subjectivity < 0.3 else "Mixed"

# Save results

def save_results_to_excel(data):
    today_date = datetime.now().strftime("%m_%d_%y")
    results_filename = f"Results_of_{today_date}.xlsx"
    if os.path.exists(results_filename):
        existing_data = pd.read_excel(results_filename)
        new_data = pd.concat([existing_data, pd.DataFrame(data)], ignore_index=True)
    else:
        new_data = pd.DataFrame(data)
    new_data.to_excel(results_filename, index=False)
    return results_filename

def save_results_to_text(data):
    today_date = datetime.now().strftime("%m_%d_%y")
    results_filename = f"Results_of_{today_date}.txt"
    with open(results_filename, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(f"Lecture Title: {entry.get('Lecture Title', '')}\n")
            file.write(f"Audio File Name: {entry['Audio File Name']}\n")
            file.write(f"Transcript/Translation: {entry['Transcript/Translation']}\n")
            file.write(f"Format Chosen: {entry['Format Chosen']}\n")
            file.write(f"Sentiment: {entry['Sentiment']}\n")
            file.write("\n" + "-"*50 + "\n\n")
    return results_filename

# Audio recording class for webrtc
class AudioRecorder:
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return av.AudioFrame.from_ndarray(audio, layout="mono")

# Main app
def main():
    st.markdown("""
    <style>
        .stButton>button {
            background-color: #0c4b9c;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h1 style='text-align: center; font-size: 2.5em;'>🎧 Audio Transcription & Translation</h1>
    <p style='text-align: center; color: gray;'>Upload or record audio. Get transcripts instantly with sentiment insights.</p>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📤 Upload Audio Files", "🎙️ Record Audio Live"])

    with tabs[0]:
        audio_files = st.file_uploader("Upload audio files", accept_multiple_files=True, type=["wav", "mp3", "flac"])
        if audio_files:
            lecture_title = st.text_input("Lecture Title")
            format_options = ["Conversation style, accurately identify speakers", "Paragraph", "Bullet points", "Summary"]
            selected_format = st.selectbox("Choose output format", format_options)
            option = st.selectbox("Choose an option", ["Transcribe", "Translate"])
            if st.button("Process Uploaded Files"):
                results_data = []
                progress_bar = st.progress(0)
                for idx, audio_file in enumerate(audio_files):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file.write(audio_file.read())
                        audio_path = temp_file.name
                    if option == "Transcribe":
                        result_text = transcribe(audio_path, selected_format)
                    else:
                        result_text = translate(audio_path, selected_format)
                    sentiment = analyze_sentiment(result_text)
                    results_data.append({
                        "Lecture Title": lecture_title,
                        "Audio File Name": audio_file.name,
                        "Transcript/Translation": result_text,
                        "Format Chosen": selected_format,
                        "Sentiment": sentiment
                    })
                    progress_bar.progress(int(((idx + 1) / len(audio_files)) * 100))
                excel_path = save_results_to_excel(results_data)
                text_path = save_results_to_text(results_data)
                with open(excel_path, "rb") as f:
                    st.download_button("⬇️ Download Excel", f, file_name=excel_path)
                with open(text_path, "rb") as f:
                    st.download_button("⬇️ Download Text File", f, file_name=text_path)

    with tabs[1]:
        st.subheader("🎙️ Record Audio Now")
        lecture_title = st.text_input("Lecture Title (for recorded audio)")

        recorder = AudioRecorder()
        ctx = webrtc_streamer(
            key="audio",
            mode=WebRtcMode.SENDRECV,
            audio_receiver_size=1024,
            client_settings=ClientSettings(media_stream_constraints={"audio": True, "video": False}),
            rtc_configuration={},
            audio_frame_callback=recorder.recv,
        )

        if ctx.state.playing:
            st.write("🎙️ Recording... Speak now.")

        elif recorder.frames:
            st.success("✅ Recording complete.")
            filename = f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

            with wave.open(audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                for frame in recorder.frames:
                    wf.writeframes(frame.tobytes())

            with open(audio_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/wav")

            selected_format = st.selectbox("Choose output format for recorded audio", ["Conversation style, accurately identify speakers", "Paragraph", "Bullet points", "Summary"], key="record_format")
            option = st.selectbox("Choose an option", ["Transcribe", "Translate"], key="record_option")

            if st.button("Transcribe/Translate Recorded Audio"):
                if option == "Transcribe":
                    result_text = transcribe(audio_path, selected_format)
                else:
                    result_text = translate(audio_path, selected_format)
                sentiment = analyze_sentiment(result_text)
                st.text_area("📝 Transcript Output", result_text, height=300)
                st.markdown(f"**Sentiment:** {sentiment}")
                result_data = [{
                    "Lecture Title": lecture_title,
                    "Audio File Name": f"recorded_audio.wav",
                    "Transcript/Translation": result_text,
                    "Format Chosen": selected_format,
                    "Sentiment": sentiment
                }]
                excel_path = save_results_to_excel(result_data)
                text_path = save_results_to_text(result_data)
                col1, col2 = st.columns(2)
                with col1:
                    with open(excel_path, "rb") as f:
                        st.download_button("⬇️ Download Excel", f, file_name=excel_path)
                with col2:
                    with open(text_path, "rb") as f:
                        st.download_button("⬇️ Download Text File", f, file_name=text_path)

if __name__ == "__main__":
    main()
