# -*- coding: utf-8 -*-
"""
Multimodal RAG (Whisper + LLaVA + gTTS) - Enhanced Version
This script integrates Whisper for speech-to-text, LLaVA for vision-language understanding,
and gTTS for text-to-speech, with improved error handling, long audio chunking, and JSON logging.
It is designed for a Gradio interface to create a multimodal assistant.
Author: Abu Hanifah
Date: 2025-08-12
Features:
    ✅ Long audio chunking for Whisper
    ✅ Retry logic for Ollama API calls
    ✅ gTTS auto language switching based on Whisper
    ✅ JSON conversation logging for RAG
    ✅ Timestamped logging
    ✅ Ready for streaming upgrade
"""

import os
import time
import json
import base64
import datetime
import warnings
import numpy as np
import requests
import torch
import whisper
import gradio as gr
from gtts import gTTS
from PIL import Image

warnings.filterwarnings("ignore")

# ==============================
# DEVICE + MODEL LOADING
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using torch {torch.__version__} ({DEVICE})")

model = whisper.load_model("medium", device=DEVICE)
print(
    f"[INFO] Whisper model: {'multilingual' if model.is_multilingual else 'English-only'} "
    f"with {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

# ==============================
# LOG FILE SETUP
# ==============================
tstamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logfile = f"{tstamp}_log.txt"

def writehistory(text: str):
    """Write a timestamped log line."""
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.datetime.now()}] {text}\n")

# ==============================
# JSON HISTORY LOGGING
# ==============================
def store_conversation(history_item, json_file="conversation_history.json"):
    """Append a JSON record for later RAG processing."""
    try:
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        history.append(history_item)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        writehistory(f"[ERROR] Failed to store JSON conversation: {e}")

# ==============================
# OLLAMA CONFIG
# ==============================
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_CONFIG = {
    "model": "llava:7b",   # Make sure you've run: ollama pull llava:7b
    "stream": False
}

def img2txt(input_text, input_image_path, max_retries=3, retry_delay=2):
    """
    Enhanced Ollama LLaVA: Adds retry and error handling.
    """
    writehistory(f"Input text: {input_text}")
    prompt_text = (
        input_text if input_text and not isinstance(input_text, tuple)
        else "Describe the image in detail."
    )
    writehistory(f"Prompt for Ollama: {prompt_text}")

    if not os.path.exists(input_image_path):
        return "Error: Image file not found."

    try:
        with open(input_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return f"Error encoding image: {e}"

    payload = {**OLLAMA_CONFIG, "prompt": prompt_text, "images": [encoded_image]}

    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=30)
            response.raise_for_status()
            reply = response.json().get("response", "No response from model.")
            return reply.strip()
        except requests.exceptions.RequestException as e:
            writehistory(f"Ollama request failed (attempt {attempt+1}): {e}")
            time.sleep(retry_delay)

    return "Error: Ollama request repeatedly failed. Check server status."

# ==============================
# WHISPER TRANSCRIPTION (LONG AUDIO CHUNKING)
# ==============================
def transcribe(audio_path, chunk_duration_sec=30):
    """
    Transcribe audio in chunks to handle long inputs (>30s).
    Returns (full_transcript, detected_language_code).
    """
    import math
    if not audio_path or not os.path.exists(audio_path):
        return ("", "en")

    audio = whisper.load_audio(audio_path)
    sample_rate = whisper.audio.SAMPLE_RATE
    total_frames = audio.shape[0]
    chunk_size = chunk_duration_sec * sample_rate
    num_chunks = math.ceil(total_frames / chunk_size)

    transcript = []
    lang_probs = {}

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, total_frames)
        chunk_audio = whisper.pad_or_trim(audio[start:end])
        mel = whisper.log_mel_spectrogram(chunk_audio).to(model.device)
        if i == 0:  # detect language only once
            _, lang_probs = model.detect_language(mel)
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        transcript.append(result.text.strip())

    detected_lang = max(lang_probs, key=lang_probs.get) if lang_probs else "en"
    return " ".join(transcript), detected_lang

# ==============================
# TEXT TO SPEECH (AUTO LANGUAGE)
# ==============================
def text_to_speech(text, file_path, lang="en"):
    """Convert text to speech with detected language."""
    if not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(file_path)
        return file_path
    except Exception as e:
        writehistory(f"[ERROR] gTTS failed: {e}")
        return None

# ==============================
# MAIN PIPELINE
# ==============================
def process_inputs(audio_path, image_path):
    """Main processing pipeline for Gradio interface."""
    speech_to_text_output, detected_lang = transcribe(audio_path)

    if image_path:
        llava_output = img2txt(speech_to_text_output, image_path)
    else:
        llava_output = "No image provided."

    audio_reply_path = text_to_speech(llava_output, "Temp3.mp3", lang=detected_lang)

    store_conversation({
        "timestamp": str(datetime.datetime.now()),
        "audio_path": audio_path,
        "image_path": image_path,
        "speech_to_text_output": speech_to_text_output,
        "llava_output": llava_output,
        "voice_lang": detected_lang,
        "audio_response_path": audio_reply_path
    })

    return speech_to_text_output, llava_output, audio_reply_path

# ==============================
# GRADIO UI
# ==============================
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak Here"),
        gr.Image(type="filepath", label="🖼 Upload Image")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Vision-Language Model Output"),
        gr.Audio(type="filepath", label="AI Voice Response")
    ],
    title="🎙 Whisper + LLaVA Multimodal Assistant (Enhanced)",
    description=(
        "Speak into the microphone and/or provide an image.\n"
        "Whisper transcribes your speech (with chunking for long audio), LLaVA "
        "analyzes the image & text, and gTTS speaks the response in the detected language."
    )
)

if __name__ == "__main__":
    iface.launch(debug=True)
