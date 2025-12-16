

import gradio as gr
import os
import tempfile
import time
import scipy.io.wavfile as wav

from stt import stt_inference
from llm import chat_with_llm, llm_initializer_with_fallback
from tts import tts_converter_with_fallback

print("🔄 Initializing LLM...")
llm_model = llm_initializer_with_fallback()

def voice_chatbot(audio_input, text_input):
    if audio_input is not None:
        sample_rate, audio_array = audio_input
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(temp_audio.name, sample_rate, audio_array)
        user_text = stt_inference("Faster-Whisper - Base", temp_audio.name)
        os.remove(temp_audio.name)
        if not user_text:
            return "❌ Could not transcribe audio", None
    elif text_input and text_input.strip():
        user_text = text_input.strip()
    else:
        return "⚠️ Please provide either audio or text input", None
    
    llm_response = chat_with_llm(user_text, llm_model)
    if not llm_response or llm_response.startswith("❌"):
        return llm_response, None
    
    output_filename = f"response_{int(time.time())}"
    audio_path = tts_converter_with_fallback(output_filename, llm_response)
    
    return llm_response, audio_path

demo = gr.Interface(
    fn=voice_chatbot,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy", label="🎤 Speak"),
        gr.Textbox(label="✍️ Or type", placeholder="Ask anything...", lines=3)
    ],
    outputs=[
        gr.Textbox(label="💬 Response", lines=5),
        gr.Audio(label="🔊 Audio", type="filepath")
    ],
    title="🎙️ Voice AI Chatbot",
    description="Ask using voice or text, get responses in both formats!",
    allow_flagging="never"
)

demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)


