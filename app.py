

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

# Global variable for autoplay control (initialized before UI creation)
auto_audio = True


def transcribe_audio(audio_input):
    """Convert audio to text (STT only)"""
    if audio_input is None:
        return ""
    
    sample_rate, audio_array = audio_input
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_audio.name, sample_rate, audio_array)
    
    user_text = stt_inference("Faster-Whisper - Base", temp_audio.name)
    os.remove(temp_audio.name)
    
    return user_text if user_text else "❌ Could not transcribe audio"


def process_query(text_input, auto_voice_response=True):
    """Process text with LLM and TTS"""
    
    global auto_audio
    auto_audio = auto_voice_response

    if not text_input or not text_input.strip():
        return "⚠️ Please provide text input", None
    
    user_text = text_input.strip()
    
    # LLM response
    llm_response = chat_with_llm(user_text, llm_model)
    if not llm_response or llm_response.startswith("❌"):
        return llm_response, None
    
    # TTS conversion
    output_filename = f"response_{int(time.time())}"
    audio_path = tts_converter_with_fallback(output_filename, llm_response)
    
    return llm_response, audio_path


# Gradio Blocks UI
with gr.Blocks(title="🎙️ Voice AI Chatbot") as demo:
    gr.Markdown("# 🎙️ Voice AI Chatbot")
    gr.Markdown("Record your voice or type text, then ask AI anything!")
    
    with gr.Row():
        with gr.Column():
            
            # Text input (auto-filled from audio)
            text_input = gr.Textbox(
                label="📝 Type your question",
                placeholder="Type your messageTranscription will appear here, or type directly...",
                lines=3
            )

            auto_voice_reply = gr.Checkbox(
                label="🔊 Auto-play voice response",
                value=True
            )
            
            # Collapsible Audio Recording section
            with gr.Accordion("🎤 Leave your message as voice?", open=False):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Record your voice message"
                )

            # Submit button
            submit_btn = gr.Button("Ask AI 🚀", variant="primary")
        
        with gr.Column():
            # Response text
            response_text = gr.Textbox(
                label="💬 AI Response",
                lines=8,
                interactive=False
            )
            
            # Response audio
            with gr.Accordion("🎤 Voice response", open=False):
                response_audio = gr.Audio(
                    label="🔊 Listen to response",
                    type="filepath",
                    autoplay=auto_audio
                )
    
    # Event: Audio → STT → Fill textbox
    audio_input.change(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=text_input
    )
    
    # Event: Submit → LLM+TTS → Response
    submit_btn.click(
        fn=process_query,
        inputs=[text_input, auto_voice_reply],
        outputs=[response_text, response_audio]
    )
demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)