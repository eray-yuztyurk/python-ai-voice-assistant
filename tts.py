
from config.utils import EdgeTTS_DEFAULT_VOICES, PIPER_DEFAULT_VOICES, detect_text_language

def tts_converter_with_fallback(file_output_name, text_input):
    
    # detect language code
    lang_code = detect_text_language(text_input)
    
    try:
        from gtts import gTTS
        tts = gTTS(text=text_input, lang=lang_code)
        tts.save(f"./data/audio/output_audio/{file_output_name}.mp3")

        return f"./data/audio/output_audio/{file_output_name}.mp3"
    
    except:
        pass

    try:
        import edge_tts
        import asyncio
        
        async def edge_tts_get():
            tts = edge_tts.Communicate(text_input, EdgeTTS_DEFAULT_VOICES.get(lang_code, "en-US-AriaNeural"))
            await tts.save(f"./data/audio/output_audio/{file_output_name}.mp3")
        
        asyncio.run(edge_tts_get())

        return f"./data/audio/output_audio/{file_output_name}.mp3"
    
    except:
        pass
    
    try:

        import subprocess
        subprocess.run([
            "piper",
            "--model", PIPER_DEFAULT_VOICES.get(lang_code, "en_US-amy-medium"),
            "--output_file", f"./data/audio/output_audio/{file_output_name}.wav",
        ],
        input=text_input.encode('utf-8'),
        check=True)

        return f"./data/audio/output_audio/{file_output_name}.wav"
    
    except Exception as e:
        print(f"Error during Piper TTS inference: {e}")
        pass

"""
#######################################################################################################
import gradio as gr

# Gradio UI (delete later when frontend is prepared)
_ui = gr.Interface(
    fn=tts_converter_with_fallback,
    inputs=[
        gr.Textbox(label="Please provide a file name..."),
        gr.Textbox(label="Please paste your text here...", lines=10, max_lines=40)
    ],
    outputs=[
        #gr.Textbox(label=f"Status", lines=2),
        gr.Audio(label="Converted Speech", type="filepath")
    ],
    title=""
        <div style='text-align:center; font-size:28px; font-weight:600;'>
        🎧 Speech from Text Generator
        </div>
        "",
    description=""
        <p>This application converts the provided text into speech audio using pre-trained TTS models from Hugging Face Transformers.</p>
        <p>Select the language, provide a file name, and paste your text to generate the audio.</p>
        ""
)

_ui.launch(server_name="0.0.0.0", server_port=7861, inbrowser=True)
"""