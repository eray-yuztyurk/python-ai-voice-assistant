
from src.utils.utils import EdgeTTS_DEFAULT_VOICES, PIPER_DEFAULT_VOICES, detect_text_language
from src.config.settings import AUDIO_OUTPUT_DIR, TTS_RATE, TTS_VOLUME, TTS_PITCH

def tts_converter_with_fallback(file_output_name, text_input):
    """
    Convert text to speech with automatic fallback mechanism.
    
    Tries in order: Edge-TTS → gTTS → Piper
    
    Args:
        file_output_name (str): Output filename (without extension)
        text_input (str): Text to convert to speech
        
    Returns:
        str: Path to generated audio file
    """
    lang_code = detect_text_language(text_input)

    try:
        os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    except Exception as _:
        pass
        
    try:
        import edge_tts
        import asyncio
        
        async def edge_tts_get():
            tts = edge_tts.Communicate(
                text_input, 
                EdgeTTS_DEFAULT_VOICES.get(lang_code, "en-US-AriaNeural"),
                rate=TTS_RATE,  
                volume=TTS_VOLUME, 
                pitch=TTS_PITCH 
            )
            await tts.save(f"{AUDIO_OUTPUT_DIR}/{file_output_name}.mp3")
        
        asyncio.run(edge_tts_get())

        return f"{AUDIO_OUTPUT_DIR}/{file_output_name}.mp3"
    
    except:
        pass

    try:
        from gtts import gTTS
        tts = gTTS(text=text_input, lang=lang_code, slow=False)
        tts.save(f"{AUDIO_OUTPUT_DIR}/{file_output_name}.mp3")

        return f"{AUDIO_OUTPUT_DIR}/{file_output_name}.mp3"
    
    except:
        pass

    try:

        import subprocess
        subprocess.run([
            "piper",
            "--model", PIPER_DEFAULT_VOICES.get(lang_code, "en_US-amy-medium"),
            "--output_file", f"{AUDIO_OUTPUT_DIR}/{file_output_name}.wav",
        ],
        input=text_input.encode('utf-8'),
        check=True)

        return f"{AUDIO_OUTPUT_DIR}/{file_output_name}.wav"
    
    except Exception as e:
        print(f"Error during Piper TTS inference: {e}")
        pass
