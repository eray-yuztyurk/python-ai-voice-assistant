
from config.utils import EdgeTTS_DEFAULT_VOICES, PIPER_DEFAULT_VOICES, detect_text_language

def tts_converter_with_fallback(file_output_name, text_input):
    
    # detect language code
    lang_code = detect_text_language(text_input)
    
    try:
        import edge_tts
        import asyncio
        
        async def edge_tts_get():
            tts = edge_tts.Communicate(
                text_input, 
                EdgeTTS_DEFAULT_VOICES.get(lang_code, "en-US-AriaNeural"),
                rate="+12%",  
                volume="+5%", 
                pitch="+0Hz" 
            )
            await tts.save(f"./data/audio/output_audio/{file_output_name}.mp3")
        
        asyncio.run(edge_tts_get())

        return f"./data/audio/output_audio/{file_output_name}.mp3"
    
    except:
        pass

    try:
        from gtts import gTTS
        tts = gTTS(text=text_input, lang=lang_code, slow=False)
        tts.save(f"./data/audio/output_audio/{file_output_name}.mp3")

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
