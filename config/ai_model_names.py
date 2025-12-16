"""
AI Model Names Configuration
Organized by library/framework for STT and TTS models
"""

# ================================================================================================
# SPEECH-TO-TEXT (STT) MODELS
# ================================================================================================

# Faster-Whisper Models (faster-whisper library - CTranslate2 optimized)
# RAM: ~200-600MB | Speed: Very Fast | Quality: High
faster_whisper_stt_models = {
    "Faster-Whisper - Tiny": "tiny",
    "Faster-Whisper - Base": "base",
    "Faster-Whisper - Small": "small",
    "Faster-Whisper - Medium": "medium",
    "Faster-Whisper - Large-V2": "large-v2"
}

# Transformers Whisper Models (transformers library - PyTorch)
# RAM: ~1-6GB | Speed: Medium | Quality: High
transformers_whisper_stt_models = {
    # Official OpenAI Whisper
    "Whisper - Tiny": "openai/whisper-tiny",
    "Whisper - Base": "openai/whisper-base",
    "Whisper - Small": "openai/whisper-small",
    "Whisper - Medium": "openai/whisper-medium",
    "Whisper - Large V1": "openai/whisper-large",
    "Whisper - Large V2": "openai/whisper-large-v2",
    "Whisper - Large V3": "openai/whisper-large-v3",

    # Distil-Whisper (optimized, English-only)
    "Distil-Whisper - Tiny (EN)": "distil-whisper/distil-tiny.en",
    "Distil-Whisper - Base (EN)": "distil-whisper/distil-base.en",
    "Distil-Whisper - Small (EN)": "distil-whisper/distil-small.en",
    "Distil-Whisper - Medium (EN)": "distil-whisper/distil-medium.en",
    "Distil-Whisper - Medium (Multilingual)": "distil-whisper/distil-medium",

    # Specialized Models
    "Whisper - Large-V2 (Noise-Robust)": "biodatlab/whisper-noise-large-v2",
}

# Combined STT models for UI
stt_models = {**faster_whisper_stt_models, **transformers_whisper_stt_models}


# ================================================================================================
# TEXT-TO-SPEECH (TTS) MODELS
# ================================================================================================

# Transformers-based TTS Models (transformers library - PyTorch)
transformers_tts_models = {
    "SpeechT5 - Multilingual [High | Local]": {
        "id": "microsoft/speecht5_tts",
        "lang": "multi",
        "lib": "transformers",
        "ram_mb": 1500,
    },
    "MMS-TTS - English [Good | Local]": {
        "id": "facebook/mms-tts-eng",
        "lang": "en",
        "lib": "transformers",
        "ram_mb": 800,
    },
    "MMS-TTS - Turkish [Good | Local]": {
        "id": "facebook/mms-tts-tur",
        "lang": "tr",
        "lib": "transformers",
        "ram_mb": 800,
    },
    "MMS-TTS - German [Good | Local]": {
        "id": "facebook/mms-tts-deu",
        "lang": "de",
        "lib": "transformers",
        "ram_mb": 800,
    },
    "MMS-TTS - French [Good | Local]": {
        "id": "facebook/mms-tts-fra",
        "lang": "fr",
        "lib": "transformers",
        "ram_mb": 800,
    },
    "MMS-TTS - Spanish [Good | Local]": {
        "id": "facebook/mms-tts-spa",
        "lang": "es",
        "lib": "transformers",
        "ram_mb": 800,
    },
    "Parler-TTS - Mini [High | Local]": {
        "id": "parler-tts/parler-tts-mini-v1",
        "lang": "multi",
        "lib": "transformers",
        "ram_mb": 1200,
    },
    "Parler-TTS - Large [Excellent | Local]": {
        "id": "parler-tts/parler-tts-large-v1",
        "lang": "multi",
        "lib": "transformers",
        "ram_mb": 2500,
    },
    "Bark - Small [Excellent | Local]": {
        "id": "suno/bark-small",
        "lang": "multi",
        "lib": "transformers",
        "ram_mb": 3000,
    },
    "Bark - Large [Excellent | Local]": {
        "id": "suno/bark",
        "lang": "multi",
        "lib": "transformers",
        "ram_mb": 5000,
    },
}

# Piper TTS Models (piper-tts library - ONNX optimized)
piper_tts_models = {
    # English voices - Female
    "Piper - English (Amy, Female) [Fast | Local]": {
        "id": "en_US-amy-low",
        "lang": "en",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 150,
    },
    "Piper - English (Amy, Female) [Good | Local]": {
        "id": "en_US-amy-medium",
        "lang": "en",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 200,
    },
    "Piper - English (Libritts, Female) [High | Local]": {
        "id": "en_US-libritts-high",
        "lang": "en",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 300,
    },
    "Piper - English (Kristin, Female) [Good | Local]": {
        "id": "en_US-kristin-medium",
        "lang": "en",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 200,
    },
    "Piper - English (Kath, Female) [Good | Local]": {
        "id": "en_US-kath-medium",
        "lang": "en",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 200,
    },
    
    # English voices - Male
    "Piper - English (Lessac, Male) [Good | Local]": {
        "id": "en_US-lessac-medium",
        "lang": "en",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 200,
    },

    "Piper - English (Ryan, Male) [High | Local]": {
        "id": "en_US-ryan-high",
        "lang": "en",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 300,
    },
    "Piper - English (Danny, Male) [Fast | Local]": {
        "id": "en_US-danny-low",
        "lang": "en",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 150,
    },
    
    # Turkish voices - Male
    "Piper - Turkish (Fettah, Male) [Good | Local]": {
        "id": "tr_TR-fettah-medium",
        "lang": "tr",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 200,
    },
    "Piper - Turkish (Dfki, Male) [Good | Local]": {
        "id": "tr_TR-dfki-medium",
        "lang": "tr",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 200,
    },
    
    # German voices - Male
    "Piper - German (Thorsten, Male) [Good | Local]": {
        "id": "de_DE-thorsten-medium",
        "lang": "de",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 200,
    },
    "Piper - German (Thorsten, Male) [Fast | Local]": {
        "id": "de_DE-thorsten-low",
        "lang": "de",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 150,
    },
    
    # German voices - Female
    "Piper - German (Pavoque, Female) [Fast | Local]": {
        "id": "de_DE-pavoque-low",
        "lang": "de",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 150,
    },
    "Piper - German (Kerstin, Female) [Good | Local]": {
        "id": "de_DE-kerstin-low",
        "lang": "de",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 150,
    },
    
    # Spanish voices - Male
    "Piper - Spanish (Davefx, Male) [Good | Local]": {
        "id": "es_ES-davefx-medium",
        "lang": "es",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 200,
    },
    
    # Spanish voices - Female
    "Piper - Spanish (Carlfm, Female) [Fast | Local]": {
        "id": "es_ES-carlfm-x_low",
        "lang": "es",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 150,
    },
    
    # French voices - Male
    "Piper - French (Siwis, Male) [Good | Local]": {
        "id": "fr_FR-siwis-medium",
        "lang": "fr",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 200,
    },
    "Piper - French (Tom, Male) [Fast | Local]": {
        "id": "fr_FR-tom-medium",
        "lang": "fr",
        "lib": "piper",
        "gender": "m",
        "ram_mb": 150,
    },
    
    # French voices - Female
    "Piper - French (Upmc, Female) [Fast | Local]": {
        "id": "fr_FR-upmc-medium",
        "lang": "fr",
        "lib": "piper",
        "gender": "f",
        "ram_mb": 150,
    },
}

# VITS Models (transformers or vits library)
# RAM: ~200-500MB | Speed: Fast | Quality: High
vits_tts_models = {
    "VITS - LJ Speech (EN) [High | Local]": {
        "id": "facebook/mms-tts-eng",
        "lang": "en",
        "lib": "vits",
        "gender": "m",
        "ram_mb": 300,
    },
    "VITS - Turkish [High | Local]": {
        "id": "facebook/mms-tts-tur",
        "lang": "tr",
        "lib": "vits",
        "gender": "m",
        "ram_mb": 300,
    },
}

# Coqui TTS Models (TTS library - advanced features)
# RAM: ~2-6GB | Speed: Slow | Quality: Excellent (voice cloning)
coqui_tts_models = {
    "XTTS-v2 - Multilingual [Excellent | Local]": {
        "id": "tts_models/multilingual/multi-dataset/xtts_v2",
        "lang": "multi",
        "lib": "coqui",
        "ram_mb": 4000,
    },
    "XTTS-v2 - Voice Clone [Excellent | Local]": {
        "id": "tts_models/multilingual/multi-dataset/xtts_v2",
        "lang": "multi",
        "lib": "coqui",
        "ram_mb": 4000,
    },
}

# API-based TTS (requires internet, free tiers available)
# RAM: ~50MB | Speed: Depends on network | Quality: High
api_tts_models = {
    # Google Text-to-Speech (gTTS library)
    "gTTS - English [Good | API]": {
        "id": "en",
        "lang": "en",
        "lib": "gtts",
        "ram_mb": 50,
    },
    "gTTS - Turkish [Good | API]": {
        "id": "tr",
        "lang": "tr",
        "lib": "gtts",
        "ram_mb": 50,
    },
    "gTTS - German [Good | API]": {
        "id": "de",
        "lang": "de",
        "lib": "gtts",
        "ram_mb": 50,
    },
    "gTTS - French [Good | API]": {
        "id": "fr",
        "lang": "fr",
        "lib": "gtts",
        "ram_mb": 50,
    },
    "gTTS - Spanish [Good | API]": {
        "id": "es",
        "lang": "es",
        "lib": "gtts",
        "ram_mb": 50,
    },
    # Microsoft Edge TTS (edge-tts library)
    # English voices
    "Edge-TTS - English (Aria, Female) [High | API]": {
        "id": "en-US-AriaNeural",
        "lang": "en",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - English (Jenny, Female) [High | API]": {
        "id": "en-US-JennyNeural",
        "lang": "en",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - English (Michelle, Female) [High | API]": {
        "id": "en-US-MichelleNeural",
        "lang": "en",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - English (Guy, Male) [High | API]": {
        "id": "en-US-GuyNeural",
        "lang": "en",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
    "Edge-TTS - English (Eric, Male) [High | API]": {
        "id": "en-US-EricNeural",
        "lang": "en",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
    "Edge-TTS - English (Christopher, Male) [High | API]": {
        "id": "en-US-ChristopherNeural",
        "lang": "en",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
    # Turkish voices
    "Edge-TTS - Turkish (Emel, Female) [High | API]": {
        "id": "tr-TR-EmelNeural",
        "lang": "tr",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - Turkish (Ahmet, Male) [High | API]": {
        "id": "tr-TR-AhmetNeural",
        "lang": "tr",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
    # German voices
    "Edge-TTS - German (Katja, Female) [High | API]": {
        "id": "de-DE-KatjaNeural",
        "lang": "de",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - German (Amala, Female) [High | API]": {
        "id": "de-DE-AmalaNeural",
        "lang": "de",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - German (Conrad, Male) [High | API]": {
        "id": "de-DE-ConradNeural",
        "lang": "de",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
    "Edge-TTS - German (Klaus, Male) [High | API]": {
        "id": "de-DE-KlausNeural",
        "lang": "de",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
    # Spanish voices
    "Edge-TTS - Spanish (Elvira, Female) [High | API]": {
        "id": "es-ES-ElviraNeural",
        "lang": "es",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - Spanish (Abril, Female) [High | API]": {
        "id": "es-ES-AbrilNeural",
        "lang": "es",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - Spanish (Alvaro, Male) [High | API]": {
        "id": "es-ES-AlvaroNeural",
        "lang": "es",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
    # French voices
    "Edge-TTS - French (Denise, Female) [High | API]": {
        "id": "fr-FR-DeniseNeural",
        "lang": "fr",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - French (Brigitte, Female) [High | API]": {
        "id": "fr-FR-BrigitteNeural",
        "lang": "fr",
        "lib": "edge-tts",
        "gender": "f",
        "ram_mb": 50,
    },
    "Edge-TTS - French (Henri, Male) [High | API]": {
        "id": "fr-FR-HenriNeural",
        "lang": "fr",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
    "Edge-TTS - French (Alain, Male) [High | API]": {
        "id": "fr-FR-AlainNeural",
        "lang": "fr",
        "lib": "edge-tts",
        "gender": "m",
        "ram_mb": 50,
    },
}

# System TTS (eSpeak-NG - offline, lightweight, basic quality)
espeak_tts_models = {
    "eSpeak - English [Basic | Local]": {
        "id": "en",
        "lang": "en",
        "lib": "espeak",
        "ram_mb": 30,
    },
    "eSpeak - Turkish [Basic | Local]": {
        "id": "tr",
        "lang": "tr",
        "lib": "espeak",
        "ram_mb": 30,
    },
    "eSpeak - German [Basic | Local]": {
        "id": "de",
        "lang": "de",
        "lib": "espeak",
        "ram_mb": 30,
    },
    "eSpeak - French [Basic | Local]": {
        "id": "fr",
        "lang": "fr",
        "lib": "espeak",
        "ram_mb": 30,
    },
    "eSpeak - Spanish [Basic | Local]": {
        "id": "es",
        "lang": "es",
        "lib": "espeak",
        "ram_mb": 30,
    },
}

# Combined TTS models for UI
tts_models = {
    **transformers_tts_models,
    **piper_tts_models,
    **api_tts_models,
    **espeak_tts_models,
}