import langid

# detect the language of the given text
def detect_text_language(text):
    # Try fasttext first (better accuracy), fallback to langid
    try:
        import fasttext
        model = fasttext.load_model('lid.176.ftz')
        lang_fasttext, prob_fasttext = model.predict(text)
        
        # fallback to langid if fasttext confidence is low
        if prob_fasttext[0] < 0.7:
            lang_code, confidence = langid.classify(text)
            return lang_code
        else:
            return lang_fasttext[0].replace("__label__", "")
    except:
        # fasttext not available or error, use langid
        lang_code, confidence = langid.classify(text)
        return lang_code
   

EdgeTTS_DEFAULT_VOICES = {
    "en": "en-US-AriaNeural",
    "tr": "tr-TR-EmelNeural",
    "de": "de-DE-KatjaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
}

PIPER_DEFAULT_VOICES = {
    "en": "en_US-amy-medium",
    "tr": "tr_TR-fettah-medium",
    "de": "de_DE-thorsten-medium",
    "fr": "fr_FR-siwis-medium",
    "es": "es_ES-davefx-medium",
}
