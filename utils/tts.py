import pyttsx3
from gtts import gTTS
import os
import time

def generate_tts(text, lang):
    if not text:
        return None

    timestamp = int(time.time())
    filename = f"speech_{timestamp}.mp3"
    filepath = os.path.join("audio", filename)

    os.makedirs("audio", exist_ok=True)

    if lang == "en":
        engine = pyttsx3.init()
        engine.save_to_file(text, filepath)
        engine.runAndWait()
    else:
        tts = gTTS(text=text, lang=lang)
        tts.save(filepath)

    return filepath
