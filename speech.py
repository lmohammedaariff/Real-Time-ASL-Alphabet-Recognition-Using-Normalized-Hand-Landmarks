"""Optional offline text-to-speech; importing this module does not require audio output."""
import logging

LOGGER = logging.getLogger(__name__)

def speak(text: str) -> tuple[bool, str]:
    """Speak text using pyttsx3 when available; return status and message."""
    text = text.strip()
    if not text:
        return False, "There is no text to speak."
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True, "Speech completed."
    except Exception as exc:
        LOGGER.exception("Offline speech failed")
        return False, f"Offline speech is unavailable: {exc}"
