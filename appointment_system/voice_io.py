import sounddevice as sd
from deepgram import ( # This is for STT Handler if it were Deepgram, but STT is Whisper now.
    DeepgramClient,    # So, these might become unused if STT also changes later.
    DeepgramClientOptions
)
# New imports for gTTS
from gtts import gTTS
import playsound
import tempfile

# Standard imports
import asyncio # May become unused by TextToSpeechHandler
import threading
import os
import queue
import numpy as np
import whisper

# Configuration constants
SAMPLE_RATE = 16000 # Whisper prefers 16kHz
# SD_CHUNK_SIZE = int(SAMPLE_RATE / 10) # May not be needed for Whisper STT

class SpeechToTextHandler: # Whisper STT (Preserved from previous step)
    def __init__(self, model_name: str = "small.en", language: str = "en"):
        print(f"Whisper STT: Initializing with model '{model_name}', language '{language}'.")
        self.model_name = model_name
        self.language = language
        self.model = None
        self.final_transcript: str = ""
        self.interim_transcript: str = ""

        try:
            print(f"Whisper STT: Loading model '{self.model_name}'...")
            self.model = whisper.load_model(self.model_name)
            print(f"Whisper STT: Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Whisper STT CRITICAL Error: Failed to load model '{self.model_name}'.")
            print(f"Ensure model name is valid, you have internet for first download, or model is in cache (e.g., ~/.cache/whisper).")
            print(f"Error details: {e}")

    def listen_and_transcribe(self) -> str:
        if not self.model:
            print("Whisper STT Error: Model not loaded. Cannot transcribe.")
            return "ERROR_WHISPER_MODEL_NOT_LOADED"

        self.final_transcript = ""

        duration = 7
        channels = 1
        dtype = 'float32'

        try:
            print(f"Whisper STT: Press Enter to start recording for up to {duration} seconds...")
            input()
            print("Whisper STT: Recording...")

            myrecording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=channels, dtype=dtype)
            sd.wait()
            print("Whisper STT: Recording complete, transcribing...")

            audio_data = myrecording
            if audio_data.ndim > 1 and audio_data.shape[1] == 1:
                audio_data = audio_data.flatten()
            elif audio_data.ndim > 1 and audio_data.shape[1] > 1:
                print("Whisper STT Warning: Stereo audio detected, converting to mono by averaging channels.")
                audio_data = np.mean(audio_data, axis=1)

            result = self.model.transcribe(audio_data, language=self.language if self.language != "auto" else None, fp16=False)

            self.final_transcript = result.get("text", "").strip()

            if not self.final_transcript:
                print("Whisper STT: No speech detected or transcribed.")
                return ""

            print(f"Whisper STT: Transcription: '{self.final_transcript}'")

        except sd.PortAudioError as pae:
            print(f"Whisper STT Error: Microphone/audio device issue: {pae}")
            return "ERROR_AUDIO_DEVICE"
        except NameError as ne:
             print(f"Whisper STT Error: NameError - {ne}. Check imports.")
             return "ERROR_WHISPER_IMPORT_ISSUE"
        except Exception as e:
            print(f"Whisper STT Error: Transcription failed: {e}")
            return "ERROR_WHISPER_TRANSCRIPTION_FAILED"

        return self.final_transcript

# --- TextToSpeechHandler class (gTTS) ---
class TextToSpeechHandler:
    def __init__(self, lang: str = "en"):
        self.lang = lang
        # The DeepgramClient is no longer passed to or used by this handler.
        print(f"gTTS Handler initialized for language: {self.lang}")

    def speak(self, text_to_speak: str): # text_to_speak should be str
        if not text_to_speak:
            print("TTS (gTTS): No text to speak.")
            return

        print(f"TTS Speaking (gTTS): {text_to_speak[:60]}{'...' if len(text_to_speak) > 60 else ''}")

        temp_audio_file = None
        try:
            tts_obj = gTTS(text=text_to_speak, lang=self.lang, slow=False)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                temp_audio_file = fp.name

            tts_obj.save(temp_audio_file)

            playsound.playsound(temp_audio_file)

        except ImportError:
            print("gTTS Error: gTTS or playsound library not found. Please ensure they are installed.")
        except Exception as e:
            print(f"gTTS Error in speak(): {e}")
        finally:
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                except Exception as e_del:
                    print(f"TTS (gTTS) Error: Failed to delete temporary file {temp_audio_file}: {e_del}")

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv() # Load .env file for any optional configurations

    print("--- Voice I/O Module Test (Whisper STT & gTTS) ---")

    # Ensure os is imported if using os.environ.get (it's imported at module level)

    # Test TextToSpeechHandler (gTTS)
    print("\n--- Testing TextToSpeechHandler (gTTS) ---")
    try:
        # Optional: allow language to be set via env var, e.g., TTS_LANG
        tts_lang = os.environ.get("TTS_LANG", "en")
        tts_handler = TextToSpeechHandler(lang=tts_lang)

        print(f"Attempting to speak a short phrase with gTTS (lang={tts_lang})...")
        tts_handler.speak("Hello, this is a test of Google Text to Speech using the gTTS library.")

        print("Attempting to speak a slightly longer phrase with gTTS...")
        tts_handler.speak("This audio is generated by gTTS, saved to a temporary MP3 file, and then played.")

    except Exception as e:
        print(f"Error during gTTS test: {e}")
    print("--- Finished gTTS Test ---")

    # Test SpeechToTextHandler (Whisper STT)
    print("\n--- Testing SpeechToTextHandler (Whisper STT) ---")
    try:
        # Allow Whisper model name to be set via environment variable, default to "small.en"
        whisper_model_name = os.environ.get("WHISPER_MODEL_NAME", "small.en")
        # Optional: allow language to be set, default "en" or "auto" for Whisper
        whisper_lang = os.environ.get("WHISPER_LANGUAGE", "en")

        print(f"Initializing Whisper STT with model: '{whisper_model_name}', language: '{whisper_lang}'")
        stt_handler = SpeechToTextHandler(model_name=whisper_model_name, language=whisper_lang)

        if stt_handler.model is None:
            print("Whisper STT model failed to load. Skipping STT test. Check error messages above during init.")
        else:
            for i in range(2): # Allow a couple of attempts
                print(f"\nSTT Attempt {i+1}/2 (Whisper). Press Enter to start recording, then speak. (Ctrl+C to skip)")
                try:
                    input()
                    print("Recording for Whisper STT...")
                    text = stt_handler.listen_and_transcribe()
                    if text and not text.startswith("ERROR_"):
                        print(f"--- You said (Whisper): {text} ---")
                    elif text.startswith("ERROR_"):
                        print(f"--- Whisper STT Error: {text} ---")
                    else: # Empty string result
                        print(f"--- No speech detected by Whisper or empty transcription. ---")
                except KeyboardInterrupt:
                    print("\nSkipped STT attempt by user.")
                    break # Exit loop on Ctrl+C
    except Exception as e:
        print(f"Error during Whisper STT test setup or execution: {e}")
        # import traceback
        # traceback.print_exc()
    print("--- Finished Whisper STT Test ---")

    print("\n--- Voice I/O Module Test Complete ---")
