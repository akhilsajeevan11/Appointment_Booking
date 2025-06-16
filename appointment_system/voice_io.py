import sounddevice as sd
from google.cloud import speech
from google.cloud import texttospeech
import playsound
import tempfile
import os # For os.remove with NamedTemporaryFile on Windows
import queue # For thread-safe data passing from callback to generator

# Configuration constants
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10) # 100ms

class SpeechToTextHandler:
    def __init__(self, language_code="en-US"):
        self.client = speech.SpeechClient()
        self.language_code = language_code
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
            single_utterance=True
        )
        self._buffer = queue.Queue()
        self._audio_stream_active = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Sounddevice status: {status}", flush=True)
        if self._audio_stream_active:
            self._buffer.put(bytes(indata))

    def _audio_generator(self):
        while self._audio_stream_active:
            chunk = self._buffer.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

            while not self._buffer.empty():
                try:
                    chunk = self._buffer.get_nowait()
                    if chunk is None: return
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except queue.Empty:
                    break

    def listen_and_transcribe(self):
        final_transcript = ""
        displayed_interim_transcript = ""

        print("Listening...")
        self._audio_stream_active = True
        # Ensure buffer is clean for a new attempt
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break

        # Ensure the responses attribute is reset/available
        self.responses = None

        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                                   device=None, dtype='int16', channels=1,
                                   callback=self._audio_callback) as stream:

                audio_gen = self._audio_generator()
                self.responses = self.client.streaming_recognize(
                    config=self.streaming_config,
                    requests=audio_gen
                )

                for response in self.responses:
                    if not self._audio_stream_active and not response.results : # Check if stream was stopped early
                        # If buffer is also empty, it means we gracefully stopped or API ended.
                        if self._buffer.empty():
                            break
                    if not response.results:
                        continue

                    result = response.results[0]
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript

                    if result.is_final:
                        if displayed_interim_transcript:
                             print(f"\r{' ' * len(displayed_interim_transcript)}\r", end='')
                        final_transcript += transcript # Append, though single_utterance usually means one final result
                        print(f"STT Final: {final_transcript}")
                        self._audio_stream_active = False
                        # Send sentinel only if generator might be waiting.
                        # If API closed stream (normal for single_utterance), generator should exit.
                        # Putting None ensures _audio_generator stops if it's in a blocking get().
                        self._buffer.put(None)
                        break
                    else: # Interim result
                        if displayed_interim_transcript:
                            print(f"\r{' ' * len(displayed_interim_transcript)}\r", end='')
                        interim_display = f"STT Interim: {transcript}"
                        print(interim_display, end='')
                        displayed_interim_transcript = interim_display

            if displayed_interim_transcript: # Clear any final interim line
                 print(f"\r{' ' * len(displayed_interim_transcript)}\r", end='')

        except sd.PortAudioError as pae:
            print(f"STT Error: Microphone/audio device issue: {pae}")
            self._audio_stream_active = False
            # Ensure generator is stopped if it was started
            if hasattr(self, '_buffer') and isinstance(self._buffer, queue.Queue'):
                 self._buffer.put(None)
            return "ERROR_AUDIO_DEVICE"
        except Exception as e: # Catch other exceptions, including Google API errors
            print(f"STT Error: General STT service error: {e}")
            self._audio_stream_active = False
            if hasattr(self, '_buffer') and isinstance(self._buffer, queue.Queue'):
                 self._buffer.put(None)
            return "ERROR_STT_SERVICE"
        finally:
            # This block executes regardless of exceptions in try.
            # Ensures that if _audio_stream_active was True, it's set to False
            # and the generator is signalled to stop.
            if self._audio_stream_active: # If stream was active and didn't stop cleanly
                 self._audio_stream_active = False
                 if hasattr(self, '_buffer') and isinstance(self._buffer, queue.Queue'):
                    self._buffer.put(None) # Signal generator to stop

        if not final_transcript:
            # This means the loop completed without result.is_final being true,
            # or self.responses was empty/None.
            print("No speech detected or transcribed.")
            return "" # Return empty string if no final transcript was captured.

        return final_transcript.strip()

class TextToSpeechHandler:
    def __init__(self, language_code="en-US", voice_name="en-US-Standard-C"):
        self.client = texttospeech.TextToSpeechClient()
        self.voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

    def speak(self, text_to_speak):
        if not text_to_speak:
            print("TTS: No text to speak.")
            return

        print(f"TTS Speaking: {text_to_speak[:100]}{'...' if len(text_to_speak) > 100 else ''}")

        try:
            synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice_params,
                audio_config=self.audio_config
            )
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.write(response.audio_content)
            temp_file_path = temp_file.name
            temp_file.close()

            playsound.playsound(temp_file_path)

        except Exception as e:
            print(f"TTS Error: Failed to synthesize or play speech: {e}")
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    print(f"TTS Error: Failed to delete temporary file {temp_file_path}: {e}")

if __name__ == '__main__':
    # STT Test
    print("--- STT Test ---")
    stt_handler = SpeechToTextHandler()
    try:
        print("\nSpeak now for STT test (or press Ctrl+C to skip to TTS test)...")
        text = stt_handler.listen_and_transcribe()
        if text == "ERROR_AUDIO_DEVICE":
            print("STT Test: Audio device error. Cannot perform test.")
        elif text == "ERROR_STT_SERVICE":
            print("STT Test: STT service error. Cannot perform test.")
        elif text:
            print(f"--- You said: {text} ---")
        else:
            print("--- No transcription or an error occurred ---")
    except KeyboardInterrupt:
        print("\nSkipping STT test.")
    except Exception as e: # Catch any other unexpected error during test setup/call
        print(f"Error during STT test setup/call: {e}")
    finally:
        print("--- End of STT Test ---")

    # Example TTS usage
    print("\n--- TTS Test ---")
    tts_handler = TextToSpeechHandler()
    tts_handler.speak("Hello, this is a test of the Text to Speech system.")
    tts_handler.speak("I should be able to say another sentence after this one.")
    print("--- End of TTS Test ---")
