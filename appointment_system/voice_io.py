import sounddevice as sd
# from google.cloud import speech # Removed
from google.cloud import texttospeech # Keep for TTS
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
import asyncio
import threading
import os
import queue

# Configuration constants
SAMPLE_RATE = 16000
SD_CHUNK_SIZE = int(SAMPLE_RATE / 10) # 100ms for sounddevice RawInputStream

class SpeechToTextHandler:
    def __init__(self, deepgram_api_key):
        if not deepgram_api_key:
            raise ValueError("Deepgram API key is required for SpeechToTextHandler.")

        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.deepgram_client = DeepgramClient(api_key=deepgram_api_key, config=config)

        self.final_transcript = ""
        self.interim_transcript = "" # For displaying interim results
        self._full_final_speech = "" # To accumulate final segments for current utterance
        self.transcript_ready_event = threading.Event()
        self.dg_connection = None
        self._audio_stream_active = False
        self._audio_buffer = queue.Queue()
        self._deepgram_thread = None


    def _audio_callback(self, indata, frames, time, status):
        if status:
            # sd.CallbackFlags also has .input_underflow, .input_overflow etc.
            print(f"Sounddevice status: {status}", flush=True)
        if self._audio_stream_active:
            self._audio_buffer.put(bytes(indata))

    async def _on_open(self, open_result, **kwargs):
        print(f"Deepgram Connection Open: {open_result}")

    async def _on_message(self, result, **kwargs):
        # This handler is called for LiveTranscriptionEvents.TRANSCRIPT_RECEIVED
        transcript = result.channel.alternatives[0].transcript

        if result.is_final and transcript.strip(): # A final segment of speech
            self._full_final_speech += transcript + " "
            # Update interim display to show the latest final part clearly before it's fully final for utterance
            # print(f"\rSTT Interim (Deepgram): {self._full_final_speech}", end='')
            # self.interim_transcript = self._full_final_speech # Keep interim updated with latest confirmed speech

        if result.speech_final: # End of an utterance
            print(f"\r{' ' * len(self.interim_transcript)}\r", end='') # Clear last interim line
            self.final_transcript = self._full_final_speech.strip()
            print(f"STT Final (Deepgram): {self.final_transcript}")
            self.interim_transcript = ""
            self._full_final_speech = "" # Reset for next utterance
            self.transcript_ready_event.set() # Signal main thread: speech is fully processed

        elif not result.is_final and transcript.strip(): # Interim result for current segment
            current_display_transcript = self._full_final_speech + transcript
            print(f"\r{' ' * len(self.interim_transcript)}\r", end='')
            self.interim_transcript = current_display_transcript
            print(f"STT Interim (Deepgram): {self.interim_transcript}", end='')


    async def _on_error(self, error, **kwargs):
        error_message = str(error.get('message', str(error))) if isinstance(error, dict) else str(error)
        print(f"Deepgram Error: {error_message}")
        self.final_transcript = f"ERROR_DEEPGRAM_STT: {error_message}"
        if self._full_final_speech: # If there was some speech before error
            self.final_transcript = self._full_final_speech.strip() + f" (Error after: {error_message})"
        self.transcript_ready_event.set()

    async def _on_close(self, close, **kwargs):
        print(f"Deepgram Connection Closed: {close}")
        if not self.transcript_ready_event.is_set():
            if self._full_final_speech: # If connection closes mid-utterance with some final segments
                self.final_transcript = self._full_final_speech.strip()
            elif not self.final_transcript: # If no final transcript was set by error or speech_final
                 self.final_transcript = "ERROR_DEEPGRAM_CLOSED" # Or empty if preferred
            self.transcript_ready_event.set()

    async def _run_deepgram_async_tasks(self):
        # This is the core async part that will run in the dedicated thread
        self.dg_connection.on(LiveTranscriptionEvents.OPEN, self._on_open)
        self.dg_connection.on(LiveTranscriptionEvents.TRANSCRIPT_RECEIVED, self._on_message)
        self.dg_connection.on(LiveTranscriptionEvents.ERROR, self._on_error)
        self.dg_connection.on(LiveTranscriptionEvents.CLOSE, self._on_close)

        options = LiveOptions(
            model="nova-2", language="en-US", smart_format=True,
            encoding="linear16", sample_rate=SAMPLE_RATE, channels=1,
            interim_results=True, utterance_end_ms="1200", # Adjusted for slightly longer pauses
            # vad_events=True # If on_speech_started etc. are needed
        )

        try:
            print("Deepgram: Starting connection...")
            if not await self.dg_connection.start(options): # Returns True on success
                print("Deepgram: Failed to start connection.")
                self.final_transcript = "ERROR_DEEPGRAM_START_FAILED"
                self.transcript_ready_event.set()
                return

            # Audio sending loop (moved into the async thread for better control with async dg_connection)
            while self._audio_stream_active:
                try:
                    audio_chunk = await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self._audio_buffer.get, True, 0.1), timeout=0.2)
                    if audio_chunk is None: # Sentinel to stop
                        break
                    if not self.dg_connection.send(audio_chunk):
                        print("Deepgram: Failed to send audio, closing stream from here.")
                        self._audio_stream_active = False # Stop further audio processing
                        break
                except queue.Empty: # Expected when no audio, continue loop
                    continue
                except asyncio.TimeoutError: # Expected when no audio, continue loop
                    continue
                except Exception as e:
                    print(f"Deepgram: Error in audio sending loop: {e}")
                    self._audio_stream_active = False
                    break

            print("Deepgram: Audio sending loop finished.")

        except Exception as e:
            print(f"Deepgram: Error in run_deepgram_async_tasks: {e}")
            if not self.transcript_ready_event.is_set():
                self.final_transcript = f"ERROR_DEEPGRAM_ASYNC_TASK: {e}"
                self.transcript_ready_event.set()
        finally:
            if self.dg_connection and self.dg_connection.is_connected:
                await self.dg_connection.finish() # Ensure connection is closed
            print("Deepgram: Async tasks finished.")


    def _start_deepgram_thread(self):
        # Runs the asyncio event loop for Deepgram in a separate thread
        async def main_async_loop():
            await self._run_deepgram_async_tasks()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main_async_loop())
        except Exception as e:
            print(f"Critical error in Deepgram thread: {e}")
            if not self.transcript_ready_event.is_set():
                 self.final_transcript = "ERROR_DEEPGRAM_THREAD_CRASH"
                 self.transcript_ready_event.set()


    def listen_and_transcribe(self):
        self.final_transcript = ""
        self.interim_transcript = ""
        self._full_final_speech = ""
        self.transcript_ready_event.clear()
        self._audio_stream_active = True

        while not self._audio_buffer.empty(): # Clear buffer from previous runs
            try: self._audio_buffer.get_nowait()
            except queue.Empty: break

        try:
            self.dg_connection = self.deepgram_client.listen.live.v("1")

            self._deepgram_thread = threading.Thread(target=self._start_deepgram_thread)
            self._deepgram_thread.daemon = True
            self._deepgram_thread.start()

            print("Listening (Deepgram)...")
            # Using sounddevice RawInputStream
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=SD_CHUNK_SIZE,
                                   device=None, dtype='int16', channels=1,
                                   callback=self._audio_callback) as stream:

                # Wait for the final transcript or an error
                timeout_seconds = 30  # Max time to wait for a complete utterance
                if not self.transcript_ready_event.wait(timeout=timeout_seconds):
                    print("Deepgram STT: Timed out waiting for transcript.")
                    if not self.final_transcript: # If no error/transcript set by Deepgram handlers
                        self.final_transcript = "ERROR_DEEPGRAM_TIMEOUT"

                # Ensure interim transcript display is cleared after waiting
                if self.interim_transcript:
                    print(f"\r{' ' * len(self.interim_transcript)}\r", end='')


        except sd.PortAudioError as pae:
            print(f"STT Error: Microphone/audio device issue: {pae}")
            self.final_transcript = "ERROR_AUDIO_DEVICE"
        except Exception as e:
            print(f"STT Error: General Deepgram setup/service error: {e}")
            if not self.final_transcript: # Avoid overwriting specific Deepgram error
                self.final_transcript = "ERROR_DEEPGRAM_STT_SETUP"
        finally:
            self._audio_stream_active = False # Signal audio callback and sending loop to stop
            self._audio_buffer.put(None) # Sentinel for audio sending loop if it's blocking on get

            if self.dg_connection and self.dg_connection.is_connected:
                 # dg_connection.finish() should be called from the thread where it runs.
                 # Here, we rely on the _start_deepgram_thread's finally block.
                 pass

            if self._deepgram_thread and self._deepgram_thread.is_alive():
                print("Deepgram STT: Waiting for Deepgram thread to finish...")
                self._deepgram_thread.join(timeout=5.0) # Wait for thread to complete
                if self._deepgram_thread.is_alive():
                    print("Deepgram STT: Deepgram thread did not finish cleanly.")
            print("Deepgram STT: listen_and_transcribe finished.")


        if not self.final_transcript.strip() and not self.final_transcript.startswith("ERROR_"):
            print(f"Deepgram STT: No valid transcript obtained. Final raw: '{self.final_transcript}'")
            # Return empty string for "no speech" or only errors.
            if not self.final_transcript.startswith("ERROR_"):
                return ""

        return self.final_transcript.strip()


# --- TextToSpeechHandler class remains unchanged from previous steps ---
class TextToSpeechHandler:
    def __init__(self, language_code="en-US", voice_name="en-US-Standard-C"):
        self.client = texttospeech.TextToSpeechClient()
        self.language_code = language_code
        self.voice_name = voice_name
        self.sample_rate_hertz = SAMPLE_RATE

        self.voice_params = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hertz,
        )

    def speak(self, text_to_speak):
        if not text_to_speak:
            print("TTS: No text to speak.")
            return

        print(f"TTS Speaking (streaming): {text_to_speak[:60]}{'...' if len(text_to_speak) > 60 else ''}")

        try:
            streaming_config_request = texttospeech.StreamingSynthesizeConfig(
                audio_config=self.audio_config,
                voice=self.voice_params
            )
            request_config = texttospeech.StreamingSynthesizeRequest(
                streaming_config=streaming_config_request
            )
            synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
            request_text = texttospeech.StreamingSynthesizeRequest(
                synthesis_input=synthesis_input
            )
            requests_iterable = [request_config, request_text]

            streaming_responses = self.client.streaming_synthesize(requests=requests_iterable)

            with sd.RawOutputStream(samplerate=self.sample_rate_hertz,
                                    channels=1,
                                    dtype='int16',
                                    ) as stream:
                for response_chunk in streaming_responses:
                    if response_chunk.audio_content:
                        stream.write(response_chunk.audio_content)
        except sd.PortAudioError as pae:
            print(f"TTS Playback Error (PortAudioError): {pae}. Check your audio output device and configuration.")
        except Exception as e:
            print(f"TTS Streaming or Playback Error: {e}")

if __name__ == '__main__':
    # STT Test (Deepgram)
    print("--- Deepgram STT Test ---")
    try:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set for testing.")

        stt_handler = SpeechToTextHandler(deepgram_api_key=api_key)

        # Test STT once
        print("\nSpeak now for Deepgram STT test (Ctrl+C to skip/exit)...")
        text = stt_handler.listen_and_transcribe()

        if text and not text.startswith("ERROR_"):
            print(f"--- You said (Deepgram): '{text}' ---")
        elif text.startswith("ERROR_AUDIO_DEVICE"):
            print(f"--- STT Test: Audio device error: {text} ---")
        else:
            print(f"--- No valid transcription from Deepgram. Result: '{text}' ---")

    except KeyboardInterrupt:
        print("\nUser skipped/exited Deepgram STT test.")
    except ValueError as ve:
        print(f"Config Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during STT test setup: {e}")
    finally:
        print("--- End of Deepgram STT Test Section ---")

    # TTS Test (Google) - remains the same
    print("\n--- TTS Streaming Test (Google) ---")
    tts_handler = TextToSpeechHandler() # Google TTS
    tts_handler.speak("This is a test of the Google Text to Speech system using streaming playback.")
    print("--- End of TTS Test ---")
