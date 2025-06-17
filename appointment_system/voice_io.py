import sounddevice as sd
from deepgram import DeepgramClient, DeepgramClientOptions, LiveOptions
import asyncio
import threading
import os
import queue
# Removed subprocess, shutil, json, Path as they were for Piper TTS

# Configuration constants
SAMPLE_RATE = 16000 # This is for STT (Deepgram)
SD_CHUNK_SIZE = int(SAMPLE_RATE / 10) # 100ms for sounddevice RawInputStream with STT

class SpeechToTextHandler:
    def __init__(self, deepgram_api_key):
        if not deepgram_api_key:
            raise ValueError("Deepgram API key is required for SpeechToTextHandler.")

        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.deepgram_client = DeepgramClient(api_key=deepgram_api_key, config=config)

        self.final_transcript = ""
        self.interim_transcript = ""
        self._current_utterance_final_transcript = ""
        self.transcript_ready_event = threading.Event()
        self.dg_connection = None
        self._audio_stream_active = False
        self._audio_buffer = queue.Queue()
        self._deepgram_thread = None


    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Sounddevice status: {status}", flush=True)
        if self._audio_stream_active:
            self._audio_buffer.put(bytes(indata))

    async def _on_open(self, open_result, **kwargs):
        print(f"Deepgram Connection Open: {open_result}")

    async def _on_message(self, result, **kwargs):
        try:
            transcript = result.channel.alternatives[0].transcript
            if transcript:
                if result.is_final:
                    self._current_utterance_final_transcript += transcript + " "
                    if self.interim_transcript:
                        print(f"\r{' ' * (len(self.interim_transcript) + 30)}\r", end='')
                    if result.speech_final:
                        self.final_transcript = self._current_utterance_final_transcript.strip()
                        self._current_utterance_final_transcript = ""
                        print(f"STT Final (Deepgram): {self.final_transcript}")
                        self.interim_transcript = ""
                        self.transcript_ready_event.set()
                    else:
                        print(f"STT Update (Deepgram): {self._current_utterance_final_transcript.strip()}", end='')
                        self.interim_transcript = f"STT Update (Deepgram): {self._current_utterance_final_transcript.strip()}"
                else:
                    if self.interim_transcript:
                        print(f"\r{' ' * (len(self.interim_transcript) + 30)}\r", end='')
                    current_full_interim = self._current_utterance_final_transcript + transcript
                    self.interim_transcript = f"STT Interim (Deepgram): {current_full_interim.strip()}"
                    print(self.interim_transcript, end='')
        except Exception as e:
            print(f"Error processing Deepgram message: {e} - Result: {result}")

    async def _on_error(self, error, **kwargs):
        from_start_call = kwargs.get('from_start_call', False)
        error_message_detail = "Unknown Deepgram Error"
        if isinstance(error, dict) and 'message' in error:
            error_message_detail = error['message']
        elif isinstance(error, Exception):
            error_message_detail = str(error)
        elif isinstance(error, str):
            error_message_detail = error

        full_error_message = f"ERROR_DEEPGRAM_STT: {error_message_detail}"
        print(f"Deepgram Error Captured: {full_error_message}")

        self.final_transcript = full_error_message
        if not self.transcript_ready_event.is_set():
            self.transcript_ready_event.set()

    async def _on_close(self, close, **kwargs):
        print(f"Deepgram Connection Closed: {close}")
        if not self.transcript_ready_event.is_set():
            if self._current_utterance_final_transcript:
                self.final_transcript = self._current_utterance_final_transcript.strip()
                print(f"STT Final (Deepgram - on close): {self.final_transcript}")
            elif not self.final_transcript:
                 self.final_transcript = "ERROR_DEEPGRAM_CLOSED"
            self._current_utterance_final_transcript = ""
            self.transcript_ready_event.set()

    async def _start_and_run_deepgram(self, options):
        self.dg_connection.on("open", self._on_open)
        self.dg_connection.on("transcript_received", self._on_message)
        self.dg_connection.on("error", self._on_error)
        self.dg_connection.on("close", self._on_close)
        print("Deepgram STT: Attempting to start connection with options...")
        try:
            start_status = self.dg_connection.start(options)
            print(f"Deepgram STT: dg_connection.start() called. Returned status: {start_status}")
            if isinstance(start_status, bool) and not start_status:
                print("Deepgram STT Error: start() returned False. Connection failed to initialize properly.")
                await self._on_error({"message": "Connection start returned false"}, from_start_call=True)
        except Exception as e:
            print(f"Deepgram STT Error: Exception during Deepgram start or while it was running: {e}")
            await self._on_error({"message": f"Exception in _start_and_run_deepgram: {e}"})
        try: # Audio sending loop
            while self._audio_stream_active:
                try:
                    audio_chunk = await asyncio.get_event_loop().run_in_executor(None, self._audio_buffer.get, True, 0.1)
                    if audio_chunk is None:
                        print("Deepgram: Sentinel received, stopping audio sending.")
                        break
                    if not self.dg_connection.send(audio_chunk):
                        print("Deepgram: Failed to send audio, connection might be closing.")
                        self._audio_stream_active = False
                        break
                except queue.Empty:
                    if not self._audio_stream_active: break
                    continue
                except Exception as e:
                    print(f"Deepgram: Error in audio sending loop: {e}")
                    self._audio_stream_active = False
                    break
        finally:
            print("Deepgram: Audio sending loop finished or exited.")
            if self.dg_connection and self.dg_connection.is_connected():
                print("Deepgram: Proactively finishing connection from client side after audio sending.")
                await self.dg_connection.finish()

    def _run_deepgram_in_thread(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            options = LiveOptions(
                model="nova-2", language="en-US", smart_format=True,
                encoding="linear16", sample_rate=SAMPLE_RATE, channels=1,
                interim_results=True, utterance_end_ms="1000",
            )
            loop.run_until_complete(self._start_and_run_deepgram(options))
        except Exception as e:
            print(f"Critical error in Deepgram thread: {e}")
            self.final_transcript = "ERROR_DEEPGRAM_THREAD_CRASH"
            if not self.transcript_ready_event.is_set():
                self.transcript_ready_event.set()
        finally:
            print("Deepgram: _run_deepgram_in_thread finished.")

    def listen_and_transcribe(self):
        self.final_transcript = ""
        self.interim_transcript = ""
        self._current_utterance_final_transcript = ""
        self.transcript_ready_event.clear()
        self._audio_stream_active = True
        while not self._audio_buffer.empty():
            try: self._audio_buffer.get_nowait()
            except queue.Empty: break
        self._deepgram_thread = None
        try:
            self.dg_connection = self.deepgram_client.listen.live.v("1")
            self._deepgram_thread = threading.Thread(target=self._run_deepgram_in_thread)
            self._deepgram_thread.daemon = True
            self._deepgram_thread.start()
            print("Listening (Deepgram)...")
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=SD_CHUNK_SIZE,
                                   device=None, dtype='int16', channels=1,
                                   callback=self._audio_callback) as stream:
                timeout_seconds = 30
                if not self.transcript_ready_event.wait(timeout=timeout_seconds):
                    print("Deepgram STT: Timed out waiting for transcript.")
                    if not self.final_transcript:
                        self.final_transcript = "ERROR_DEEPGRAM_TIMEOUT"
                    self._audio_stream_active = False
                    self._audio_buffer.put(None)
            if self._deepgram_thread and self._deepgram_thread.is_alive() and not self.transcript_ready_event.is_set():
                print("Deepgram STT: Forcing event set after timeout/stream closure.")
                self.transcript_ready_event.set()
            if self.interim_transcript:
                print(f"\r{' ' * (len(self.interim_transcript) + 30)}\r", end='')
        except sd.PortAudioError as pae:
            print(f"STT Error: Microphone/audio device issue: {pae}")
            self.final_transcript = "ERROR_AUDIO_DEVICE"
            self._audio_stream_active = False
            self._audio_buffer.put(None)
        except Exception as e:
            print(f"STT Error: General Deepgram setup/runtime error: {e}")
            if not self.final_transcript:
                 self.final_transcript = "ERROR_DEEPGRAM_UNEXPECTED"
            self._audio_stream_active = False
            self._audio_buffer.put(None)
        finally:
            self._audio_stream_active = False
            while not self._audio_buffer.empty():
                try: self._audio_buffer.get_nowait()
                except queue.Empty: break
            self._audio_buffer.put(None)
            if self._deepgram_thread and self._deepgram_thread.is_alive():
                print("Deepgram STT: Waiting for Deepgram thread to join...")
                self.transcript_ready_event.set()
                self._deepgram_thread.join(timeout=3.0)
                if self._deepgram_thread.is_alive():
                    print("Deepgram STT: Warning - Deepgram thread did not join cleanly.")
            print(f"Deepgram STT: listen_and_transcribe finished. Final transcript: '{self.final_transcript}'")
        if self.final_transcript.startswith("ERROR_"):
            return self.final_transcript.strip()
        elif not self.final_transcript.strip():
            return ""
        return self.final_transcript.strip()

# --- TextToSpeechHandler class (Deepgram TTS) ---
class TextToSpeechHandler:
    def __init__(self, deepgram_api_key: str, model: str = "aura-asteria-en",
                 sample_rate: int = 24000, encoding: str = "linear16",
                 container: str = "none"):
        if not deepgram_api_key:
            raise ValueError("Deepgram API key is required for TextToSpeechHandler.")

        client_config = DeepgramClientOptions(options={"keepalive": "true"})
        self.deepgram_client = DeepgramClient(api_key=deepgram_api_key, config=client_config)

        self.tts_model = model
        self.tts_sample_rate = sample_rate
        self.tts_encoding = encoding
        self.tts_container = container

        if self.tts_encoding != "linear16" or self.tts_container != "none":
            print(f"Warning: TextToSpeechHandler is optimized for linear16 encoding and 'none' container for direct playback. Current settings: encoding='{self.tts_encoding}', container='{self.tts_container}'. Playback might fail if not raw PCM.")

    async def _speak_async(self, text_to_speak: str):
        if not text_to_speak:
            print("TTS (Deepgram): No text to speak.")
            return

        print(f"TTS Speaking (Deepgram): {text_to_speak[:60]}{'...' if len(text_to_speak) > 60 else ''}")

        source = {"text": text_to_speak}

        try:
            # Using speak.v("1").stream which returns an object with 'stream' (aiohttp.StreamReader) and 'headers'
            response = await self.deepgram_client.speak.v("1").stream(
                 source,
                 model=self.tts_model,
                 encoding=self.tts_encoding,
                 sample_rate=self.tts_sample_rate,
                 container=self.tts_container
                 # Additional options like voice, pitch, speaking_rate can be added as kwargs
            )

            audio_stream = response.stream
            # print(f"Deepgram TTS Headers: {response.headers}") # For debugging audio format

            if not audio_stream:
                print("Deepgram TTS Error: Failed to obtain audio stream.")
                return

            with sd.RawOutputStream(samplerate=self.tts_sample_rate,
                                    channels=1, # Assuming mono, typical for TTS
                                    dtype='int16', # For linear16 encoding
                                    device=None) as stream_player:

                chunk_size = 1024 * 4 # 4KB chunks
                while True:
                    chunk = await audio_stream.read(chunk_size)
                    if not chunk:
                        break # End of stream
                    stream_player.write(chunk)
            print("Deepgram TTS: Finished speaking.")

        except sd.PortAudioError as pae:
            print(f"TTS Playback Error (PortAudioError with Deepgram TTS): {pae}.")
        except Exception as e:
            # This will catch errors from Deepgram API (e.g., auth, bad request) or other issues.
            print(f"Deepgram TTS Error: {e}")

    def speak(self, text_to_speak: str):
        try:
            # Run the async _speak_async method in a blocking way
            asyncio.run(self._speak_async(text_to_speak))
        except RuntimeError as re:
            if "cannot run event loop while another loop is running" in str(re) or \
               "Nesting asyncio event loops is not supported" in str(re):
                print(f"TTS Async Error: Could not run speak_async due to existing event loop: {re}")
                print("This TTS handler needs to be called from a synchronous context or adapted for nested loops if used within another asyncio app.")
            else:
                # Re-raise other RuntimeErrors if they are not related to event loop nesting
                raise

if __name__ == '__main__':
    print("--- Voice I/O Module Test ---")

    # Test TextToSpeechHandler (Deepgram TTS)
    print("\n--- Testing TextToSpeechHandler (Deepgram TTS) ---")
    dg_api_key_env = os.environ.get("DEEPGRAM_API_KEY")
    if not dg_api_key_env:
        print("Skipping Deepgram TTS test: DEEPGRAM_API_KEY environment variable not set.")
    else:
        try:
            # Example with default Aura model (aura-asteria-en, 24000 Hz)
            # If using a different model, ensure sample_rate matches.
            # For "aura-luna-en" or "aura-stella-en", sample_rate is often 16000.
            # Check Deepgram model documentation for correct sample rates.
            tts_handler = TextToSpeechHandler(deepgram_api_key=dg_api_key_env, model="aura-asteria-en", sample_rate=24000)
            tts_handler.speak("Hello from Deepgram Text to Speech, using the Aura model.")
            tts_handler.speak("This audio is being streamed directly to your speakers.")

            # Example for a model that might use 16000 Hz
            # print("\nTesting with a 16kHz Aura model (example, ensure model name is correct if used)")
            # tts_handler_16khz = TextToSpeechHandler(deepgram_api_key=dg_api_key_env, model="aura-luna-en", sample_rate=16000)
            # tts_handler_16khz.speak("This is a test with Luna at sixteen kilohertz.")

        except Exception as e:
            print(f"Error during Deepgram TTS test: {e}")
    print("--- Finished Deepgram TTS Test ---")


    # Test SpeechToTextHandler (Deepgram STT)
    print("\n--- Testing SpeechToTextHandler (Deepgram STT) ---")
    # dg_api_key_env is already fetched from above
    if not dg_api_key_env: # Check again in case only TTS was skipped
        print("Skipping Deepgram STT test: DEEPGRAM_API_KEY environment variable not set.")
    else:
        try:
            stt_handler = SpeechToTextHandler(deepgram_api_key=dg_api_key_env)
            for i in range(1): # Reduced to 1 attempt for brevity
                print(f"\nSTT Attempt {i+1}/1. Press Enter to start speaking, then speak. (Ctrl+C to skip)")
                try:
                    input()
                    text = stt_handler.listen_and_transcribe()
                    if text and not text.startswith("ERROR_"):
                        print(f"--- You said (Deepgram): {text} ---")
                    else:
                        print(f"--- No valid transcription from Deepgram. Result: {text} ---")
                except KeyboardInterrupt: # Catch Ctrl+C during input()
                    print("\nSTT attempt skipped by user.")
                    break
        except KeyboardInterrupt: # Catch Ctrl+C during handler init or loop
            print("\nSkipped remaining STT tests.")
        except Exception as e:
            print(f"Error during Deepgram STT test setup or execution: {e}")
    print("--- Finished Deepgram STT Test ---")

    print("\n--- Voice I/O Module Test Complete ---")
