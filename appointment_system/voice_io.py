import sounddevice as sd
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions
)
# Ensure ListenWebSocketOptions and ListenWebSocketResponse are NOT imported from specific submodules
# if they were causing ImportErrors. Options will be passed as dict.
import asyncio
import threading
import os
import queue

# Configuration constants
SAMPLE_RATE = 16000 # This is for STT (Deepgram)
SD_CHUNK_SIZE = int(SAMPLE_RATE / 10) # 100ms for sounddevice RawInputStream with STT

class SpeechToTextHandler:
    def __init__(self, client: DeepgramClient):
        self.deepgram_client: DeepgramClient = client
        self.final_transcript: str = ""
        self.interim_transcript: str = ""
        self._current_utterance_final_transcript: str = ""

        self.transcript_ready_event: threading.Event = threading.Event()
        self._dg_async_completion_event: asyncio.Event = None

        self.dg_connection = None # Will be an instance from listen.websocket.v("1")
        self._audio_stream_active: bool = False
        self._audio_buffer: queue.Queue = queue.Queue()
        self._deepgram_thread = None # To store the thread object

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Sounddevice status: {status}", flush=True)
        if self._audio_stream_active:
            self._audio_buffer.put(bytes(indata))

    async def _on_open(self, dg_connection_instance, open_response, **kwargs): # dg_connection_instance is self.dg_connection
        print(f"Deepgram STT (WebSocket): Connection Open: {open_response}")

    async def _on_message(self, dg_connection_instance, result, **kwargs): # result is typically a dict
        try:
            message_type = result.get("type")

            if message_type == "Results":
                channel = result.get("channel", {}).get("alternatives", [{}])[0]
                transcript = channel.get("transcript", "")
                is_final = result.get("is_final", False)
                speech_final = result.get("speech_final", False)

                if transcript:
                    if is_final:
                        self._current_utterance_final_transcript += transcript
                        # Add space if segment doesn't end with one and is not empty
                        if transcript and not transcript.isspace() and not self._current_utterance_final_transcript.endswith(" "):
                             self._current_utterance_final_transcript += " "

                        if speech_final:
                            self.final_transcript = self._current_utterance_final_transcript.strip()
                            self._current_utterance_final_transcript = "" # Reset for next utterance

                            if self.interim_transcript: # Clear any lingering interim display
                                print(f"\r{' ' * (len(self.interim_transcript) + 40)}\r", end='') # Increased padding
                            print(f"STT Final (Deepgram WebSocket): {self.final_transcript}")
                            self.interim_transcript = "" # Clear interim

                            if not self.transcript_ready_event.is_set(): self.transcript_ready_event.set()
                            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()
                        else: # is_final but not speech_final
                            if self.interim_transcript: print(f"\r{' ' * (len(self.interim_transcript) + 40)}\r", end='')
                            self.interim_transcript = self._current_utterance_final_transcript.strip()
                            print(f"STT Update (Deepgram WebSocket): {self.interim_transcript}", end='')
                    else: # Not is_final (interim result)
                        if self.interim_transcript: print(f"\r{' ' * (len(self.interim_transcript) + 40)}\r", end='')
                        # For interim, display current segment's interim + what was building up if utterance is long
                        current_segment_interim = self._current_utterance_final_transcript + transcript
                        self.interim_transcript = current_segment_interim.strip()
                        print(f"STT Interim (Deepgram WebSocket): {self.interim_transcript}", end='')

            elif message_type == "Metadata":
                print(f"Deepgram STT Metadata (WebSocket): {result.get('metadata')}")
            elif message_type == "SpeechStarted":
                 print("Deepgram STT (WebSocket): Speech started.")
            elif message_type == "UtteranceEnd":
                 print("Deepgram STT (WebSocket): Utterance ended by VAD.")

        except Exception as e:
            print(f"Error processing Deepgram STT message (WebSocket): {e} - Result: {result}")

    async def _on_error(self, dg_connection_instance, error, **kwargs):
        error_message_detail = str(error.get('message') if isinstance(error, dict) else error)
        self.final_transcript = f"ERROR_DEEPGRAM_STT_WS: {error_message_detail}"
        print(f"Deepgram STT Error (WebSocket): {error_message_detail}")
        if not self.transcript_ready_event.is_set(): self.transcript_ready_event.set()
        if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()

    async def _on_close(self, dg_connection_instance, close_code, close_reason, **kwargs):
        print(f"Deepgram STT Connection Closed (WebSocket): Code {close_code}, Reason: {close_reason}")
        if self._current_utterance_final_transcript and not self.final_transcript.startswith("ERROR_") and not self.final_transcript :
             self.final_transcript = self._current_utterance_final_transcript.strip()
             print(f"STT Final (Deepgram WebSocket - on close): {self.final_transcript}")
        elif not self.final_transcript :
             self.final_transcript = "ERROR_DEEPGRAM_CLOSED_UNEXPECTEDLY_WS"
        self._current_utterance_final_transcript = ""

        if not self.transcript_ready_event.is_set(): self.transcript_ready_event.set()
        if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()

    def _run_deepgram_in_thread(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._dg_async_completion_event = asyncio.Event()

            options_dict = {
                "model": "nova-2", "language": "en-US", "smart_format": True,
                "encoding": "linear16", "sample_rate": SAMPLE_RATE, "channels": 1,
                "interim_results": True, "utterance_end_ms": "1000",
                "vad_events": True, "punctuate": True
            }
            loop.run_until_complete(self._start_and_run_deepgram(options_dict, self._dg_async_completion_event))
        except Exception as e:
            print(f"Critical error in Deepgram STT thread (WebSocket V2): {e}")
            self.final_transcript = "ERROR_DEEPGRAM_THREAD_CRASH_WS_V2"
            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set():
                self._dg_async_completion_event.set()
            if not self.transcript_ready_event.is_set():
                self.transcript_ready_event.set()
        finally:
            print("Deepgram STT (WebSocket V2): _run_deepgram_in_thread finished.")

    async def _start_and_run_deepgram(self, options_dict: dict, completion_event: asyncio.Event):
        self.dg_connection.on("open", self._on_open)
        self.dg_connection.on("message", self._on_message)
        self.dg_connection.on("error", self._on_error)
        self.dg_connection.on("close", self._on_close)

        print(f"Deepgram STT (WebSocket V2): Attempting to start connection with options: {options_dict}")
        try:
            await self.dg_connection.start(**options_dict)

            print("Deepgram STT (WebSocket V2): Connection `start()` method has completed (connection lifecycle finished).")
            await asyncio.wait_for(completion_event.wait(), timeout=10.0)
            print("Deepgram STT (WebSocket V2): Completion signal processed.")
        except asyncio.TimeoutError:
            print("Deepgram STT (WebSocket V2): Timeout waiting for completion signal after start() should have managed the connection.")
        except Exception as e:
            print(f"Deepgram STT Error (WebSocket V2): Exception during start or while running: {e}")
        finally:
            print("Deepgram STT (WebSocket V2): _start_and_run_deepgram coroutine finishing. Ensuring connection is closed.")
            if self.dg_connection:
                try:
                    await self.dg_connection.finish()
                    print("Deepgram STT (WebSocket V2): Connection finished via finally block.")
                except Exception as e_finish:
                    print(f"Deepgram STT (WebSocket V2): Error during finish() in finally: {e_finish}")

            if not completion_event.is_set():
                completion_event.set()

    def listen_and_transcribe(self):
        self.final_transcript = ""
        self.interim_transcript = ""
        self._current_utterance_final_transcript = ""
        self.transcript_ready_event.clear()
        self._audio_stream_active = True

        while not self._audio_buffer.empty():
            try: self._audio_buffer.get_nowait()
            except queue.Empty: break

        deepgram_thread = None # Renamed from self._deepgram_thread for local scope clarity
        try:
            self.dg_connection = self.deepgram_client.listen.websocket.v("1")

            deepgram_thread = threading.Thread(target=self._run_deepgram_in_thread)
            deepgram_thread.daemon = True
            deepgram_thread.start()

            print("Listening (Deepgram WebSocket V2)...")
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=SD_CHUNK_SIZE,
                                   device=None, dtype='int16', channels=1,
                                   callback=self._audio_callback) as stream:

                while self._audio_stream_active and not self.transcript_ready_event.is_set():
                    try:
                        audio_chunk = self._audio_buffer.get(timeout=0.1)
                        if audio_chunk is None: self._audio_stream_active = False; break
                        if self.dg_connection:
                            if self.dg_connection.send(audio_chunk) is False:
                                print("Deepgram STT (WebSocket V2): send returned false. Stopping audio send.")
                                self._audio_stream_active = False; break
                    except queue.Empty: continue

            self._audio_stream_active = False
            self._audio_buffer.put(None)

            timeout_seconds = 30
            if not self.transcript_ready_event.wait(timeout=timeout_seconds):
                print(f"Deepgram STT (WebSocket V2) Error: Timed out after {timeout_seconds}s waiting for transcript.")
                self.final_transcript = "ERROR_DEEPGRAM_TIMEOUT_WS_V2"

            if self.interim_transcript: print(f"\r{' ' * (len(self.interim_transcript) + 40)}\r", end='') # Increased padding
            print("Deepgram STT (WebSocket V2): listen_and_transcribe finished waiting for event.")

        except sd.PortAudioError as pae:
            self.final_transcript = "ERROR_AUDIO_DEVICE_WS_V2"
            print(f"STT Error (WebSocket V2): Mic issue: {pae}")
        except Exception as e:
            self.final_transcript = "ERROR_DEEPGRAM_UNEXPECTED_WS_V2"
            print(f"STT Error (WebSocket V2): General setup/runtime: {e}")
        finally:
            self._audio_stream_active = False
            while not self._audio_buffer.empty():
                try: self._audio_buffer.get_nowait()
                except queue.Empty: break
            self._audio_buffer.put(None)

            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set():
                print("Deepgram STT (WebSocket V2): Forcing async completion from listen_and_transcribe finally.")
                self._dg_async_completion_event.set()

            if deepgram_thread and deepgram_thread.is_alive():
                print("Deepgram STT (WebSocket V2): Waiting for Deepgram thread to join...")
                deepgram_thread.join(timeout=5.0)
                if deepgram_thread.is_alive(): print("Deepgram STT (WebSocket V2) Warning: Thread did not join cleanly.")
            print("Deepgram STT (WebSocket V2): Exiting listen_and_transcribe.")

        if not self.final_transcript.strip() or self.final_transcript.startswith("ERROR_"):
            print(f"Deepgram STT (WebSocket V2): No valid transcript. Result: '{self.final_transcript}'")
        return self.final_transcript.strip()

# --- TextToSpeechHandler class (Deepgram TTS) ---
class TextToSpeechHandler: # This class is preserved from the input file
    def __init__(self, client: DeepgramClient, model: str = "aura-asteria-en",
                 sample_rate: int = 24000, encoding: str = "linear16", container: str = "none"):
        self.deepgram_client = client
        self.tts_model = model
        self.tts_sample_rate = sample_rate
        self.tts_encoding = encoding
        self.tts_container = container

        if self.tts_encoding != "linear16" or self.tts_container != "none":
            print(f"Warning: TextToSpeechHandler is currently optimized for linear16 encoding and 'none' container for direct playback. Current settings: encoding='{self.tts_encoding}', container='{self.tts_container}'")

    async def _speak_async(self, text_to_speak):
        if not text_to_speak:
            print("TTS (Deepgram): No text to speak.")
            return

        print(f"TTS Speaking (Deepgram SDK v4.3.1 - speak.rest.stream_memory with params dict V3): {text_to_speak[:60]}{'...' if len(text_to_speak) > 60 else ''}")

        source_payload = {"text": text_to_speak} # This is the JSON body

        # These options MUST be passed as a dictionary to the 'params' keyword argument
        tts_query_params = {
            "model": self.tts_model,
            "encoding": self.tts_encoding,
            "sample_rate": self.tts_sample_rate,
            "container": self.tts_container
            # e.g., "voice": "aura-helios-en"
        }

        try:
            # API call with source_payload and params=tts_query_params
            response = await self.deepgram_client.speak.rest.v("1").stream_memory(
                source_payload,      # First argument is the source (JSON body)
                params=tts_query_params  # TTS options as a dictionary passed to 'params' kwarg
            )

            audio_stream = response.stream
            if audio_stream:
                chunk_size = 1024 * 4
                with sd.RawOutputStream(samplerate=self.tts_sample_rate,
                                        channels=1,
                                        dtype='int16',
                                        device=None) as stream_player:
                    while True:
                        chunk = await audio_stream.read(chunk_size)
                        if not chunk:
                            break
                        stream_player.write(chunk)
            else:
                print("Deepgram TTS Error: Failed to obtain audio stream from response using stream_memory.")

        except sd.PortAudioError as pae:
            print(f"TTS Playback Error (PortAudioError with Deepgram TTS): {pae}.")
        except Exception as e:
            print(f"Deepgram TTS Error in _speak_async (using stream_memory with params dict V2): {e}") # V2 in log is from prompt, actual is V3

    def speak(self, text_to_speak):
        # This synchronous wrapper should remain
        try:
            # Ensure asyncio is imported at module level
            asyncio.run(self._speak_async(text_to_speak))
        except RuntimeError as re:
            if "cannot run event loop while another loop is running" in str(re) or                "Nesting asyncio event loops is not supported" in str(re):
                print(f"TTS Async Error: Could not run speak_async due to existing event loop: {re}. This can happen if speak() is called from an async context.")
            else:
                # Re-raise other RuntimeErrors if they are not related to event loop nesting
                print(f"TTS Runtime Error: {re}")
                # raise # Optionally re-raise, or just log and continue
        except Exception as e:
            print(f"Unexpected error in speak() method: {e}")

if __name__ == '__main__': # This block is preserved from the input file
    from dotenv import load_dotenv
    load_dotenv()

    print("--- Voice I/O Module Test (Deepgram SDK v4.x Interfaces) ---")

    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")

    if not deepgram_api_key:
        print("CRITICAL ERROR: DEEPGRAM_API_KEY environment variable not set. Cannot run STT or TTS tests.")
    else:
        print(f"Using DEEPGRAM_API_KEY: ...{deepgram_api_key[-4:] if len(deepgram_api_key) > 4 else '...key_is_short'}")

        try:
            client_config = DeepgramClientOptions(options={"keepalive": "true"})
            dg_client = DeepgramClient(api_key=deepgram_api_key, config=client_config)
        except Exception as e:
            print(f"Failed to initialize DeepgramClient: {e}")
            dg_client = None

        if dg_client:
            print("\n--- Testing TextToSpeechHandler (Deepgram TTS - speak.rest.stream_memory) ---")
            try:
                tts_handler = TextToSpeechHandler(client=dg_client, model="aura-asteria-en", sample_rate=24000)

                print("Attempting to speak a short phrase with Deepgram TTS (stream_memory)...")
                tts_handler.speak("Hello, this is a test of Deepgram Text to Speech using the stream memory method.")

                print("Attempting to speak a slightly longer phrase...")
                tts_handler.speak("This audio should be streamed directly from Deepgram and played in real-time via sounddevice.")

            except Exception as e:
                print(f"Error during Deepgram TTS (stream_memory) test: {e}")
            print("--- Finished Deepgram TTS (stream_memory) Test ---")

            print("\n--- Testing SpeechToTextHandler (Deepgram STT - listen.websocket) ---")
            try:
                stt_handler = SpeechToTextHandler(client=dg_client)
                for i in range(2):
                    print(f"\nSTT Attempt {i+1}/2 (WebSocket). Press Enter to start speaking, then speak. (Ctrl+C to skip)")
                    input()
                    print("Recording (Deepgram WebSocket)...")
                    text = stt_handler.listen_and_transcribe()
                    if text and not text.startswith("ERROR_"):
                        print(f"--- You said (Deepgram WebSocket): {text} ---")
                    else:
                        print(f"--- No valid transcription from Deepgram (WebSocket). Result: {text} ---")
            except KeyboardInterrupt:
                print("\nSkipped remaining STT tests.")
            except Exception as e:
                print(f"Error during Deepgram STT (WebSocket) test setup or execution: {e}")
            print("--- Finished Deepgram STT (WebSocket) Test ---")
        else:
            print("Skipping STT/TTS tests due to DeepgramClient initialization failure.")

    print("\n--- Voice I/O Module Test Complete ---")
