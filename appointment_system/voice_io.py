import sounddevice as sd
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions
)
# from deepgram.clients.listen.live.v1options import LiveOptions # Old
from deepgram.clients.listen.websocket.v1.options import ListenWebSocketOptions
from deepgram.clients.listen.websocket.v1.response import ListenWebSocketResponse # For _on_message type hint if used
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

        self.dg_connection = None # Will be ListenWebSocketClient instance
        self._audio_stream_active: bool = False
        self._audio_buffer: queue.Queue = queue.Queue()
        self._deepgram_thread = None


    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Sounddevice status: {status}", flush=True)
        if self._audio_stream_active:
            self._audio_buffer.put(bytes(indata))

    async def _on_open(self, connection, open_response, **kwargs): # connection is self.dg_connection
        print(f"Deepgram STT (WebSocket): Connection Open: {open_response}")

    async def _on_message(self, connection, result: ListenWebSocketResponse, **kwargs): # result is often a dict
        try:
            # The result object from listen.websocket is a dict (parsed JSON).
            message_type = result.get("type")

            if message_type == "Results":
                channel = result.get("channel", {}).get("alternatives", [{}])[0]
                transcript = channel.get("transcript", "")
                is_final = result.get("is_final", False)
                speech_final = result.get("speech_final", False)

                if transcript:
                    if is_final:
                        # Accumulate final segments; some models send word-by-word is_final
                        self._current_utterance_final_transcript += transcript
                        # Add a space if the segment likely needs one, based on if it's a new word.
                        # This simple space adding might not be perfect for all models/languages.
                        if transcript and not transcript.isspace():
                             self._current_utterance_final_transcript += " "

                        if speech_final:
                            self.final_transcript = self._current_utterance_final_transcript.strip()
                            self._current_utterance_final_transcript = ""
                            if self.interim_transcript:
                                print(f"\r{' ' * (len(self.interim_transcript) + 40)}\r", end='') # Clear line
                            print(f"STT Final (Deepgram WebSocket): {self.final_transcript}")
                            self.interim_transcript = ""
                            if not self.transcript_ready_event.is_set(): self.transcript_ready_event.set()
                            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()
                        else: # is_final but not speech_final
                            if self.interim_transcript: print(f"\r{' ' * (len(self.interim_transcript) + 40)}\r", end='')
                            self.interim_transcript = self._current_utterance_final_transcript.strip()
                            print(f"STT Update (Deepgram WebSocket): {self.interim_transcript}", end='')
                    else: # Not is_final (interim result)
                        if self.interim_transcript: print(f"\r{' ' * (len(self.interim_transcript) + 40)}\r", end='')
                        # For interim, display accumulated final parts + current interim segment
                        display_interim = self._current_utterance_final_transcript + transcript
                        self.interim_transcript = display_interim.strip()
                        print(f"STT Interim (Deepgram WebSocket): {self.interim_transcript}", end='')

            elif message_type == "Metadata":
                print(f"Deepgram STT Metadata (WebSocket): {result.get('metadata')}")
            elif message_type == "SpeechStarted":
                 print("Deepgram STT (WebSocket): Speech started.")
            elif message_type == "UtteranceEnd":
                 print("Deepgram STT (WebSocket): Utterance ended by VAD.")

        except Exception as e:
            print(f"Error processing Deepgram STT message (WebSocket): {e} - Result: {result}")

    async def _on_error(self, connection, error, **kwargs):
        error_message_detail = str(error.get('message') if isinstance(error, dict) else error)
        self.final_transcript = f"ERROR_DEEPGRAM_STT_WS: {error_message_detail}"
        print(f"Deepgram STT Error (WebSocket): {error_message_detail}")
        if not self.transcript_ready_event.is_set(): self.transcript_ready_event.set()
        if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()

    async def _on_close(self, connection, code: int, reason: str, **kwargs):
        print(f"Deepgram STT Connection Closed (WebSocket): Code {code}, Reason: {reason}")
        # If connection closed before speech_final, finalize with what we have
        if self._current_utterance_final_transcript and not self.final_transcript.startswith("ERROR_"):
            self.final_transcript = self._current_utterance_final_transcript.strip()
            print(f"STT Final (Deepgram WebSocket - on close): {self.final_transcript}")
        elif not self.final_transcript: # If no transcript and no error set yet
            self.final_transcript = "ERROR_DEEPGRAM_CLOSED_UNEXPECTEDLY_WS"

        if not self.transcript_ready_event.is_set(): self.transcript_ready_event.set()
        if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()

    def _run_deepgram_in_thread(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._dg_async_completion_event = asyncio.Event()

            options = ListenWebSocketOptions(
                model="nova-2", language="en-US", smart_format=True,
                encoding="linear16", sample_rate=SAMPLE_RATE, channels=1,
                interim_results=True, utterance_end_ms="1000", # Consider making this configurable
                vad_events=True
            )
            loop.run_until_complete(self._start_and_run_deepgram(options, self._dg_async_completion_event))
        except Exception as e:
            print(f"Critical error in Deepgram STT thread (WebSocket): {e}")
            self.final_transcript = "ERROR_DEEPGRAM_THREAD_CRASH_WS"
            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()
            if not self.transcript_ready_event.is_set(): self.transcript_ready_event.set()
        finally:
            print("Deepgram STT (WebSocket): _run_deepgram_in_thread finished.")

    async def _start_and_run_deepgram(self, options: ListenWebSocketOptions, completion_event: asyncio.Event):
        self.dg_connection.on("open", self._on_open)
        self.dg_connection.on("message", self._on_message)
        # self.dg_connection.on("results", self._on_message) # 'message' event should cover 'Results' type
        self.dg_connection.on("error", self._on_error)
        self.dg_connection.on("close", self._on_close)
        self.dg_connection.on("metadata", lambda connection, metadata, **kwargs: print(f"STT Meta (WS): {metadata.get('request_id')}"))
        self.dg_connection.on("speech_started", lambda connection, speech_started, **kwargs: print("STT Speech Started (WS)"))
        self.dg_connection.on("utterance_end", lambda connection, utterance_end, **kwargs: print("STT Utterance Ended (WS)"))

        print("Deepgram STT (WebSocket): Attempting to start connection...")
        try:
            options_dict = options.to_dict() if hasattr(options, 'to_dict') else vars(options)

            # For listen.websocket.v1, start() is async and should be awaited.
            # This call itself will run the connection and only complete when the connection is closed.
            await self.dg_connection.start(options_dict)

            print("Deepgram STT (WebSocket): Connection `start()` method completed (implies connection closed or error).")
            # If start() completes, it means the connection lifecycle is done.
            # The completion_event should have been set by on_close or on_error.
            # We can await it here with a small timeout as a final check or if start() returns prematurely.
            await asyncio.wait_for(completion_event.wait(), timeout=5.0)
            print("Deepgram STT (WebSocket): Completion signal processed after start() returned.")

        except asyncio.TimeoutError:
            print("Deepgram STT (WebSocket): Timeout waiting for completion signal after start() returned. This might be okay if connection closed normally.")
        except Exception as e:
            print(f"Deepgram STT Error (WebSocket): Exception during start or while running: {e}")
            if not completion_event.is_set(): completion_event.set()
        finally:
            print("Deepgram STT (WebSocket): _start_and_run_deepgram coroutine finishing.")
            # Connection should be closed by `await self.dg_connection.start()` completing.
            # Explicitly calling finish() here might be redundant or cause issues if called on an already closing connection.
            # However, if start() was not awaited or if it's non-blocking, finish() is crucial.
            # Given `await start()`, this finish call is more of a safeguard.
            if self.dg_connection and not self.dg_connection.finished:
                try:
                    await self.dg_connection.finish()
                    print("Deepgram STT (WebSocket): Connection finished via finally block.")
                except Exception as e_finish:
                    print(f"Deepgram STT (WebSocket): Error during finish() in finally: {e_finish}")
            if not completion_event.is_set(): # Ensure completion event is set
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

        self._deepgram_thread = None # Ensure it's reset
        try:
            self.dg_connection = self.deepgram_client.listen.websocket.v("1")

            self._deepgram_thread = threading.Thread(target=self._run_deepgram_in_thread)
            self._deepgram_thread.daemon = True
            self._deepgram_thread.start()

            print("Listening (Deepgram WebSocket)...")
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=SD_CHUNK_SIZE,
                                   device=None, dtype='int16', channels=1,
                                   callback=self._audio_callback) as stream:

                while self._audio_stream_active and not self.transcript_ready_event.is_set():
                    try:
                        audio_chunk = self._audio_buffer.get(timeout=0.1)
                        if audio_chunk is None:
                            self._audio_stream_active = False; break
                        if self.dg_connection and self.dg_connection.websocket: # Check if websocket is connected
                            if self.dg_connection.send(audio_chunk) is False:
                                print("Deepgram STT (WebSocket): send returned false. Stopping audio send.")
                                self._audio_stream_active = False; break
                        elif not self.dg_connection or not self.dg_connection.websocket:
                            print("Deepgram STT (WebSocket): Connection not available for sending audio.")
                            self._audio_stream_active = False; break
                    except queue.Empty: continue

            self._audio_stream_active = False
            # Signal end of audio stream to Deepgram if connection is still open by sending empty bytes
            # This is not standard for Deepgram SDK; usually, just stopping send and calling finish() is enough.
            # if self.dg_connection and self.dg_connection.websocket:
            #    self.dg_connection.send(b'')

            timeout_seconds = 30
            if not self.transcript_ready_event.wait(timeout=timeout_seconds):
                print(f"Deepgram STT (WebSocket) Error: Timed out after {timeout_seconds}s waiting for transcript.")
                self.final_transcript = "ERROR_DEEPGRAM_TIMEOUT_WS"
                if self._dg_async_completion_event and not self._dg_async_completion_event.is_set():
                    self._dg_async_completion_event.set()

            if self.interim_transcript: print(f"\r{' ' * (len(self.interim_transcript) + 40)}\r", end='') # Clear last interim line
            print("Deepgram STT (WebSocket): Finished waiting for transcript_ready_event.")

        except sd.PortAudioError as pae:
            self.final_transcript = "ERROR_AUDIO_DEVICE_WS"
            print(f"STT Error (WebSocket): Mic issue: {pae}")
            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()
        except Exception as e:
            self.final_transcript = "ERROR_DEEPGRAM_UNEXPECTED_WS"
            print(f"STT Error (WebSocket): General setup/runtime: {e}")
            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set(): self._dg_async_completion_event.set()
        finally:
            self._audio_stream_active = False
            # Clear buffer and add sentinel for _audio_buffer.get() in audio sending loop if it were separate.
            # Since send loop is in this method, just clearing is fine.
            while not self._audio_buffer.empty():
                try: self._audio_buffer.get_nowait()
                except queue.Empty: break
            self._audio_buffer.put(None) # Ensure any get() unblocks if used elsewhere.

            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set():
                print("Deepgram STT (WebSocket): Forcing async completion from listen_and_transcribe finally.")
                self._dg_async_completion_event.set()

            if self._deepgram_thread and self._deepgram_thread.is_alive():
                print("Deepgram STT (WebSocket): Waiting for Deepgram thread to join...")
                self._deepgram_thread.join(timeout=5.0)
                if self._deepgram_thread.is_alive(): print("Deepgram STT (WebSocket) Warning: Thread did not join cleanly.")
            print(f"Deepgram STT (WebSocket): Exiting listen_and_transcribe. Final transcript: '{self.final_transcript}'")

        if not self.final_transcript.strip() or self.final_transcript.startswith("ERROR_"):
            # If an error occurred, self.final_transcript already contains the error message.
            print(f"Deepgram STT (WebSocket): Returning transcript indicating error or empty: '{self.final_transcript}'")
        return self.final_transcript.strip()


# --- TextToSpeechHandler class (Deepgram TTS) ---
# ... (TextToSpeechHandler remains unchanged) ...
# ... (if __name__ == '__main__' block remains unchanged) ...
# Note: The TextToSpeechHandler and if __name__ block are not being modified here as per subtask.
# Only SpeechToTextHandler is. The full file will be overwritten with these changes
# integrated into the existing full file content.
class TextToSpeechHandler:
    def __init__(self, client: DeepgramClient, model: str = "aura-asteria-en",
                 sample_rate: int = 24000, encoding: str = "linear16",
                 container: str = "none"):
        self.deepgram_client = client

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

        print(f"TTS Speaking (Deepgram SDK v4 - speak.rest.stream_memory): {text_to_speak[:60]}{'...' if len(text_to_speak) > 60 else ''}")

        source_payload = {"text": text_to_speak}

        try:
            response = await self.deepgram_client.speak.rest.v("1").stream_memory(
                source_payload,
                model=self.tts_model,
                encoding=self.tts_encoding,
                sample_rate=self.tts_sample_rate,
                container=self.tts_container
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
            print(f"Deepgram TTS Error in _speak_async (using stream_memory): {e}")

    def speak(self, text_to_speak: str):
        try:
            asyncio.run(self._speak_async(text_to_speak))
        except RuntimeError as re:
            if "cannot run event loop while another loop is running" in str(re) or \
               "Nesting asyncio event loops is not supported" in str(re):
                print(f"TTS Async Error: Could not run speak_async due to existing event loop: {re}. This can happen if speak() is called from an async context.")
            else:
                print(f"TTS Runtime Error: {re}")
        except Exception as e:
            print(f"Unexpected error in speak() method: {e}")

if __name__ == '__main__':
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
