import sounddevice as sd
from google.cloud import texttospeech # Keep for TTS
# Remove LiveTranscriptionEvents if no longer used, keep others
from deepgram import DeepgramClient, DeepgramClientOptions, LiveOptions
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
        self.interim_transcript = ""
        self._current_utterance_final_transcript = "" # Accumulates final segments for the current utterance
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
            # Assuming 'result' is ListenLiveResponse from deepgram-sdk v0.16+
            transcript = result.channel.alternatives[0].transcript
            if transcript: # Only process if there's a transcript
                if result.is_final: # This is a final segment of speech
                    self._current_utterance_final_transcript += transcript + " "

                    # Clear previous interim/update line before printing new one
                    if self.interim_transcript:
                        print(f"\r{' ' * (len(self.interim_transcript) + 30)}\r", end='') # +30 for "STT Interim/Update (Deepgram): "

                    if result.speech_final: # End of the entire utterance
                        self.final_transcript = self._current_utterance_final_transcript.strip()
                        self._current_utterance_final_transcript = "" # Reset for next utterance
                        print(f"STT Final (Deepgram): {self.final_transcript}")
                        self.interim_transcript = "" # Clear any interim display
                        self.transcript_ready_event.set() # Signal main thread for final transcript
                    else:
                        # It's a final segment, but not the end of the whole speech.
                        # Update display with this more solid part.
                        # self.interim_transcript = self._current_utterance_final_transcript.strip() # Keep this as interim
                        # For more dynamic display, show accumulating final segments as "Update"
                        print(f"STT Update (Deepgram): {self._current_utterance_final_transcript.strip()}", end='')
                        self.interim_transcript = f"STT Update (Deepgram): {self._current_utterance_final_transcript.strip()}"


                else: # This is an interim result for the current segment
                    # Clear previous interim line
                    if self.interim_transcript:
                        print(f"\r{' ' * (len(self.interim_transcript) + 30)}\r", end='')

                    # Display current full interim (accumulated finals + current interim segment)
                    current_full_interim = self._current_utterance_final_transcript + transcript
                    self.interim_transcript = f"STT Interim (Deepgram): {current_full_interim.strip()}"
                    print(self.interim_transcript, end='')

        except Exception as e:
            print(f"Error processing Deepgram message: {e} - Result: {result}")
            # Potentially set error state or log more formally

    async def _on_error(self, error, **kwargs):
        error_message = str(error)
        if isinstance(error, dict) and 'message' in error:
             error_message = error['message']
        print(f"Deepgram Error: {error_message}")

        # Preserve any partial transcript before error
        if self._current_utterance_final_transcript:
            self.final_transcript = self._current_utterance_final_transcript.strip() + f" (ERROR_DEEPGRAM_STT: {error_message})"
        else:
            self.final_transcript = f"ERROR_DEEPGRAM_STT: {error_message}"

        self._current_utterance_final_transcript = "" # Clear accumulator
        if not self.transcript_ready_event.is_set():
            self.transcript_ready_event.set()

    async def _on_close(self, close, **kwargs):
        print(f"Deepgram Connection Closed: {close}")
        if not self.transcript_ready_event.is_set():
            if self._current_utterance_final_transcript: # If connection closes mid-utterance
                self.final_transcript = self._current_utterance_final_transcript.strip()
                print(f"STT Final (Deepgram - on close): {self.final_transcript}")
            elif not self.final_transcript: # If no final transcript was set by error or speech_final
                 self.final_transcript = "ERROR_DEEPGRAM_CLOSED"
            self._current_utterance_final_transcript = ""
            self.transcript_ready_event.set() # Ensure main thread unblocks

    async def _start_and_run_deepgram(self, options):
        # Assign handlers using string literals
        self.dg_connection.on("open", self._on_open)
        self.dg_connection.on("transcript_received", self._on_message)
        self.dg_connection.on("error", self._on_error)
        self.dg_connection.on("close", self._on_close)

        print("Deepgram: Starting connection with options...")
        if not await self.dg_connection.start(options):
            print("Deepgram: Failed to start connection.")
            self.final_transcript = "ERROR_DEEPGRAM_START_FAILED"
            self.transcript_ready_event.set()
            return

        # Audio sending loop
        while self._audio_stream_active:
            try:
                # Using run_in_executor for blocking queue.get in async code
                audio_chunk = await asyncio.get_event_loop().run_in_executor(None, self._audio_buffer.get, True, 0.1)
                if audio_chunk is None: # Sentinel
                    print("Deepgram: Sentinel received, stopping audio sending.")
                    break
                if not self.dg_connection.send(audio_chunk):
                    print("Deepgram: Failed to send audio, connection might be closing.")
                    self._audio_stream_active = False
                    break
            except queue.Empty: # Expected when no audio, continue polling
                if not self._audio_stream_active: break # Exit if stream became inactive
                continue
            except Exception as e:
                print(f"Deepgram: Error in audio sending loop: {e}")
                self._audio_stream_active = False
                break

        print("Deepgram: Audio sending loop finished.")
        # Ensure connection finish is called if loop terminates before on_close from server
        if self.dg_connection and self.dg_connection.is_connected():
            print("Deepgram: Proactively finishing connection from client side.")
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

                # Main loop for this method is just waiting for the event or timeout
                # Audio is captured by sounddevice's thread and sent by Deepgram's thread
                timeout_seconds = 30
                if not self.transcript_ready_event.wait(timeout=timeout_seconds):
                    print("Deepgram STT: Timed out waiting for transcript.")
                    if not self.final_transcript:
                        self.final_transcript = "ERROR_DEEPGRAM_TIMEOUT"
                    # Ensure audio stream stops on timeout
                    self._audio_stream_active = False
                    self._audio_buffer.put(None) # Signal audio sending loop

            # After stream closes or timeout, ensure the event is set if thread is still running
            if self._deepgram_thread and self._deepgram_thread.is_alive() and not self.transcript_ready_event.is_set():
                print("Deepgram STT: Forcing event set after timeout/stream closure.")
                self.transcript_ready_event.set() # This might be too late or abrupt for the thread.

            # Clean up any final interim display
            if self.interim_transcript:
                print(f"\r{' ' * (len(self.interim_transcript) + 30)}\r", end='')

        except sd.PortAudioError as pae:
            print(f"STT Error: Microphone/audio device issue: {pae}")
            self.final_transcript = "ERROR_AUDIO_DEVICE"
            self._audio_stream_active = False
            self._audio_buffer.put(None)
            if self.dg_connection and self.dg_connection.is_connected(): # Try to signal close
                 asyncio.run(self.dg_connection.finish()) # This is tricky from non-async context
        except Exception as e:
            print(f"STT Error: General Deepgram setup/runtime error: {e}")
            if not self.final_transcript: # Avoid overwriting specific error
                 self.final_transcript = "ERROR_DEEPGRAM_UNEXPECTED"
            self._audio_stream_active = False
            self._audio_buffer.put(None)
        finally:
            self._audio_stream_active = False
            # Ensure buffer is cleared and sentinel added
            while not self._audio_buffer.empty():
                try: self._audio_buffer.get_nowait()
                except queue.Empty: break
            self._audio_buffer.put(None)

            if self._deepgram_thread and self._deepgram_thread.is_alive():
                print("Deepgram STT: Waiting for Deepgram thread to join...")
                # The thread should ideally exit on its own when dg_connection.finish() is called
                # or its audio sending loop terminates.
                self.transcript_ready_event.set() # Ensure it's set so thread can exit if waiting on something
                self._deepgram_thread.join(timeout=3.0)
                if self._deepgram_thread.is_alive():
                    print("Deepgram STT: Warning - Deepgram thread did not join cleanly.")

            print(f"Deepgram STT: listen_and_transcribe finished. Final transcript: '{self.final_transcript}'")

        # Return only if it's not an error string or empty
        if self.final_transcript.startswith("ERROR_"):
            return self.final_transcript.strip()
        elif not self.final_transcript.strip():
            return "" # Return empty for no speech

        return self.final_transcript.strip()


# --- TextToSpeechHandler class ---
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
        self.audio_config_for_tts = texttospeech.AudioConfig( # Renamed
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hertz
        )

    def speak(self, text_to_speak):
        if not text_to_speak:
            print("TTS: No text to speak.")
            return

        print(f"TTS Speaking (streaming): {text_to_speak[:60]}{'...' if len(text_to_speak) > 60 else ''}")

        try:
            # Correctly prepare StreamingSynthesizeConfig
            streaming_config_obj = texttospeech.StreamingSynthesizeConfig()
            streaming_config_obj.audio_config = self.audio_config_for_tts # Set attribute
            streaming_config_obj.voice = self.voice_params           # Set attribute

            # First request: configuration
            request_config = texttospeech.StreamingSynthesizeRequest(
                streaming_config=streaming_config_obj # Use the configured object
            )

            # Second request: text input
            synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
            request_text = texttospeech.StreamingSynthesizeRequest(
                synthesis_input=synthesis_input
            )
            requests_iterable = [request_config, request_text]

            streaming_responses = self.client.streaming_synthesize(requests=requests_iterable)

            with sd.RawOutputStream(samplerate=self.sample_rate_hertz,
                                    channels=1,
                                    dtype='int16',
                                    device=None) as stream: # Explicitly set device=None for default
                for response_chunk in streaming_responses:
                    if response_chunk.audio_content:
                        stream.write(response_chunk.audio_content)
        except sd.PortAudioError as pae:
            print(f"TTS Playback Error (PortAudioError): {pae}. Check your audio output device and configuration.")
        except Exception as e:
            print(f"TTS Streaming or Playback Error: {e}")

if __name__ == '__main__':
    print("--- Voice I/O Module Test ---")

    # Test TextToSpeechHandler (Google Cloud TTS)
    print("\n--- Testing TextToSpeechHandler (Google Cloud TTS - Streaming) ---")
    try:
        tts_handler = TextToSpeechHandler()
        tts_handler.speak("Hello, this is a test of the streaming Text to Speech system using Google Cloud.")
        tts_handler.speak("Audio should start playing almost immediately.")
        # tts_handler.speak("Let's try a slightly longer sentence to see how the streaming performs with more content. This should demonstrate the audio starting while the rest is still being synthesized and sent over.")
    except Exception as e:
        print(f"Error during TTS test: {e}")
    print("--- Finished TTS Test ---")

    # Test SpeechToTextHandler (Deepgram STT)
    print("\n--- Testing SpeechToTextHandler (Deepgram STT) ---")
    try:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            print("DEEPGRAM_API_KEY environment variable not set. Skipping Deepgram STT test.")
        else:
            stt_handler = SpeechToTextHandler(deepgram_api_key=api_key)
            for i in range(2): # Allow a couple of attempts
                print(f"\nSTT Attempt {i+1}/2. Press Enter to start speaking, then speak. (Ctrl+C to skip remaining STT tests)")
                input()
                text = stt_handler.listen_and_transcribe()
                if text and not text.startswith("ERROR_"):
                    print(f"--- You said (Deepgram): {text} ---")
                else:
                    print(f"--- No valid transcription from Deepgram. Result: {text} ---")
    except KeyboardInterrupt:
        print("\nSkipped remaining STT tests.")
    except Exception as e:
        print(f"Error during STT test setup or execution: {e}")
    print("--- Finished STT Test ---")

    print("\n--- Voice I/O Module Test Complete ---")
