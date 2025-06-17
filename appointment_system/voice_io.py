import sounddevice as sd
# from google.cloud import texttospeech # Removed
from deepgram import DeepgramClient, DeepgramClientOptions, LiveOptions
import asyncio
import threading
import os
import queue
import subprocess # Added
import shutil   # Added
import json     # Added
from pathlib import Path # Added

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
        error_message_str = "Unknown Deepgram Error"
        if isinstance(error, dict) and 'message' in error:
            error_message_str = error['message']
        elif isinstance(error, Exception):
            error_message_str = str(error)
        elif isinstance(error, str):
            error_message_str = error
        else:
            error_message_str = str(error)
        print(f"Deepgram Error: {error_message_str}")
        if not self.final_transcript or self.final_transcript.startswith("ERROR_DEEPGRAM_STT: Unknown Deepgram Error") or from_start_call:
            if self._current_utterance_final_transcript and not from_start_call:
                self.final_transcript = self._current_utterance_final_transcript.strip() + f" (ERROR_DEEPGRAM_STT: {error_message_str})"
            else:
                self.final_transcript = f"ERROR_DEEPGRAM_STT: {error_message_str}"
        self._current_utterance_final_transcript = ""
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
        print("Deepgram: Starting connection with options...")
        try:
            connection_status = self.dg_connection.start(options)
            print(f"Deepgram: dg_connection.start() called. Status/Result (if any): {connection_status}")
            if isinstance(connection_status, bool) and not connection_status:
                print("Deepgram: start() returned False. Connection might have failed to initialize properly.")
                await self._on_error({"message": "Connection start returned false"}, from_start_call=True)
                return
        except Exception as e:
            print(f"Exception during Deepgram start: {e}")
            await self._on_error({"message": f"Exception in _start_and_run_deepgram during start: {e}"}, from_start_call=True)
            return
        try:
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

# --- TextToSpeechHandler class (Piper TTS) ---
class TextToSpeechHandler:
    def __init__(self, piper_exe_path: str, model_onnx_path: str, model_json_path: str):
        self.piper_exe_path = shutil.which(piper_exe_path) or piper_exe_path
        if not shutil.which(self.piper_exe_path):
            raise FileNotFoundError(f"Piper executable not found at '{self.piper_exe_path}' or in PATH.")

        self.model_path = Path(model_onnx_path)
        self.model_config_path = Path(model_json_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Piper model (.onnx) not found at '{self.model_path}'")
        if not self.model_config_path.is_file():
            raise FileNotFoundError(f"Piper model config (.json) not found at '{self.model_config_path}'")

        self.sample_rate_hertz = self._load_sample_rate_from_config()

    def _load_sample_rate_from_config(self) -> int:
        try:
            with open(self.model_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            sample_rate = int(config_data.get("audio", {}).get("sample_rate", 22050))
            print(f"Piper TTS: Loaded sample rate {sample_rate} Hz from {self.model_config_path}")
            return sample_rate
        except Exception as e:
            print(f"Piper TTS: Error loading sample rate from '{self.model_config_path}': {e}. Using default 22050 Hz.")
            return 22050

    def speak(self, text_to_speak):
        if not text_to_speak:
            print("TTS (Piper): No text to speak.")
            return

        print(f"TTS Speaking (Piper): {text_to_speak[:60]}{'...' if len(text_to_speak) > 60 else ''}")

        command = [
            str(self.piper_exe_path),
            '--model', str(self.model_path),
            '--config', str(self.model_config_path),
            '--output-raw'
        ]

        process = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.stdin:
                process.stdin.write(text_to_speak.encode('utf-8'))
                process.stdin.close()

            with sd.RawOutputStream(samplerate=self.sample_rate_hertz,
                                    channels=1, dtype='int16', device=None) as stream:
                while True:
                    audio_chunk = process.stdout.read(1024)
                    if not audio_chunk:
                        break
                    stream.write(audio_chunk)

            stderr_data_bytes = process.stderr.read()
            process.wait()

            if process.returncode != 0:
                print(f"Piper TTS Error: Exit code {process.returncode}")
                if stderr_data_bytes:
                    print(f"Piper STDERR: {stderr_data_bytes.decode('utf-8', errors='ignore')}")
            elif stderr_data_bytes:
                print(f"Piper STDERR (info/warnings): {stderr_data_bytes.decode('utf-8', errors='ignore')}")

        except FileNotFoundError:
            print(f"Piper TTS Error: Executable not found at '{self.piper_exe_path}'.")
        except sd.PortAudioError as pae:
            print(f"TTS Playback Error (PortAudioError with Piper): {pae}.")
        except Exception as e:
            print(f"General Piper TTS Error: {e}")
        finally:
            if process and process.poll() is None:
                print("Piper TTS: Terminating Piper process.")
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except Exception as e_term:
                    print(f"Piper TTS: Error during terminate/wait: {e_term}")
                    try:
                        process.kill()
                        process.wait(timeout=2.0)
                    except Exception as e_kill:
                         print(f"Piper TTS: Error during kill/wait: {e_kill}")

if __name__ == '__main__':
    print("--- Voice I/O Module Test ---")

    # Test TextToSpeechHandler (Piper TTS)
    print("\n--- Testing TextToSpeechHandler (Piper TTS) ---")
    piper_exe = os.environ.get("PIPER_EXE_PATH")
    piper_onnx = os.environ.get("PIPER_MODEL_ONNX_PATH")
    piper_json = os.environ.get("PIPER_MODEL_JSON_PATH")

    if not all([piper_exe, piper_onnx, piper_json]):
        print("Skipping Piper TTS test: One or more environment variables not set:")
        print("  PIPER_EXE_PATH, PIPER_MODEL_ONNX_PATH, PIPER_MODEL_JSON_PATH")
    else:
        try:
            print(f"Piper paths: EXE='{piper_exe}', ONNX='{piper_onnx}', JSON='{piper_json}'")
            tts_handler = TextToSpeechHandler(
                piper_exe_path=piper_exe,
                model_onnx_path=piper_onnx,
                model_json_path=piper_json
            )
            tts_handler.speak("Hello, this is a test of Piper Text to Speech.")
            tts_handler.speak("Audio should be generated locally and streamed for playback.")
        except FileNotFoundError as fnf:
            print(f"Piper TTS FileNotFoundError: {fnf}. Ensure paths are correct and Piper executable has permissions.")
        except Exception as e:
            print(f"Error during Piper TTS test: {e}")
    print("--- Finished Piper TTS Test ---")

    # Test SpeechToTextHandler (Deepgram STT)
    print("\n--- Testing SpeechToTextHandler (Deepgram STT) ---")
    deepgram_api_key_env = os.environ.get("DEEPGRAM_API_KEY") # Renamed to avoid conflict
    if not deepgram_api_key_env:
        print("Skipping Deepgram STT test: DEEPGRAM_API_KEY environment variable not set.")
    else:
        try:
            stt_handler = SpeechToTextHandler(deepgram_api_key=deepgram_api_key_env)
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
            print(f"Error during Deepgram STT test setup or execution: {e}")
    print("--- Finished Deepgram STT Test ---")

    print("\n--- Voice I/O Module Test Complete ---")
