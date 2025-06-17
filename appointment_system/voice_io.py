import sounddevice as sd
from deepgram import DeepgramClient, DeepgramClientOptions, LiveOptions
import asyncio
import threading
import os
import queue
import time
import numpy as np
import logging
from typing import Optional, Callable
from deepgram import Deepgram, DeepgramLiveConnection
from deepgram.exceptions import DeepgramApiError
import httpx
import json
# Removed subprocess, shutil, json, Path as they were for Piper TTS

# Configuration constants
SAMPLE_RATE = 16000 # This is for STT (Deepgram)
SD_CHUNK_SIZE = int(SAMPLE_RATE / 10) # 100ms for sounddevice RawInputStream with STT

class SpeechToTextHandler:
    def __init__(self, client: DeepgramClient): # Changed parameter
        self.deepgram_client = client # Use passed client
        # if not deepgram_api_key: # Removed API key check, client is now passed
        #     raise ValueError("Deepgram API key is required for SpeechToTextHandler.")
        # config = DeepgramClientOptions(options={"keepalive": "true"}) # Client created outside
        # self.deepgram_client = DeepgramClient(api_key=deepgram_api_key, config=config)

        self.final_transcript = ""
        self.interim_transcript = ""
        self._current_utterance_final_transcript = ""
        self.transcript_ready_event = threading.Event() # For sync between STT thread and main app thread
        self.dg_connection = None
        self._audio_stream_active = False
        self._audio_buffer = queue.Queue()
        self._deepgram_thread = None
        self._dg_async_completion_event = None # Will be an asyncio.Event, created in the thread


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
                        if self._dg_async_completion_event and not self._dg_async_completion_event.is_set():
                            self._dg_async_completion_event.set()
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
        if self._dg_async_completion_event and not self._dg_async_completion_event.is_set():
            self._dg_async_completion_event.set()

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
        if self._dg_async_completion_event and not self._dg_async_completion_event.is_set():
            self._dg_async_completion_event.set()

    async def _start_and_run_deepgram(self, options: LiveOptions, completion_event: asyncio.Event):
        self.dg_connection.on("open", self._on_open)
        self.dg_connection.on("transcript_received", self._on_message)
        self.dg_connection.on("error", self._on_error)
        self.dg_connection.on("close", self._on_close)

        print("Deepgram STT: Attempting to start connection with options...")
        try:
            start_status = self.dg_connection.start(options) # Synchronous call
            print(f"Deepgram STT: dg_connection.start() called. Returned status: {start_status}")

            if isinstance(start_status, bool) and not start_status:
                print("Deepgram STT Error: start() returned False. Connection failed to initialize properly.")
                await self._on_error({"message": "Connection start returned false"}, from_start_call=True)
                completion_event.set()
            else:
                print("Deepgram STT: Connection started, awaiting completion signal...")
                await completion_event.wait()
                print("Deepgram STT: Completion signal received.")

        except Exception as e:
            print(f"Deepgram STT Error: Exception during Deepgram start or while running: {e}")
            await self._on_error({"message": f"Exception in _start_and_run_deepgram: {e}"})
            completion_event.set() # Ensure completion event is set on error
        finally:
            print("Deepgram STT: _start_and_run_deepgram coroutine is finishing.")
        # The audio sending loop is removed from here and managed by the SDK or higher level logic if start is blocking
        # If start() is non-blocking and needs an explicit send loop, that would be different.
        # Based on the problem (await bool), start() is sync. The callbacks manage completion.


    def _run_deepgram_in_thread(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create the asyncio.Event within the loop it will be used in
            self._dg_async_completion_event = asyncio.Event()

            options = LiveOptions(
                model="nova-2", language="en-US", smart_format=True,
                encoding="linear16", sample_rate=SAMPLE_RATE, channels=1,
                interim_results=True, utterance_end_ms="1000",
            )
            # Pass the event to the async function
            loop.run_until_complete(self._start_and_run_deepgram(options, self._dg_async_completion_event))
        except Exception as e: # This is the correct handler for errors in loop.run_until_complete or _start_and_run_deepgram
            print(f"Critical error in Deepgram thread execution: {e}")
            self.final_transcript = "ERROR_DEEPGRAM_THREAD_CRASH"
            if not self.transcript_ready_event.is_set(): # Ensure main thread is signaled
                self.transcript_ready_event.set()
            # Also ensure the async completion event is set if the error happened before it was naturally set
            if self._dg_async_completion_event and not self._dg_async_completion_event.is_set():
                self._dg_async_completion_event.set()
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
    def __init__(self, client: DeepgramClient, model: str = "aura-asteria-en",
                 sample_rate: int = 24000, encoding: str = "linear16",
                 container: str = "none"): # Changed first parameter
        self.deepgram_client = client # Use passed client
        # if not deepgram_api_key: # Removed API key check
        #     raise ValueError("Deepgram API key is required for TextToSpeechHandler.")
        # client_config = DeepgramClientOptions(options={"keepalive": "true"}) # Client created outside
        # self.deepgram_client = DeepgramClient(api_key=deepgram_api_key, config=client_config)

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

        # Source dictionary should ONLY contain 'text'
        source_payload = {"text": text_to_speak}

        try:
            # Pass other options as direct keyword arguments
            response = await self.deepgram_client.speak.v("1").stream(
                source_payload,
                model=self.tts_model,
                encoding=self.tts_encoding,
                sample_rate=self.tts_sample_rate,
                container=self.tts_container
                # Add other valid keyword arguments for the stream method if needed
            )

            audio_stream = response.stream
            if audio_stream:
                chunk_size = 1024 * 4
                with sd.RawOutputStream(samplerate=self.tts_sample_rate,
                                        channels=1,
                                        dtype='int16',
                                        device=None) as stream_player: # Ensure sd is imported
                    while True:
                        chunk = await audio_stream.read(chunk_size)
                        if not chunk:
                            break
                        stream_player.write(chunk)
                # print("Deepgram TTS: Finished playing audio stream.") # Optional debug
            else:
                print("Deepgram TTS Error: Failed to obtain audio stream from response.")

        except sd.PortAudioError as pae:
            print(f"TTS Playback Error (PortAudioError with Deepgram TTS): {pae}.")
        except Exception as e:
            print(f"Deepgram TTS Error in _speak_async: {e}")
            else:
                print(f"Deepgram TTS Error: {e}")

    async def speak(self, text: str):
        """Speak the given text using Deepgram TTS."""
        try:
            print(f"TTS Speaking (Deepgram): {text[:50]}...")
            
            # Prepare the request payload
            payload = {
                "text": text,
                "model": "aura-asteria-en",
                "encoding": "mp3",
                "container": "mp3",
                "sample_rate": 24000
            }
            
            # Make the API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.deepgram.com/v1/speak",
                    headers={
                        "Authorization": f"Token {self.deepgram_client.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    # Save the audio to a temporary file
                    temp_file = "temp_tts.mp3"
                    with open(temp_file, "wb") as f:
                        f.write(response.content)
                    
                    # Play the audio
                    self._play_audio(temp_file)
                    
                    # Clean up
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                else:
                    print(f"Deepgram TTS Error: {response.text}")
                    
        except Exception as e:
            print(f"Deepgram TTS Error: {e}")

if __name__ == '__main__':
    print("--- Voice I/O Module Test ---")

    # Test TextToSpeechHandler (Deepgram TTS)
    print("\n--- Testing TextToSpeechHandler (Deepgram TTS) ---")
    dg_api_key_env = os.environ.get("DEEPGRAM_API_KEY")

    if not dg_api_key_env:
        print("CRITICAL ERROR: DEEPGRAM_API_KEY environment variable not set. Cannot run STT or TTS tests.")
    else:
        print(f"Using DEEPGRAM_API_KEY: ...{dg_api_key_env[-4:] if len(dg_api_key_env) > 4 else '...key_is_short'}")

        # Create a single DeepgramClient instance for tests
        client_config = DeepgramClientOptions(options={"keepalive": "true"})
        shared_deepgram_client = DeepgramClient(api_key=dg_api_key_env, config=client_config)

        # Test TextToSpeechHandler (Deepgram TTS)
        print("\n--- Testing TextToSpeechHandler (Deepgram TTS) ---")
        try:
            tts_handler = TextToSpeechHandler(client=shared_deepgram_client, model="aura-asteria-en", sample_rate=24000)
            tts_handler.speak("Hello from Deepgram Text to Speech, using the Aura model.")
            tts_handler.speak("This audio is being streamed directly to your speakers.")
        except Exception as e:
            print(f"Error during Deepgram TTS test: {e}")
        print("--- Finished Deepgram TTS Test ---")

        # Test SpeechToTextHandler (Deepgram STT)
        print("\n--- Testing SpeechToTextHandler (Deepgram STT) ---")
        try:
            stt_handler = SpeechToTextHandler(client=shared_deepgram_client)
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
