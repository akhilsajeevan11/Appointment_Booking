import sounddevice as sd # Make sure sd is imported
from google.cloud import speech, texttospeech
import queue
import os

# Configuration constants
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10) # 100ms for STT

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
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break

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
                    if not self._audio_stream_active and not response.results :
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
                        final_transcript += transcript
                        print(f"STT Final: {final_transcript}")
                        self._audio_stream_active = False
                        self._buffer.put(None)
                        break
                    else:
                        if displayed_interim_transcript:
                            print(f"\r{' ' * len(displayed_interim_transcript)}\r", end='')
                        interim_display = f"STT Interim: {transcript}"
                        print(interim_display, end='')
                        displayed_interim_transcript = interim_display

            if displayed_interim_transcript:
                 print(f"\r{' ' * len(displayed_interim_transcript)}\r", end='')

        except sd.PortAudioError as pae:
            print(f"STT Error: Microphone/audio device issue: {pae}")
            self._audio_stream_active = False
            if hasattr(self, '_buffer') and isinstance(self._buffer, queue.Queue'):
                 self._buffer.put(None)
            return "ERROR_AUDIO_DEVICE"
        except Exception as e:
            print(f"STT Error: General STT service error: {e}")
            self._audio_stream_active = False
            if hasattr(self, '_buffer') and isinstance(self._buffer, queue.Queue'):
                 self._buffer.put(None)
            return "ERROR_STT_SERVICE"
        finally:
            if self._audio_stream_active:
                 self._audio_stream_active = False
                 if hasattr(self, '_buffer') and isinstance(self._buffer, queue.Queue'):
                    self._buffer.put(None)

        if not final_transcript:
            print("No speech detected or transcribed.")
            return ""

        return final_transcript.strip()

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
            # effects_profile_id=['telephony-class-application'] # Optional
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

                # print("TTS: Stream opened. Receiving and playing audio chunks...") # Debug
                for response_chunk in streaming_responses:
                    if response_chunk.audio_content:
                        stream.write(response_chunk.audio_content)
                # print("TTS: Finished receiving and playing audio.") # Debug

        except sd.PortAudioError as pae:
            print(f"TTS Playback Error (PortAudioError): {pae}. Check your audio output device and configuration.")
        except Exception as e:
            print(f"TTS Streaming or Playback Error: {e}")

if __name__ == '__main__':
    # STT Test (optional, can be commented out if focusing on TTS)
    # stt_handler = SpeechToTextHandler()
    # try:
    #     while True: # Keep prompting for STT until Ctrl+C
    #         print("\nPress Enter to start STT, then speak. Ctrl+C to exit STT test loop.")
    #         input()
    #         text = stt_handler.listen_and_transcribe()
    #         if text == "ERROR_AUDIO_DEVICE":
    #             print("STT Test: Audio device error. Cannot perform test.")
    #             break # Exit STT loop on audio device error
    #         elif text == "ERROR_STT_SERVICE":
    #             print("STT Test: STT service error.")
    #         elif text:
    #             print(f"--- You said: {text} ---")
    #         else:
    #             print("--- No STT transcription ---")
    # except KeyboardInterrupt:
    #     print("\nExiting STT test.")
    # except Exception as e:
    #     print(f"Error during STT test: {e}")
    # finally:
    #     print("--- End of STT Test Section ---")


    # TTS Test
    print("\n--- TTS Streaming Test ---")
    tts_handler = TextToSpeechHandler()
    tts_handler.speak("Hello, this is a test of the new streaming Text to Speech system.")
    tts_handler.speak("Audio should start playing almost immediately, and not wait for the full sentence to be processed.")
    tts_handler.speak("Let's try a slightly longer sentence to see how the streaming performs with more content. This should demonstrate the audio starting while the rest is still being synthesized and sent over.")
    print("--- End of TTS Test ---")
