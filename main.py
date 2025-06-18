import os # Keep for GOOGLE_API_KEY and potentially Whisper model paths later
from appointment_system.agent import AppointmentAgent, CS_INITIAL_GREETING
from appointment_system.voice_io import SpeechToTextHandler, TextToSpeechHandler # Imports remain
# Remove Deepgram specific imports if no longer used by any handler
# from deepgram import DeepgramClient, DeepgramClientOptions

def main():
    print("Welcome to the Appointment Booking System!")
    print("Say 'exit' to quit.\n")

    # GOOGLE_API_KEY for LangChain agent is still loaded by dotenv in agent.py or main.py if load_dotenv is called here
    # No DEEPGRAM_API_KEY needed for Whisper STT or gTTS.
    # No DeepgramClient needed.

    try:
        # Initialize Whisper STT Handler (defaults to "small.en", lang="en")
        stt_handler = SpeechToTextHandler()

        # Initialize gTTS Handler (defaults to lang="en")
        tts_handler = TextToSpeechHandler()

        # Check if Whisper model loaded successfully (stt_handler.model will be None if failed)
        if stt_handler.model is None:
            # This error is already printed by SpeechToTextHandler.__init__
            print("Critical: Whisper STT model failed to load. Exiting.")
            # tts_handler.speak("Error: The speech recognition model could not be loaded. The application cannot start.") # Optional TTS feedback
            return

    except Exception as e:
        # This will catch errors if the classes themselves can't be instantiated for other reasons
        # or if sounddevice/audio backend issues occur at a very basic level.
        print(f"Critical Error: Failed to initialize voice handlers: {e}")
        return

    agent = AppointmentAgent().create_agent() # This needs GOOGLE_API_KEY from .env
    
    state = {
        "messages": [], "next": "agent", "current_step": "",
        "booking_info": {"name": None, "date": None, "time": None, "purpose": None},
        "last_action": None, "action_count": 0, "conversation_state": CS_INITIAL_GREETING
    }
    
    initial_greeting = "Welcome to the voice-enabled appointment system. How can I help you today?"
    print(f"Agent: {initial_greeting}")
    tts_handler.speak(initial_greeting) # Test gTTS

    while True:
        raw_stt_result = stt_handler.listen_and_transcribe()
        user_input = raw_stt_result.strip()

        if user_input.startswith("ERROR_WHISPER_") or user_input == "ERROR_AUDIO_DEVICE":
            error_message = f"Speech input error: {user_input}. Please try again."
            print(f"Agent: {error_message}")
            tts_handler.speak(error_message)
            if user_input == "ERROR_AUDIO_DEVICE":
                print("Exiting due to critical audio device error.")
                break
            continue

        if not user_input:
            no_input_message = "I didn't catch that. Could you please say it again?"
            print(f"Agent: {no_input_message}")
            tts_handler.speak(no_input_message)
            continue
        
        print(f"You said: {user_input}")

        if user_input.lower() == 'exit':
            exit_message = "Thank you for using the Appointment Booking System. Goodbye!"
            print(exit_message)
            tts_handler.speak(exit_message)
            break
            
        state["messages"].append({"role": "user", "content": user_input})
        
        result = agent.invoke(state)
        state = result
        
        agent_response_text = None
        if result["messages"]:
            last_message_content = result["messages"][-1]["content"]
            if "Final Answer:" in last_message_content:
                final_answer = last_message_content.split("Final Answer:")[-1].strip()
                agent_response_text = final_answer
                print(f"\nAgent: {final_answer}\n")
                
            elif "Action:" in last_message_content and "Observation:" in last_message_content:
                print(f"\nAgent (tool observation - not spoken): {last_message_content}\n")
                continue
            else:
                agent_response_text = last_message_content
                print(f"\nAgent: {last_message_content}\n")
        else:
            agent_response_text = "I'm not sure how to respond to that. Could you please rephrase?"
            print(f"\nAgent: {agent_response_text}\n")

        if agent_response_text:
            tts_handler.speak(agent_response_text)

if __name__ == "__main__":
    main()
