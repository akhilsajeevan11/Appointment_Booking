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
        
        # New logic for determining text_to_speak_and_print
        text_to_speak_and_print = None
        agent_llm_output_step = result.get("current_step", "")
        last_message_from_history = result.get("messages", [])[-1] if result.get("messages") else None

        # First, check if the last message in history is a tool call/observation sequence.
        # This existing logic with `continue` should take precedence if it's a tool step.
        if last_message_from_history and isinstance(last_message_from_history.get("content"), str):
            last_content = last_message_from_history.get("content")
            if "Action:" in last_content and "Action Input:" in last_content and "Observation:" in last_content:
                print(f"\nAgent (tool observation - not spoken): {last_content}\n")
                continue # Skip speaking and go to next user input or agent cycle

        # If not a tool observation to be skipped, then determine response:
        # Priority 1: "Final Answer:" in current_step (direct output from LLM in the agent graph)
        if isinstance(agent_llm_output_step, str) and "Final Answer:" in agent_llm_output_step:
            final_answer_text = agent_llm_output_step.split("Final Answer:", 1)[-1].strip()
            if final_answer_text: # Ensure it's not empty after stripping
                text_to_speak_and_print = final_answer_text

        # Priority 2: Fallback if no "Final Answer:" in current_step.
        # Check the last message in history if it's from the assistant and not what we already parsed.
        if not text_to_speak_and_print and last_message_from_history:
            if last_message_from_history.get("role") == "assistant":
                content = last_message_from_history.get("content", "")
                # Avoid using the raw current_step content again if it was already considered and didn't have "Final Answer:"
                if content != agent_llm_output_step:
                    if "Final Answer:" in content: # Agent might put "Final Answer:" in message history too
                        final_answer_text_from_msg = content.split("Final Answer:", 1)[-1].strip()
                        if final_answer_text_from_msg:
                             text_to_speak_and_print = final_answer_text_from_msg
                    else: # A simple assistant message without the prefix
                         text_to_speak_and_print = content.strip()

        # Final Fallback: If no specific agent output identified
        if not text_to_speak_and_print:
            text_to_speak_and_print = "I'm not sure how to respond to that. Could you please rephrase?"

        # Output the determined response
        print(f"\nAgent: {text_to_speak_and_print}\n")
        tts_handler.speak(text_to_speak_and_print)

if __name__ == "__main__":
    main()
