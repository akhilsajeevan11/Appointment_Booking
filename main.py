import os # Add this import
from appointment_system.agent import AppointmentAgent, CS_INITIAL_GREETING
from appointment_system.voice_io import SpeechToTextHandler, TextToSpeechHandler

def main():
    print("Welcome to the Appointment Booking System!")
    print("Say 'exit' to quit.\n")

    # Initialize Deepgram API Key for STT
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        error_message = "Error: DEEPGRAM_API_KEY environment variable not set. This key is required for both Speech-to-Text and Text-to-Speech."
        print(error_message)
        # No STT or TTS available to speak this, as both depend on the key.
        return

    # Remove Piper TTS path retrieval and checks
    # piper_exe_path = os.environ.get("PIPER_EXE_PATH")
    # ... (lines for piper paths removed)
    # if not all([...]):
    # ... (check for piper paths removed)

    try:
        stt_handler = SpeechToTextHandler(deepgram_api_key=deepgram_api_key)
        # Initialize TextToSpeechHandler with Deepgram API Key
        tts_handler = TextToSpeechHandler(deepgram_api_key=deepgram_api_key)
    except Exception as e:
        print(f"Error initializing voice handlers (STT or TTS): {e}")
        return

    agent = AppointmentAgent().create_agent() # GOOGLE_API_KEY is used inside AppointmentAgent
    
    state = { # state initialization remains the same
        "messages": [], "next": "agent", "current_step": "",
        "booking_info": {"name": None, "date": None, "time": None, "purpose": None},
        "last_action": None, "action_count": 0, "conversation_state": CS_INITIAL_GREETING
    }
    
    initial_greeting = "Welcome to the voice-enabled appointment system, now powered by Deepgram. How can I help you today?"
    print(f"Agent: {initial_greeting}")
    tts_handler.speak(initial_greeting)

    while True:
        user_input = stt_handler.listen_and_transcribe().strip()

        if user_input.startswith("ERROR_"): # Catch all STT errors (Deepgram or audio device)
            error_message = f"Speech input error: {user_input}. Please try again."
            print(f"Agent: {error_message}")
            tts_handler.speak(error_message)
            if user_input == "ERROR_AUDIO_DEVICE": # If critical audio device error, might be best to exit
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
