import os # Add this import
from appointment_system.agent import AppointmentAgent, CS_INITIAL_GREETING
from appointment_system.voice_io import SpeechToTextHandler, TextToSpeechHandler

def main():
    print("Welcome to the Appointment Booking System!")
    print("Say 'exit' to quit.\n")

    # Initialize Deepgram API Key for STT
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        error_message = "Error: DEEPGRAM_API_KEY environment variable not set. Cannot initialize Speech-to-Text."
        print(error_message)
        try:
            # Attempt to use TTS for this critical startup error
            tts_handler_startup_error = TextToSpeechHandler()
            tts_handler_startup_error.speak(error_message + " Please set the key and restart.")
        except Exception as tts_init_error:
            print(f"TTS handler could not be initialized to speak the error: {tts_init_error}")
        return # Exit if Deepgram key is missing

    agent = AppointmentAgent().create_agent()
    stt_handler = SpeechToTextHandler(deepgram_api_key=deepgram_api_key)
    tts_handler = TextToSpeechHandler() # Google TTS remains, uses ADC
    
    # Initialize state
    state = {
        "messages": [],
        "next": "agent",
        "current_step": "",
        "booking_info": {
            "name": None,
            "date": None,
            "time": None,
            "purpose": None
        },
        "last_action": None,
        "action_count": 0,
        "conversation_state": CS_INITIAL_GREETING
    }
    
    # Removed duplicated stt_handler instantiation
    while True:
        user_input = stt_handler.listen_and_transcribe().strip()

        # Check for specific error strings from the STT handler
        if user_input == "ERROR_AUDIO_DEVICE":
            error_message = "There seems to be an issue with your audio input device. Please check your microphone and ensure permissions are correct. Exiting."
            print(f"Agent: {error_message}")
            # No TTS here as audio input itself failed.
            break
        elif user_input.startswith("ERROR_DEEPGRAM_"): # Catch all Deepgram specific errors
            error_message = f"Sorry, I'm having trouble with the speech recognition service ({user_input}). Please try again in a moment."
            print(f"Agent: {error_message}")
            tts_handler.speak(error_message)
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
