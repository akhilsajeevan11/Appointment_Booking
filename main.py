from appointment_system.agent import AppointmentAgent, CS_INITIAL_GREETING
# Corrected import: only one line for TextToSpeechHandler and SpeechToTextHandler
from appointment_system.voice_io import SpeechToTextHandler, TextToSpeechHandler

def main():
    print("Welcome to the Appointment Booking System!")
    print("Say 'exit' to quit.\n")
    
    agent = AppointmentAgent().create_agent()
    stt_handler = SpeechToTextHandler()
    tts_handler = TextToSpeechHandler()
    
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

        if user_input == "ERROR_AUDIO_DEVICE":
            error_message = "There seems to be an issue with your audio input device. Please check your microphone and ensure permissions are correct. Exiting."
            print(f"Agent: {error_message}")
            # No TTS here as audio output might also be affected.
            break
        elif user_input == "ERROR_STT_SERVICE":
            error_message = "Sorry, I'm having trouble with the speech recognition service right now. Please try again in a moment."
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
