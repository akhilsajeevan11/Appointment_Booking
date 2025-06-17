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
        # No TTS available yet to speak this error, as TTS init might also fail or depends on other env vars
        return

    # Initialize Piper TTS paths
    piper_exe_path = os.environ.get("PIPER_EXE_PATH")
    piper_model_onnx_path = os.environ.get("PIPER_MODEL_ONNX_PATH")
    piper_model_json_path = os.environ.get("PIPER_MODEL_JSON_PATH")

    if not all([piper_exe_path, piper_model_onnx_path, piper_model_json_path]):
        error_message = ("Error: Piper TTS environment variables (PIPER_EXE_PATH, "
                         "PIPER_MODEL_ONNX_PATH, PIPER_MODEL_JSON_PATH) not fully set. Cannot initialize Text-to-Speech.")
        print(error_message)
        # No TTS available to speak this error.
        return

    try:
        stt_handler = SpeechToTextHandler(deepgram_api_key=deepgram_api_key)
        tts_handler = TextToSpeechHandler(
            piper_exe_path=piper_exe_path,
            model_onnx_path=piper_model_onnx_path,
            model_json_path=piper_model_json_path
        )
    except Exception as e:
        print(f"Error initializing voice handlers: {e}")
        return

    agent = AppointmentAgent().create_agent()
    
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
    

    # Initial greeting by TTS
    initial_greeting = "Welcome to the voice-enabled appointment system. How can I help you today?"
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
