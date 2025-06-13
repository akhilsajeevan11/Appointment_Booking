from appointment_system.agent import AppointmentAgent, CS_INITIAL_GREETING

def main():
    print("Welcome to the Appointment Booking System!")
    print("Type 'exit' to quit.\n")
    
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
    
    while True:
        user_input = input("How can I assist you? ").strip()
        
        if user_input.lower() == 'exit':
            print("Thank you for using the Appointment Booking System. Goodbye!")
            break
            
        if not user_input:
            print("Please provide a valid input.")
            continue
            
        # Add the user's message to the state
        state["messages"].append({"role": "user", "content": user_input})
        
        # Run the agent
        result = agent.invoke(state)
        
        # Update state with the result
        state = result
        
        # Extract the final answer from the last message
        if result["messages"]:
            last_message = result["messages"][-1]["content"]
            if "Final Answer:" in last_message:
                # Extract everything after "Final Answer:"
                final_answer = last_message.split("Final Answer:")[-1].strip()
                print(f"\nAgent: {final_answer}\n")
                
            elif "Action:" in last_message and "Observation:" in last_message:
                # This is a tool response, don't display it directly
                continue
            else:
                # This is a regular message, display it
                print(f"\nAgent: {last_message}\n")
        else:
            print("\nAgent: I'm not sure how to respond to that. Could you please rephrase?\n")

if __name__ == "__main__":
    main()
