from appointment_system.agent import AppointmentAgent

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
        }
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
                
                # Update booking info if provided in the response
                if "name:" in final_answer.lower():
                    state["booking_info"]["name"] = final_answer.split("name:")[-1].split(",")[0].strip()
                if "date:" in final_answer.lower():
                    state["booking_info"]["date"] = final_answer.split("date:")[-1].split(",")[0].strip()
                if "time:" in final_answer.lower():
                    state["booking_info"]["time"] = final_answer.split("time:")[-1].split(",")[0].strip()
                if "purpose:" in final_answer.lower():
                    state["booking_info"]["purpose"] = final_answer.split("purpose:")[-1].split(".")[0].strip()
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
