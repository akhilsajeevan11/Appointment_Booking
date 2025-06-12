from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AgentAction, AgentFinish
from langgraph.graph import StateGraph, END
from .tools import AppointmentTools
from dotenv import load_dotenv
import os
from typing import List, Union, Dict, Any, TypedDict
import re
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    next: str
    current_step: str
    booking_info: Dict[str, Any]
    last_action: str
    action_count: int

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs):
        # Add the tools to the prompt
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        tool_names = [tool.name for tool in self.tools]
        
        # Format the template with all required variables
        return self.template.format(
            tools=tools_str,
            tool_names=tool_names,
            **kwargs
        )

class AppointmentAgent:
    def __init__(self):
        self.tools_handler = AppointmentTools()
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        self.booking_info = {
            "name": None,
            "date": None,
            "time": None,
            "purpose": None
        }
        
    def create_agent(self):
        # Get the tools
        tools = [Tool(**tool) for tool in self.tools_handler.get_tools()]
        logger.info(f"Available tools: {[tool.name for tool in tools]}")
        tool_map = {tool.name: tool for tool in tools}
        
        # Set up the prompt template
        template = """
        You are a friendly and helpful appointment booking assistant. You have access to the following tools:
        
        {tools}
        
        Current booking information:
        Name: {name}
        Date: {date}
        Time: {time}
        Purpose: {purpose}
        
        Last action taken: {last_action}
        Number of consecutive actions: {action_count}
        Has all required information: {has_all_info}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (use JSON format for BookAppointment)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        For booking appointments, use this JSON format:
        {{"name": "person name", "date": "YYYY-MM-DD", "time": "HH:MM", "purpose": "appointment purpose"}}
        
        Important Instructions:
        1. Be friendly and conversational in your responses.
        2. If the user's input is unclear or too short (like "hi", "hello", etc.), use the HandleGreeting tool ONCE.
        3. When booking appointments:
           - First, ask for the person's name if not already provided
           - Then, ask for their preferred date (in YYYY-MM-DD format) if not already provided
           - Next, ask for their preferred time (in HH:MM 24-hour format, e.g., 14:30) if not already provided
           - Finally, ask for the purpose of the appointment if not already provided
           - Only use the BookAppointment tool when you have all the required information
        4. If the user wants to view appointments:
           - Use the ViewAppointments tool
           - If no appointments are found, suggest booking a new one
        5. If the user asks about valid purposes or examples:
           - Use the GetPurposeExamples tool to show them valid options
        6. For ViewAppointments, HandleGreeting, and GetPurposeExamples tools, do not provide any Action Input as they don't accept any input.
        7. Always maintain a natural conversation flow and be helpful.
        8. When you get a Final Answer, make sure to return it directly without any additional formatting.
        9. Never use "N/A" for Action or Action Input. If no action is needed, go straight to Final Answer.
        10. If you receive a date or time input, check if you have the name first. If not, ask for the name before proceeding.
        11. If you receive a name input, check if you have the date. If not, ask for the date.
        12. If you receive a date input, check if you have the time. If not, ask for the time.
        13. If you receive a time input, check if you have the purpose. If not, ask for the purpose.
        14. Only use the BookAppointment tool when you have all required information (name, date, time, purpose).
        15. Always check the current booking information before asking for details.
        16. If the user provides information out of order, store it and ask for the missing information.
        17. NEVER use the same tool more than 3 times in a row.
        18. If you've used HandleGreeting twice in a row, provide a Final Answer instead.
        19. If you've used any tool 3 times in a row, provide a Final Answer to break the loop.
        20. When asking for information, be specific about the format required:
            - For dates: "Please provide the date in YYYY-MM-DD format (e.g., 2025-09-11)"
            - For time: "Please provide the time in HH:MM 24-hour format (e.g., 09:00 or 14:30)"
            - For purpose: "Please provide the purpose of your appointment"
        21. If the user provides information in the wrong format, politely ask them to provide it in the correct format.
        22. If the user asks for examples of valid purposes, use the GetPurposeExamples tool.
        23. When you have all required information (has_all_info is true), use the BookAppointment tool to book the appointment.
        24. After successfully booking an appointment, confirm the details with the user and ask if they need anything else.
        25. If the user provides information that's already in the booking_info, acknowledge it and ask for the next missing piece of information.
        26. When providing a Final Answer related to booking, summarize all currently known booking details (Name, Date, Time, Purpose) if available.
        
        Begin!
        
        {history}
        Question: {input}
        {agent_scratchpad}"""
        
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            input_variables=["input", "history", "agent_scratchpad", "name", "date", "time", "purpose", "last_action", "action_count", "has_all_info"]
        )

        def should_continue(state: AgentState) -> str:
            """Determine if the agent should continue or finish."""
            logger.info(f"Current step: {state['current_step']}")
            if "Final Answer" in state["current_step"]:
                logger.info("Agent has reached final answer")
                return "end"
            logger.info("Agent needs to use a tool")
            return "tool"

        def call_agent(state: AgentState) -> AgentState:
            """Call the agent to get the next action."""
            messages = state["messages"]
            logger.info(f"Processing message: {messages[-1]['content']}")
            
            # Update booking info from the last message if it contains relevant information
            last_message = messages[-1]["content"]
            
            # Check for name (if it's not a date/time/purpose format)
            if not any(char.isdigit() for char in last_message) and len(last_message.split()) >= 2:
                state["booking_info"]["name"] = last_message
            
            # Check for date (YYYY-MM-DD format)
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', last_message)
            if date_match:
                state["booking_info"]["date"] = date_match.group(0)
            
            # Check for time (HH:MM format)
            time_match = re.search(r'\d{2}:\d{2}', last_message.lower())
            if time_match:
                state["booking_info"]["time"] = time_match.group(0)
            
            # If it's a longer message and not a date/time, assume it's the purpose
            if len(last_message.split()) > 3 and not date_match and not time_match:
                state["booking_info"]["purpose"] = last_message
            
            # Check if we have all required information for booking
            has_all_info = all(state["booking_info"].values())
            
            response = self.llm.invoke(prompt.format(
                input=messages[-1]["content"],
                history="\n".join([m["content"] for m in messages[:-1]]),
                agent_scratchpad="",
                name=state["booking_info"]["name"],
                date=state["booking_info"]["date"],
                time=state["booking_info"]["time"],
                purpose=state["booking_info"]["purpose"],
                last_action=state["last_action"],
                action_count=state["action_count"],
                has_all_info=has_all_info
            ))
            logger.info(f"Agent response: {response.content}")
            state["current_step"] = response.content
            return state

        def call_tool(state: AgentState) -> AgentState:
            """Execute the tool and update the state."""
            current_step = state["current_step"]
            logger.info(f"Executing tool with step: {current_step}")
            
            # Parse the action and input
            action_match = re.search(r"Action: (\w+)", current_step)
            action_input_match = re.search(r"Action Input: (.*?)(?=\n|$)", current_step)
            
            if action_match:
                action = action_match.group(1)
                logger.info(f"Tool action: {action}")
                
                # Update action tracking
                if action == state["last_action"]:
                    state["action_count"] += 1
                else:
                    state["last_action"] = action
                    state["action_count"] = 1
                
                # Force final answer if too many consecutive actions
                if state["action_count"] >= 3:
                    logger.info("Too many consecutive actions, forcing final answer")
                    state["current_step"] = f"Thought: I've used the same tool too many times. I should provide a final answer.\nFinal Answer: I apologize for the confusion. Let me help you with that directly. What would you like to do?"
                    return state
                
                result_content = ""
                if action in tool_map:
                    selected_tool = tool_map[action]
                    tool_input = None # Default for tools not needing input
                    action_input_str = "" # For logging

                    if selected_tool.name == "BookAppointment":
                        if action_input_match:
                            action_input_str = action_input_match.group(1).strip()
                            logger.info(f"BookAppointment raw input: {action_input_str}")
                            # Try to parse action input as JSON
                            if action_input_str.startswith('{') and action_input_str.endswith('}'):
                                try:
                                    tool_input = json.loads(action_input_str)
                                    logger.info(f"Parsed JSON input for BookAppointment: {tool_input}")
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse JSON for BookAppointment: {e}")
                                    result_content = f"Error: Invalid JSON input for BookAppointment: {e}"
                            else:
                                # If not JSON, it might be a simple string or malformed.
                                # The tool itself might handle this, or we can enforce JSON here.
                                # For now, pass it as is if not parsable as JSON and log.
                                tool_input = action_input_str
                                logger.warning(f"BookAppointment input is not JSON: {action_input_str}")
                                # Depending on strictness, could set result_content to an error here.
                        else:
                            logger.error("Error: Missing input for BookAppointment.")
                            result_content = "Error: Missing Action Input for BookAppointment."

                    if not result_content: # If no error so far
                        try:
                            logger.info(f"Executing {selected_tool.name} tool with input: {tool_input}")
                            result = selected_tool.func(tool_input)
                            logger.info(f"Tool result: {result}")

                            if selected_tool.name == "BookAppointment" and "successfully booked" in result.lower():
                                # Reset booking info after successful booking
                                state["booking_info"] = {
                                    "name": None,
                                    "date": None,
                                    "time": None,
                                    "purpose": None
                                }
                                # Reset action tracking
                                state["last_action"] = None
                                state["action_count"] = 0

                            # Construct message content based on whether there was an input
                            if action_input_str :
                                result_content = f"Action: {action}\nAction Input: {action_input_str}\nObservation: {result}"
                            else:
                                result_content = f"Action: {action}\nObservation: {result}"

                        except Exception as e:
                            logger.error(f"Tool execution error for {selected_tool.name}: {e}")
                            result_content = f"Error executing {action}: {str(e)}"
                else:
                    logger.error(f"Error: Tool '{action}' not found.")
                    result_content = f"Error: Tool '{action}' not found."

                state["messages"].append({
                    "role": "assistant",
                    "content": result_content
                })

                # This part seems to be a leftover or incorrect, as BookAppointment is handled above.
                # It implies that 'else' block of 'if action in ["ViewAppointments", ...]' is only for BookAppointment
                # which is not true if more tools are added.
                # The following block for BookAppointment specific logic (like resetting state)
                # should be integrated within the `if selected_tool.name == "BookAppointment":` block.
                # For now, I am commenting out the redundant part.

                # else:
                    # Handle BookAppointment tool - THIS LOGIC IS NOW MOVED/INTEGRATED ABOVE
                    # if action_input_match:
                    #     action_input = action_input_match.group(1).strip()
                    #     logger.info(f"Tool input: {action_input}")
                        
                    #     # Try to parse action input as JSON
                    #     if action_input.startswith('{') and action_input.endswith('}'):
                    #         try:
                    #             action_input = json.loads(action_input)
                    #             logger.info(f"Parsed JSON input: {action_input}")
                    #         except json.JSONDecodeError as e:
                    #             logger.error(f"Failed to parse JSON: {e}")
                        
                    #     # Execute the tool
                    #     try:
                    #         logger.info(f"Executing tool: {action}")
                    #         # result = tools[0].func(action_input)  # BookAppointment is the first tool
                                        # THIS IS THE KEY CHANGE - use tool_map
                    #         selected_tool = tool_map[action] # Assuming action is "BookAppointment"
                    #         result = selected_tool.func(action_input)

                    #         logger.info(f"Tool result: {result}")
                    #         state["messages"].append({
                    #             "role": "assistant",
                    #             "content": f"Action: {action}\nAction Input: {action_input}\nObservation: {result}"
                    #         })
                            # Reset booking info after successful booking
            
            return state

        # Create the graph
        workflow = StateGraph(AgentState)
        logger.info("Creating workflow graph")
        
        # Add nodes
        workflow.add_node("agent", call_agent)
        workflow.add_node("tool", call_tool)
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tool": "tool",
                "end": END
            }
        )
        workflow.add_edge("tool", "agent")
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Compile the graph
        logger.info("Compiling workflow graph")
        app = workflow.compile()
        
        return app
