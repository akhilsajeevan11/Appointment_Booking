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

# Conversational States
CS_INITIAL_GREETING = "INITIAL_GREETING"
CS_GENERAL_INQUIRY = "GENERAL_INQUIRY"
CS_COLLECTING_BOOKING_INFO = "COLLECTING_BOOKING_INFO"
CS_CONFIRMING_BOOKING_INFO = "CONFIRMING_BOOKING_INFO"
CS_VIEWING_APPOINTMENTS = "VIEWING_APPOINTMENTS"
CS_PROVIDING_EXAMPLES = "PROVIDING_EXAMPLES"
CS_CLARIFYING_INPUT = "CLARIFYING_INPUT"
CS_HANDLING_TOOL_ERROR = "HANDLING_TOOL_ERROR"
CS_POST_BOOKING_FEEDBACK = "POST_BOOKING_FEEDBACK"
CS_ENDING_CONVERSATION = "ENDING_CONVERSATION"

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    next: str
    current_step: str
    booking_info: Dict[str, Any]
    last_action: str
    action_count: int
    conversation_state: str

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs):
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        tool_names = [tool.name for tool in self.tools]
        
        current_conversation_state = kwargs.get("current_conversation_state", CS_GENERAL_INQUIRY)
        prompt_segments = kwargs.get("prompt_segments", {})
        state_specific_instructions = prompt_segments.get(current_conversation_state, "")

        final_kwargs = {
            **kwargs,
            "tools": tools_str,
            "tool_names": tool_names,
            "state_specific_instructions": state_specific_instructions
        }

        return self.template.format(**final_kwargs)

class AppointmentAgent:
    def __init__(self):
        self.tools_handler = AppointmentTools()
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
            "name": None, "date": None, "time": None, "purpose": None
        }
        self.prompt_segments = {
            CS_INITIAL_GREETING: (
                "You are in the INITIAL_GREETING state. "
                "Start the conversation with a friendly and welcoming message. "
                "Briefly introduce yourself and what you can help with. "
                "If the user's first message is very short or just a greeting (e.g., 'hi'), use the HandleGreeting tool. "
                "Otherwise, try to understand their intent from their first message to transition to a more specific state."
            ),
            CS_GENERAL_INQUIRY: (
                "You are in the GENERAL_INQUIRY state. "
                "The user's query is not specific yet. "
                "Listen carefully, use the conversation {history} for context. "
                "Politely ask clarifying questions if needed to understand their goal (e.g., booking, viewing, help). "
                "Guide them towards a specific task you can perform."
            ),
            CS_COLLECTING_BOOKING_INFO: (
                "You are in the COLLECTING_BOOKING_INFO state. Your goal is to gather all necessary details for an appointment. "
                "Refer to {booking_info} to see what's already collected. Check {history} for recent user inputs. "
                "Politely ask for the required information (name, date YYYY-MM-DD, time HH:MM, purpose) one by one if not provided. "
                "If the user provides information out of order, acknowledge it, store it, and ask for the next logical piece. "
                "Example: If they give a time first, say: 'Okay, {time} noted. What date would you like for this appointment?' "
                "Be encouraging and clear. If they say 'yes' or 'correct' to a piece of info you suggested, confirm it and move on."
            ),
            CS_CONFIRMING_BOOKING_INFO: (
                "You are in the CONFIRMING_BOOKING_INFO state. You should have all details: {booking_info[name]}, {booking_info[date]}, {booking_info[time]}, {booking_info[purpose]}. "
                "Clearly list all these details back to the user and ask for their explicit confirmation (e.g., 'yes' or 'correct') before using the BookAppointment tool. "
                "Example: 'So, I have an appointment for {booking_info[name]} on {booking_info[date]} at {booking_info[time]} for {booking_info[purpose]}. Is that all correct?' "
                "If they say no or want to change something, identify what needs to change and potentially transition back to COLLECTING_BOOKING_INFO for that piece."
            ),
            CS_VIEWING_APPOINTMENTS: (
                "You are in the VIEWING_APPOINTMENTS state. The user wants to see their appointments. "
                "Use the ViewAppointments tool. Present the information clearly. "
                "If there are no appointments, inform them politely and ask if they'd like to book one."
            ),
            CS_PROVIDING_EXAMPLES: (
                "You are in the PROVIDING_EXAMPLES state. The user asked for examples (e.g., of appointment purposes). "
                "Use the GetPurposeExamples tool. Present the examples clearly. Ask if they find them helpful or want to proceed with booking."
            ),
            CS_CLARIFYING_INPUT: (
                "You are in the CLARIFYING_INPUT state. The user's last input was unclear, ambiguous, or too vague. "
                "Review {history}. Politely ask for more specific information. "
                "Avoid making the user feel at fault. Example: 'I'm not quite sure I understand. Could you please tell me a bit more about what you'd like to do?' or 'To make sure I get it right, could you clarify [specific part]?'"
            ),
            CS_HANDLING_TOOL_ERROR: (
                "You are in the HANDLING_TOOL_ERROR state. A tool call resulted in an error (visible in {agent_scratchpad} or history). "
                "Do not expose raw error messages to the user. Apologize for the technical difficulty. "
                "Example: 'I seem to be having a slight technical issue. Could you please try that request again?' or 'I couldn't complete that action due to a technical hiccup. Perhaps we can try a different way?' "
                "If appropriate, suggest an alternative or ask the user to rephrase."
            ),
            CS_POST_BOOKING_FEEDBACK: (
                "You are in the POST_BOOKING_FEEDBACK state. An appointment was just successfully booked. "
                "Congratulate the user or confirm the success. "
                "Ask if there's anything else you can assist them with. "
                "Example: 'Great, your appointment is all set! Is there anything else I can help you with today?'"
            ),
            CS_ENDING_CONVERSATION: (
                "You are in the ENDING_CONVERSATION state. The user has indicated they want to end the chat (e.g., 'exit', 'bye') or the task is fully complete. "
                "Provide a friendly closing. Example: 'Thanks for using the Appointment System! Have a great day.' or 'You're welcome! Feel free to reach out if you need anything else.'"
            )
        }
        
    def create_agent(self):
        tools = [Tool(**tool) for tool in self.tools_handler.get_tools()]
        logger.info(f"Available tools: {[tool.name for tool in tools]}")
        tool_map = {tool.name: tool for tool in tools}
        
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
        (Instruction 16 removed as per subtask)
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
        (Instruction 25 removed as per subtask)
        26. When providing a Final Answer related to booking, summarize all currently known booking details (Name, Date, Time, Purpose) if available.
        
        {state_specific_instructions}

        Begin!
        
        {history}
        Question: {input}
        {agent_scratchpad}"""
        
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            input_variables=[
                "input", "history", "agent_scratchpad",
                "name", "date", "time", "purpose",
                "last_action", "action_count", "has_all_info",
                "state_specific_instructions"
            ]
        )

        def should_continue(state: AgentState) -> str:
            logger.info(f"Current step: {state['current_step']}")
            if "Final Answer" in state["current_step"]:
                logger.info("Agent has reached final answer")
                return "end"
            logger.info("Agent needs to use a tool")
            return "tool"

        def call_agent(state: AgentState) -> AgentState:
            messages = state["messages"]
            logger.info(f"Processing message: {messages[-1]['content']}")
            
            last_message = messages[-1]["content"]
            current_conversation_state = state.get("conversation_state", CS_INITIAL_GREETING)
            logger.info(f"Current conversational state (start of call_agent): {current_conversation_state}")

            # State transition logic
            if current_conversation_state == CS_INITIAL_GREETING:
                if len(last_message) > 10 or any(kw in last_message.lower() for kw in ["book", "appointment", "view", "schedule"]):
                    state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                    logger.info(f"Transitioning state from {CS_INITIAL_GREETING} to {CS_COLLECTING_BOOKING_INFO}")
                else:
                    state["conversation_state"] = CS_GENERAL_INQUIRY
                    logger.info(f"Transitioning state from {CS_INITIAL_GREETING} to {CS_GENERAL_INQUIRY}")
                current_conversation_state = state["conversation_state"] # Update for prompt formatting this turn

            # Booking info extraction (remains the same)
            if not any(char.isdigit() for char in last_message) and len(last_message.split()) >= 2:
                state["booking_info"]["name"] = last_message
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', last_message)
            if date_match:
                state["booking_info"]["date"] = date_match.group(0)
            time_match = re.search(r'\d{2}:\d{2}', last_message.lower())
            if time_match:
                state["booking_info"]["time"] = time_match.group(0)
            if len(last_message.split()) > 3 and not date_match and not time_match:
                state["booking_info"]["purpose"] = last_message
            
            has_all_info = all(state["booking_info"].values())

            # State transition based on collected info
            if current_conversation_state == CS_COLLECTING_BOOKING_INFO and has_all_info:
                state["conversation_state"] = CS_CONFIRMING_BOOKING_INFO
                logger.info(f"Transitioning state from {CS_COLLECTING_BOOKING_INFO} to {CS_CONFIRMING_BOOKING_INFO} as all info is collected.")
                current_conversation_state = state["conversation_state"] # Update for prompt formatting
            
            elif current_conversation_state == CS_CONFIRMING_BOOKING_INFO:
                if any(kw in last_message.lower() for kw in ["no", "change", "wrong", "don't", "alter", "modify"]):
                    state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                    # Potentially clear parts of booking_info here or let LLM guide
                    logger.info(f"User wants to change booking info. Transitioning from {CS_CONFIRMING_BOOKING_INFO} to {CS_COLLECTING_BOOKING_INFO}.")
                    current_conversation_state = state["conversation_state"] # Update for prompt formatting
                # If user confirms (e.g. "yes", "correct"), LLM (guided by CS_CONFIRMING_BOOKING_INFO prompt) should use BookAppointment.
                # Transition to CS_POST_BOOKING_FEEDBACK or CS_HANDLING_TOOL_ERROR will happen in call_tool.

            logger.info(f"Conversational state for LLM prompt: {current_conversation_state}")
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
                has_all_info=has_all_info,
                current_conversation_state=current_conversation_state,
                prompt_segments=self.prompt_segments
            ))
            logger.info(f"Agent response: {response.content}")
            state["current_step"] = response.content
            return state

        def call_tool(state: AgentState) -> AgentState:
            current_step = state["current_step"]
            logger.info(f"Executing tool with step: {current_step}")
            
            action_match = re.search(r"Action: (\w+)", current_step)
            action_input_match = re.search(r"Action Input: (.*?)(?=\n|$)", current_step)
            
            if action_match:
                action = action_match.group(1)
                logger.info(f"Tool action: {action}")
                
                if action == state["last_action"]:
                    state["action_count"] += 1
                else:
                    state["last_action"] = action
                    state["action_count"] = 1
                
                if state["action_count"] >= 3:
                    logger.info("Too many consecutive actions, forcing final answer")
                    state["current_step"] = f"Thought: I've used the same tool too many times. I should provide a final answer.\nFinal Answer: I apologize for the confusion. Let me help you with that directly. What would you like to do?"
                    state["conversation_state"] = CS_GENERAL_INQUIRY # Reset state on loop break
                    return state
                
                result_content = ""
                if action in tool_map:
                    selected_tool = tool_map[action]
                    tool_input = None
                    action_input_str = ""

                    if selected_tool.name == "BookAppointment":
                        if action_input_match:
                            action_input_str = action_input_match.group(1).strip()
                            logger.info(f"BookAppointment raw input: {action_input_str}")
                            if action_input_str.startswith('{') and action_input_str.endswith('}'):
                                try:
                                    tool_input = json.loads(action_input_str)
                                    logger.info(f"Parsed JSON input for BookAppointment: {tool_input}")
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse JSON for BookAppointment: {e}")
                                    result_content = f"Error: Invalid JSON input for BookAppointment: {e}"
                                    state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                                    logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to invalid JSON for BookAppointment.")
                            else:
                                tool_input = action_input_str
                                logger.warning(f"BookAppointment input is not JSON: {action_input_str}")
                                # Potentially set to CS_HANDLING_TOOL_ERROR or CS_CLARIFYING_INPUT if strict JSON is required by tool
                        else:
                            logger.error("Error: Missing input for BookAppointment.")
                            result_content = "Error: Missing Action Input for BookAppointment."
                            state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                            logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to missing input for BookAppointment.")

                    if not result_content: # If no parsing error or missing input for BookAppointment
                        try:
                            logger.info(f"Executing {selected_tool.name} tool with input: {tool_input}")
                            result = selected_tool.func(tool_input)
                            logger.info(f"Tool result: {result}")

                            # State transitions based on tool and result
                            if selected_tool.name == "BookAppointment":
                                if "error" in result.lower() or "failed" in result.lower():
                                    state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                                    logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to BookAppointment operational error: {result}")
                                elif "successfully booked" in result.lower():
                                    state["booking_info"] = {"name": None, "date": None, "time": None, "purpose": None}
                                    state["last_action"] = None # Reset action tracking post-success
                                    state["action_count"] = 0
                                    state["conversation_state"] = CS_POST_BOOKING_FEEDBACK
                                    logger.info(f"Transitioning state to {CS_POST_BOOKING_FEEDBACK} after successful booking.")
                                # Else, if BookAppointment returns something neutral, state might not change here explicitly
                            elif selected_tool.name == "HandleGreeting":
                                state["conversation_state"] = CS_GENERAL_INQUIRY
                                logger.info(f"Transitioning state to {CS_GENERAL_INQUIRY} after HandleGreeting.")
                            elif selected_tool.name in ["ViewAppointments", "GetPurposeExamples"]:
                                state["conversation_state"] = CS_GENERAL_INQUIRY
                                logger.info(f"Transitioning state to {CS_GENERAL_INQUIRY} after {selected_tool.name}.")

                            if action_input_str :
                                result_content = f"Action: {action}\nAction Input: {action_input_str}\nObservation: {result}"
                            else:
                                result_content = f"Action: {action}\nObservation: {result}"

                        except Exception as e:
                            logger.error(f"Tool execution error for {selected_tool.name}: {e}")
                            result_content = f"Error executing {action}: {str(e)}" # For agent's observation
                            state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                            logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to tool execution exception.")
                else:
                    logger.error(f"Error: Tool '{action}' not found.")
                    result_content = f"Error: Tool '{action}' not found." # For agent's observation
                    state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                    logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to tool not found.")

                state["messages"].append({"role": "assistant", "content": result_content})
            
            return state

        workflow = StateGraph(AgentState)
        logger.info("Creating workflow graph")
        
        workflow.add_node("agent", call_agent)
        workflow.add_node("tool", call_tool)
        
        workflow.add_conditional_edges("agent", should_continue, {"tool": "tool", "end": END})
        workflow.add_edge("tool", "agent")
        
        workflow.set_entry_point("agent")
        
        logger.info("Compiling workflow graph")
        app = workflow.compile()
        
        return app
