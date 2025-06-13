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
from dateutil import parser as dateutil_parser
from datetime import datetime, time as dt_time # Add dt_time for time object comparison

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
CS_AWAITING_RESPONSE_TO_OPTIONS = "AWAITING_RESPONSE_TO_OPTIONS"
CS_CLARIFYING_INPUT = "CLARIFYING_INPUT"
CS_HANDLING_TOOL_ERROR = "HANDLING_TOOL_ERROR"
CS_POST_BOOKING_FEEDBACK = "POST_BOOKING_FEEDBACK"
CS_ENDING_CONVERSATION = "ENDING_CONVERSATION"
CS_AWAITING_FINAL_CONFIRMATION = "AWAITING_FINAL_CONFIRMATION"

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
                "You are in the INITIAL_GREETING state. This is the user's first interaction or a fresh start. "
                "Analyze the user's first message ({history} will be empty or just their message). "
                "If the user's message is ONLY a simple greeting (e.g., 'hi', 'hello', 'hey'), then using the HandleGreeting tool is appropriate to provide standard options. "
                "However, if the user says they are new, asks for help, or asks what you can do (e.g., 'i am new here', 'help me', 'what can you do?'), "
                "DO NOT use HandleGreeting. Instead, formulate a direct, welcoming, and informative response. "
                "For example: 'Hello! I'm an appointment booking assistant. I can help you schedule new appointments, view existing ones, or show you examples of common appointment purposes. What would you like to do today, or would you like a bit more detail on how I can help?' "
                "Your goal is to be immediately helpful and engaging based on their initial statement."
            ),
            CS_GENERAL_INQUIRY: (
                "You are in the GENERAL_INQUIRY state. The user's immediate need isn't a specific task, or they haven't clearly affirmed a direct offer you made, or they've just responded to a set of options you provided. "
                "Review the {history} carefully. Pay special attention to your last message to the user. "

                "If your last message (from {history}) presented options or offered to explain something (e.g., 'explain X or book Y?', or 'I can explain Z'), "
                "and the user's current input ({input}) seems to be choosing or referring to that offer (e.g., 'explain X', or just 'explain' if X was the clear topic for explanation), "
                "then your Thought process should be: 'User is choosing/referring to an option I offered.' Then, proceed to fulfill that (e.g., provide the explanation). "
                "This is important even if the state isn't CS_AWAITING_RESPONSE_TO_OPTIONS, to ensure user requests are understood in context. "

                "If you just made a booking offer (e.g., 'Shall I book X for you?') and the user's response was NOT a clear 'yes' (leading here), "
                "first try to understand their response. They might have a question, a hesitation, or want to modify something. Address that first. "
                "Example: If you offered to book 'Dental Checkup' and user said 'hmm, how long does that take?', answer the question, then gently ask if they still want to book it or need other info. "

                "HANDLING NEWLY PROVIDED NAME: If the user's current input ({input}) has just resulted in {booking_info[name]} being populated (e.g., they said 'My name is John' or 'i am indhu k'), "
                "AND other booking details like {booking_info[date]} or {booking_info[purpose]} are still missing, this is a strong signal they might want to book. "
                "Your Thought process should be: 'The user has provided their name: {booking_info[name]}. I should now collect the remaining details for booking.' "
                "Acknowledge the name and ask for the next piece of information (e.g., date or purpose). For example: 'Thanks, {booking_info[name]}. What date would you like for your appointment (YYYY-MM-DD)?' or 'Got it, {booking_info[name]}. What is the purpose of your visit?' "
                "Crucially, in your Thought, include: 'Set next state to CS_COLLECTING_BOOKING_INFO.' to ensure the conversation moves to the focused collection state."

                "If the user previously indicated they are new or asked for general help (e.g., 'help me', 'i am new here'), and you haven't yet provided substantial guidance: "
                "If their statement is very broad and indicates a complete lack of understanding (e.g., 'I don't know anything', 'what is this?'), "
                "your first step should be to provide a concise overview of your main functions. For example: 'I'm an appointment booking assistant. I can help you schedule new appointments, check your existing ones, or show you examples of typical appointment reasons. To get started, you can tell me what you'd like to do, like saying 'book an appointment' or 'view my appointments'.' "
                "After providing this overview, you can then ask what they'd like to do or if they need more details on any of those functions. "
                "For less broad 'new user' queries, or if they respond to the overview by asking for more specific help, you can then offer to explain the booking process in detail, or show examples of appointment purposes (consider GetPurposeExamples tool if they seem broadly unsure of the *type* of appointment they need). "
                "Example for a new user who seems to have some idea: 'Since you're new, I can quickly explain how booking works, or I can show you some example appointment reasons. What would be more helpful for you right now?' "

                "SPECIAL CASE for name confirmation: If your last message asked the user to confirm a name stored in {booking_info[name]} AND to provide other details (like date/time), "
                "AND the {booking_info[name]} looks like a placeholder phrase (e.g., contains 'new here', 'help', 'assist', 'process', or is longer than 4 words) rather than a real name, "
                "AND the user's current input ({input}) is a general affirmation (e.g., 'yes', 'yeah', 'sure', 'okay'): "
                "DO NOT assume the placeholder name is correct. Your Thought process should be: 'User affirmed, but the stored name is suspect. I need to get their actual name.' "
                "Your Final Answer should then be a polite request for their name, like: 'Okay, great! To get started with your consultation booking, could you please tell me your name?' "
                "Then, you should expect to transition to COLLECTING_BOOKING_INFO where the name will be gathered. If you can set the next state in your thought, set it to CS_COLLECTING_BOOKING_INFO."

                "For other general inquiries, or if the user seems to be exploring options after declining a specific offer: "
                "Listen carefully, politely ask clarifying questions to guide them towards a task you can perform (booking, viewing, examples). "
                "Avoid immediately re-offering 'GetPurposeExamples' if they just sidestepped a specific booking offer, unless they explicitly ask for alternatives or seem lost. Focus on their last statement. "
                "Your aim is to understand their current need and help them navigate to a solution."
            ),
            CS_COLLECTING_BOOKING_INFO: (
                "You are in the COLLECTING_BOOKING_INFO state. Your goal is to gather all necessary details for an appointment. "
                "Review {booking_info} to see what's already collected. Also, check {history} for recent user inputs. "

                "If {booking_info[purpose]} is already set (e.g., from a previous confirmation or user statement), start by acknowledging it. "
                "Example if purpose is known: 'Okay, we're setting up your {booking_info[purpose]} appointment. ' "
                "Then, proceed to ask for the next piece of missing information in a logical order (typically: name, then date YYYY-MM-DD, then time HH:MM). "
                "If {booking_info[name]} is also known, acknowledge that too: 'For {booking_info[name]} for the {booking_info[purpose]} appointment...' "

                "If a piece of information is provided by the user in their last message, acknowledge it and then ask for the next missing item. "
                "Example if user just provided name: 'Thanks, {booking_info[name]}. Now, what date would you like for this appointment (in YYYY-MM-DD format)?' "
                "Example if user just provided date: 'Got it, {booking_info[date]}. And what time (in HH:MM 24-hour format)?' "
                "Example if user just provided time: 'Perfect, {booking_info[time]}. Lastly, what is the purpose of this appointment?' (Only ask purpose if not already known). "

                "If the user provides information out of order, acknowledge it, ensure it's stored in booking_info, and then ask for the next logical piece. "
                "Be encouraging and clear. If they say 'yes' or 'correct' to a piece of info you suggested (though less likely in this state unless you are confirming a format), confirm it and move on. "
                "Your goal is to fill all fields in {booking_info}: name, date, time, and purpose."
            ),
            CS_CONFIRMING_BOOKING_INFO: (
                "You are in the CONFIRMING_BOOKING_INFO state. You have all details: {booking_info[name]} on {booking_info[date]} at {booking_info[time]} for {booking_info[purpose]}. "
                "Your task is to clearly list ALL these details back to the user and ask for their explicit confirmation (e.g., 'Is this all correct?'). "
                "Example: 'So, I have an appointment for {booking_info[name]} on {booking_info[date]} at {booking_info[time]} for {booking_info[purpose]}. Is that all correct?' "
                "In your Thought process, you MUST include: 'Set next state to CS_AWAITING_FINAL_CONFIRMATION.' "
                "Do NOT use the BookAppointment tool in this current turn. Wait for the user's response."
            ),
            CS_AWAITING_FINAL_CONFIRMATION: (
                "You are in the AWAITING_FINAL_CONFIRMATION state. The user was just asked to confirm all appointment details. "
                "Their current input ({input}) is their response to that confirmation request. "
                "If the user's input is a clear affirmation (e.g., 'yes', 'correct', 'perfect', 'proceed'): "
                "Your Thought process should be: 'User confirmed all details. I will now book the appointment.' "
                "Then, use the BookAppointment tool with the details from {booking_info}. "
                "If the user's input is negative or suggests a change (e.g., 'no', 'that date is wrong', 'change the time'): "
                "Your Thought process should be: 'User wants to change details. I need to ask what to change and go back to collecting info.' "
                "Your Final Answer should ask the user what specific information they'd like to correct or change. For example: 'Okay, what information isn't correct or what would you like to change?' "
                "In your Thought, also include: 'Set next state to CS_COLLECTING_BOOKING_INFO.' "
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
            CS_AWAITING_RESPONSE_TO_OPTIONS: (
                "You are in the AWAITING_RESPONSE_TO_OPTIONS state. "
                "In your previous turn, you presented the user with specific choices or asked a question that expects a selection from options you provided (e.g., 'Would you like A or B?', 'Shall I explain X or Y?'). "
                "The user's current message ({input}) is their response to those options. "
                "Carefully analyze their response in the context of the options you presented (review {history}). "
                "If their response clearly maps to one of the options (e.g., they say 'A', or 'explain X'), your Thought should be to proceed with that chosen action. "
                "If their response is ambiguous, ask for clarification regarding the options you gave. Example: 'Sorry, I didn't quite catch that. Did you mean A or B?' "
                "If they ask a completely new question or make an unrelated statement, you can address it, but then try to gently guide them back to making a choice if the original options are still relevant or ask if they'd like to abandon those options."
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
                "You have already informed the user of the successful booking and asked if they need anything else. "
                "Now, you are processing their response to that question. "
                "If the user indicates they need nothing further (e.g., 'no', 'no thanks', 'that's all'), your thought process should be to end the conversation. Formulate a polite closing statement. "
                "If the user asks for something new or provides other input, your thought process should be to assess this new request and transition to a general inquiry or collecting information state. "
                "Example user says 'no': Thought: User needs nothing else. I should say goodbye. Final Answer: You're welcome! Have a great day. "
                "Example user says 'actually, can you view my appointments?': Thought: User has a new request. I should use the ViewAppointments tool. Action: ViewAppointments..."
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
        Thought: I now know the final answer. If your Final Answer presents choices to the user (e.g., 'Do you want A or B?'), include in your Thought: 'I am presenting options, the user will now choose. Set next state to CS_AWAITING_RESPONSE_TO_OPTIONS.'
        Final Answer: the final answer to the original input question
        
        For booking appointments, use this JSON format:
        {{"name": "person name", "date": "YYYY-MM-DD", "time": "HH:MM", "purpose": "appointment purpose"}}
        
        Important Instructions:
        1. Be friendly and conversational in your responses.
        2. Consider using the HandleGreeting tool if the user provides *only* a very simple greeting (e.g., 'Hi', 'Hello', 'Hey there'). For most other initial queries, even if short (like 'help me' or 'i am new'), aim to respond contextually first based on the guidance in the state-specific instructions, rather than immediately using HandleGreeting.
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

            LEARNING FROM TOOL OBSERVATIONS:
            - When you receive an Observation after an Action, pay close attention to it.
            - If the Observation indicates an error, a problem with your Action Input, or that the tool didn't behave as expected:
                1. Re-read the description of the tool you just used from the available {tools} list.
                2. Think about why it might have failed. Did you provide the wrong kind of input? Did you misunderstand what the tool does? Or did the tool itself encounter an issue?
                3. If you believe you made a mistake in how you called the tool (e.g., wrong input format, or provided input to a no-input tool like GetPurposeExamples), your next Thought should be to try the tool again with the corrected invocation (e.g., with no Action Input if that was the error).
                4. If the tool itself seems to have an internal error (e.g., 'database connection failed' from BookAppointment), or if you're unsure how to fix your tool call, DO NOT blindly repeat the same failed Action. Think about alternative tools or ways to help the user. You can also politely inform the user that you encountered a hiccup and ask them to rephrase or suggest a different approach. (The CS_HANDLING_TOOL_ERROR state will also guide you on user-facing messages).
                5. Example: If a tool observation says "this tool takes no input" and you previously provided an Action Input, your next Thought should be "I see, that tool doesn't need input. I'll call it again correctly." then "Action: [ToolName]" (with no "Action Input:" line).

        17. NEVER use the same tool more than 3 times in a row.
        18. If you've used HandleGreeting twice in a row, provide a Final Answer instead.
        19. If you've used any tool 3 times in a row, provide a Final Answer to break the loop.
        20. When asking for information, be specific about the format required:
            - For dates: "Please provide the date in YYYY-MM-DD format (e.g., 2025-09-11)"
            - For time: "Please provide the time in HH:MM 24-hour format (e.g., 09:00 or 14:30)"
            - For purpose: "Please provide the purpose of your appointment"
        21. If the user provides information in the wrong format, politely ask them to provide it in the correct format.
        22. If the user asks for examples of valid purposes, use the GetPurposeExamples tool.
        23. When you have all required information (has_all_info is true) AND you are in a state where booking is the next logical step (e.g., CS_AWAITING_FINAL_CONFIRMATION after user confirmed details), use the BookAppointment tool. Do NOT use BookAppointment if you are in CS_CONFIRMING_BOOKING_INFO waiting for user's yes/no answer.
        24. After successfully booking an appointment, confirm the details with the user and ask if they need anything else.
        (Instruction 25 removed as per subtask)
        26. When providing a Final Answer related to booking, summarize all currently known booking details (Name, Date, Time, Purpose) if available.
        27. In your 'Thought:' process, if you are clarifying, confirming, or have just determined a specific purpose for the appointment based on the conversation, explicitly state it using the format 'Effective current purpose: [the specific purpose string]'. If no specific purpose is active or clear, you can omit this or state 'Effective current purpose: None'. This helps ensure the correct purpose is tracked.
        
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
            if "Action:" not in state["current_step"] or "Final Answer:" in state["current_step"]:
                logger.info("Agent has reached Final Answer or a direct response.")
                return "end"
            logger.info("Agent needs to use a tool.")
            return "tool"

        def call_agent(state: AgentState) -> AgentState:
            messages = state["messages"]
            logger.info(f"Processing message: {messages[-1]['content']}")
            
            last_message = messages[-1]["content"].strip()

            current_conversation_state_for_prompt = state.get("conversation_state", CS_INITIAL_GREETING)
            logger.info(f"Current conversational state (start of call_agent): {current_conversation_state_for_prompt}")

            agents_previous_content = ""
            potential_purpose_from_offer = None
            made_booking_offer = False

            if len(state["messages"]) >= 2:
                if state["messages"][-2]["role"] == "assistant":
                    raw_prev_agent_msg = state["messages"][-2]["content"]
                    if "Final Answer:" in raw_prev_agent_msg:
                        agents_previous_content = raw_prev_agent_msg.split("Final Answer:", 1)[-1].strip()
                    else:
                        agents_previous_content = raw_prev_agent_msg

            if agents_previous_content:
                booking_offer_patterns = [
                    r"book a '([^']+)' for you", r"book '([^']+)'", r"booking '([^']+)'",
                    r"proceed with booking for '([^']+)'",
                    "book for you?", "proceed with booking?", "shall I book", "like me to book"
                ]
                for pattern in booking_offer_patterns:
                    if "([^']+)" in pattern:
                        match = re.search(pattern, agents_previous_content, re.IGNORECASE)
                        if match:
                            try:
                                potential_purpose_from_offer = match.group(1)
                            except IndexError:
                                potential_purpose_from_offer = None
                            made_booking_offer = True
                            logger.info(f"Detected booking offer with potential purpose: '{potential_purpose_from_offer}' from pattern '{pattern}' in: '{agents_previous_content}'")
                            break
                    elif pattern.lower() in agents_previous_content.lower():
                        made_booking_offer = True
                        logger.info(f"Detected general booking offer from pattern '{pattern}' in: '{agents_previous_content}'")
                        break

            affirmative_responses = ["yes", "yeah", "sure", "okay", "ok", "please", "do it", "proceed", "sounds good", "great"]
            user_affirmed = last_message.lower() in affirmative_responses or \
                            any(affirmative in last_message.lower().split() for affirmative in affirmative_responses)

            if made_booking_offer and user_affirmed:
                logger.info(f"User affirmed a previous booking offer. Last agent msg: '{agents_previous_content}', User input: '{last_message}'")

                # Check if the currently stored name is suspect
                current_name = state["booking_info"]["name"]
                if current_name:
                    suspect_keywords = ["new here", "help", "please", "process", "assist", "guide"]
                    words_in_name = current_name.lower().split()
                    if any(kw in words_in_name for kw in suspect_keywords) or len(words_in_name) > 4:
                        logger.info(f"Suspect name '{current_name}' found after user affirmation. Clearing it to re-prompt.")
                        state["booking_info"]["name"] = None # Clear suspect name

                state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                current_conversation_state_for_prompt = CS_COLLECTING_BOOKING_INFO
                if potential_purpose_from_offer and state["booking_info"]["purpose"] is None:
                    state["booking_info"]["purpose"] = potential_purpose_from_offer
                    logger.info(f"Pre-filled purpose from offer: '{potential_purpose_from_offer}'")
            else:
                if current_conversation_state_for_prompt == CS_INITIAL_GREETING:
                    user_input_lower = last_message.lower()
                    if any(phrase in user_input_lower for phrase in ["new here", "am new", "help", "what can you do", "how does this work", "guide me", "get started"]):
                        state["conversation_state"] = CS_GENERAL_INQUIRY
                        logger.info(f"Transitioning from {CS_INITIAL_GREETING} to {state['conversation_state']} due to new user/help query.")
                    elif len(last_message) > 15 or any(kw in user_input_lower for kw in ["book", "appointment", "view", "schedule", "check"]):
                        state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                        logger.info(f"Transitioning from {CS_INITIAL_GREETING} to {state['conversation_state']} due to specific intent.")
                    else:
                        state["conversation_state"] = CS_GENERAL_INQUIRY
                        logger.info(f"Transitioning from {CS_INITIAL_GREETING} to {state['conversation_state']} for general/short input.")

            if current_conversation_state_for_prompt == CS_AWAITING_RESPONSE_TO_OPTIONS:
                logger.info(f"In CS_AWAITING_RESPONSE_TO_OPTIONS, processing user choice: '{last_message}'. LLM will determine next specific task state.")
                pass

            elif current_conversation_state_for_prompt == CS_POST_BOOKING_FEEDBACK:
                logger.info(f"Processing user response ('{last_message}') in CS_POST_BOOKING_FEEDBACK.")
                if any(kw in last_message.lower() for kw in ["no", "nothing", "nope", "don't", "not now", "that's all", "that is all", "finished", "done", "bye"]):
                    state["conversation_state"] = CS_ENDING_CONVERSATION
                    logger.info(f"Transitioning from {CS_POST_BOOKING_FEEDBACK} to {CS_ENDING_CONVERSATION} based on user response.")
                elif not last_message:
                    state["conversation_state"] = CS_ENDING_CONVERSATION
                    logger.info(f"Transitioning from {CS_POST_BOOKING_FEEDBACK} to {CS_ENDING_CONVERSATION} due to empty input.")
                else:
                    state["conversation_state"] = CS_GENERAL_INQUIRY
                    logger.info(f"Transitioning from {CS_POST_BOOKING_FEEDBACK} to {CS_GENERAL_INQUIRY} to handle new/unclear request: {last_message}")

            elif current_conversation_state_for_prompt == CS_ENDING_CONVERSATION:
                if last_message:
                    state["conversation_state"] = CS_GENERAL_INQUIRY
                    logger.info(f"User provided new input ('{last_message}') after conversation was ending. Transitioning from {CS_ENDING_CONVERSATION} to {CS_GENERAL_INQUIRY}.")

            if current_conversation_state_for_prompt in [CS_INITIAL_GREETING, CS_COLLECTING_BOOKING_INFO, CS_GENERAL_INQUIRY, CS_CONFIRMING_BOOKING_INFO] or \
               (made_booking_offer and user_affirmed):
                # Attempt to extract name if not already set
                if state["booking_info"]["name"] is None:
                    # Avoid capturing long phrases, questions, or help requests as names
                    words_in_message = last_message.lower().split()
                    num_words = len(words_in_message)
                    is_potential_name = True

                    if num_words < 2 or num_words > 4: # Typical name length
                        is_potential_name = False

                    if any(kw in words_in_message for kw in ["help", "assist", "guide", "what", "how", "why", "can", "could", "show", "tell", "explain"]):
                        is_potential_name = False

                    if "?" in last_message:
                        is_potential_name = False

                    if not any(char.isalpha() for char in last_message): # Must contain some letters
                        is_potential_name = False

                    # Ensure it doesn't look like a date or time
                    if re.search(r'\d{4}-\d{2}-\d{2}', last_message) or re.search(r'\d{2}:\d{2}', last_message):
                        is_potential_name = False

                    if is_potential_name and not any(char.isdigit() for char in last_message):
                        logger.info(f"Potentially extracting name: {last_message}")
                        state["booking_info"]["name"] = last_message
                    else:
                        logger.info(f"Skipping name extraction for: {last_message} based on new rules.")

            # Date and Time Extraction using dateutil.parser
            if state["booking_info"]["date"] is None or state["booking_info"]["time"] is None:
                contains_date_keyword = False
                contains_day_pattern = False
                contains_year_pattern = False
                contains_time_keyword = False
                contains_hour_pattern = False
                # These flags will be set to True within the try block if relevant patterns are found.
                try:
                    # Use fuzzy parsing to ignore irrelevant parts of the string.
                    # default to now() helps if only time is given (uses today's date)
                    # or if only date is given (uses midnight time).
                    parsed_dt = dateutil_parser.parse(last_message, fuzzy=True, default=datetime.now())

                    potential_date_keywords = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'today', 'tomorrow', 'next week', 'next month']
                    contains_date_keyword = any(kw in last_message.lower() for kw in potential_date_keywords)
                    # Basic check for day numbers, e.g., "20th", "5", "august 20"
                    contains_day_pattern = re.search(r'\b(\d{1,2})(st|nd|rd|th)?\b', last_message.lower()) is not None
                    # Check for year pattern
                    contains_year_pattern = re.search(r'\b(20\d{2})\b', last_message.lower()) is not None


                    # Heuristic to decide if a date was likely mentioned by the user:
                    # 1. Did the parser pick up a date different from today's date?
                    # 2. Or, did the user's message contain explicit date-related keywords or patterns?
                    date_is_different_from_default_date = parsed_dt.date() != datetime.now().date()
                    user_likely_mentioned_date = contains_date_keyword or contains_day_pattern or contains_year_pattern

                    date_was_intended_by_user = date_is_different_from_default_date or user_likely_mentioned_date

                    if date_was_intended_by_user: # Allow overwrite
                        extracted_date_obj = parsed_dt.date()

                        # Year handling: if no year was explicitly mentioned and the parsed date (with current year) is in the past, assume next year.
                        if not contains_year_pattern and extracted_date_obj < datetime.now().date():
                            logger.info(f"Extracted date {extracted_date_obj} is in the past and no year mentioned, trying next year.")
                            try:
                                extracted_date_obj = extracted_date_obj.replace(year=extracted_date_obj.year + 1)
                            except ValueError: # Handle Feb 29 on non-leap year if year is incremented
                                extracted_date_obj = extracted_date_obj.replace(year=extracted_date_obj.year + 1, day=28)

                        state["booking_info"]["date"] = extracted_date_obj.strftime("%Y-%m-%d")
                        logger.info(f"Extracted/Updated date: {state['booking_info']['date']}")

                    # Heuristic to decide if a time was likely mentioned by the user:
                    # 1. Did the parser pick up a time different from midnight (which is default for date-only strings)?
                    # 2. Or, did the user's message contain explicit time-related keywords or patterns?
                    time_is_different_from_default_time = parsed_dt.time() != dt_time(0, 0) # dt_time(0,0) is midnight
                    contains_time_keyword = any(kw in last_message.lower() for kw in ['am', 'pm', 'noon', 'midnight', 'o\'clock', 'hour', 'minute'])
                    # Check for explicit hour numbers like 7, 09, 14, possibly with am/pm or colon
                    contains_hour_pattern = re.search(r'\b([0-1]?[0-9]|2[0-3])(:[0-5][0-9])?(\s*(am|pm))?\b', last_message.lower()) is not None

                    user_likely_mentioned_time = contains_time_keyword or contains_hour_pattern

                    # If the time parsed is not the default (midnight) OR the user explicitly mentioned time-like words
                    time_was_intended_by_user = time_is_different_from_default_time or user_likely_mentioned_time


                    if time_was_intended_by_user: # Allow overwrite
                        # Heuristic to set minutes to 00 if "11am" style input and minutes are non-zero
                        if parsed_dt.minute != 0 and (re.search(r'\bam\b|\bpm\b', last_message, re.IGNORECASE)) and not (re.search(r'\d:\d{2}', last_message)):
                             logger.info(f"Time heuristic: Input '{last_message}', parsed time {parsed_dt.time()}. Contains am/pm and no colon. Resetting minutes to 00.")
                             parsed_dt = parsed_dt.replace(minute=0, second=0)

                        state["booking_info"]["time"] = parsed_dt.strftime("%H:%M")
                        logger.info(f"Extracted/Updated time: {state['booking_info']['time']}")

                except (dateutil_parser.ParserError, OverflowError) as e:
                    logger.info(f"Could not parse date/time from '{last_message}': {e}")
                except Exception as e: # Catch any other unexpected errors during parsing
                    logger.error(f"Unexpected error during date/time parsing of '{last_message}': {e}")

            # Determine if the message was primarily about date/time based on heuristics
            message_was_mainly_datetime = (contains_date_keyword or
                                           contains_day_pattern or
                                           contains_year_pattern or
                                           contains_time_keyword or
                                           contains_hour_pattern)

            # Conditions for setting purpose:
            # 1. Purpose is not already set.
            # 2. The message was not primarily identified as date/time.
            # 3. The message has a reasonable length for a purpose (e.g., more than 1 word, or more than 3 if not specific).
            # 4. We are not in a state where purpose should not be updated (e.g., CS_CONFIRMING_BOOKING_INFO).
            # 5. It's not a leftover from a booking offer affirmation where purpose was pre-filled.

            if state["booking_info"]["purpose"] is None and \
               not message_was_mainly_datetime and \
               len(last_message.split()) > 1 and \
               current_conversation_state_for_prompt != CS_CONFIRMING_BOOKING_INFO and \
               not (made_booking_offer and user_affirmed and potential_purpose_from_offer):

                # Further refine: avoid very short messages that might just be affirmations/negations if no other context
                if len(last_message.split()) < 3 and last_message.lower() in ["yes", "no", "ok", "okay", "sure", "cancel"]:
                    logger.info(f"Skipping purpose extraction for short affirmation/negation: '{last_message}'")
                else:
                    state["booking_info"]["purpose"] = last_message
                    logger.info(f"Extracted purpose: {state['booking_info']['purpose']}")
            elif not message_was_mainly_datetime and state["booking_info"]["purpose"] is not None :
                 logger.info(f"Purpose already set to '{state['booking_info']['purpose']}', not overwriting with '{last_message}'.")
            elif message_was_mainly_datetime:
                 logger.info(f"Skipping purpose extraction for '{last_message}' as it was identified as mainly date/time.")
            
            has_all_info = all(state["booking_info"].values())

            if current_conversation_state_for_prompt == CS_COLLECTING_BOOKING_INFO and has_all_info:
                state["conversation_state"] = CS_CONFIRMING_BOOKING_INFO
                logger.info(f"Transitioning state from {CS_COLLECTING_BOOKING_INFO} to {CS_CONFIRMING_BOOKING_INFO} as all info is collected.")
            elif current_conversation_state_for_prompt == CS_CONFIRMING_BOOKING_INFO:
                if any(kw in last_message.lower() for kw in ["no", "change", "wrong", "don't", "alter", "modify"]):
                    state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                    logger.info(f"User wants to change booking info. Transitioning from {CS_CONFIRMING_BOOKING_INFO} to {CS_COLLECTING_BOOKING_INFO}.")

            logger.info(f"Conversational state for LLM prompt: {current_conversation_state_for_prompt}")
            response = self.llm.invoke(prompt.format(
                input=messages[-1]["content"],
                history="\n".join([m["content"] for m in messages[:-1]]),
                agent_scratchpad="",
                name=state["booking_info"]["name"], date=state["booking_info"]["date"], time=state["booking_info"]["time"], purpose=state["booking_info"]["purpose"],
                last_action=state["last_action"], action_count=state["action_count"], has_all_info=has_all_info,
                current_conversation_state=current_conversation_state_for_prompt,
                prompt_segments=self.prompt_segments
            ))
            logger.info(f"Agent response: {response.content}")
            state["current_step"] = response.content

            llm_signaled_next_state = False
            thought_match = re.search(r"Thought:(.*?)Action:|Thought:(.*?)Final Answer:", response.content, re.DOTALL)
            if thought_match:
                thought_text = (thought_match.group(1) or thought_match.group(2) or "").strip().lower()
                if "set next state to cs_awaiting_response_to_options" in thought_text:
                    state["conversation_state"] = CS_AWAITING_RESPONSE_TO_OPTIONS
                    logger.info(f"LLM signaled to set next state to {CS_AWAITING_RESPONSE_TO_OPTIONS}.")
                    llm_signaled_next_state = True
                elif "set next state to cs_collecting_booking_info" in thought_text:
                    state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                    logger.info(f"LLM signaled to set next state to {CS_COLLECTING_BOOKING_INFO}.")
                    llm_signaled_next_state = True
                elif "set next state to cs_awaiting_final_confirmation" in thought_text: # New
                    state["conversation_state"] = CS_AWAITING_FINAL_CONFIRMATION
                    logger.info(f"LLM signaled to set next state to {CS_AWAITING_FINAL_CONFIRMATION}.")
                    llm_signaled_next_state = True
                # Add other states here if needed in the future

                # New logic for parsing effective purpose
                purpose_signal_match = re.search(r"effective current purpose: (.*)", thought_text, re.IGNORECASE)
                if purpose_signal_match:
                    extracted_llm_purpose = purpose_signal_match.group(1).strip()
                    if extracted_llm_purpose.lower() == "none":
                        extracted_llm_purpose = None # Standardize None

                    if state["booking_info"]["purpose"] != extracted_llm_purpose:
                        if extracted_llm_purpose is not None: # Only update if LLM provided a new, non-None purpose
                            # Heuristic: Allow update if new purpose is more specific or old one was very generic
                            is_old_purpose_generic = False
                            if state["booking_info"]["purpose"]:
                                generic_phrases = ["help", "process", "new here", "i dont know", "assist"]
                                if any(phrase in state["booking_info"]["purpose"].lower() for phrase in generic_phrases) and len(state["booking_info"]["purpose"].split()) > 3:
                                    is_old_purpose_generic = True

                            if is_old_purpose_generic or state["booking_info"]["purpose"] is None or \
                               (extracted_llm_purpose and state["booking_info"]["purpose"] != extracted_llm_purpose): # Allow overwriting if different
                                logger.info(f"LLM signaled effective purpose: '{extracted_llm_purpose}'. Updating from old: '{state['booking_info']['purpose']}'.")
                                state["booking_info"]["purpose"] = extracted_llm_purpose
                        elif state["booking_info"]["purpose"] is not None and extracted_llm_purpose is None:
                            # Optional: Decide if LLM saying "None" should clear an existing specific purpose. For now, let's not clear it unless explicitly handled.
                            logger.info(f"LLM signaled effective purpose: None. Current purpose '{state['booking_info']['purpose']}' will be kept unless explicitly cleared by other logic.")

            if not llm_signaled_next_state and \
               current_conversation_state_for_prompt == CS_AWAITING_RESPONSE_TO_OPTIONS and \
               "Action:" not in response.content:
                state["conversation_state"] = CS_GENERAL_INQUIRY
                logger.info(f"After processing CS_AWAITING_RESPONSE_TO_OPTIONS with a Final Answer, transitioning next state to {CS_GENERAL_INQUIRY}.")

            return state

        def call_tool(state: AgentState) -> AgentState:
            current_step = state["current_step"]
            logger.info(f"Executing tool with step: {current_step}")
            
            action_match = re.search(r"Action: (\w+)", current_step)
            action_input_match = re.search(r"Action Input: (.*?)(?=\n|$)", current_step)
            
            no_input_tools = ["ViewAppointments", "HandleGreeting", "GetPurposeExamples"]

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
                    state["conversation_state"] = CS_GENERAL_INQUIRY
                    return state
                
                result_content = ""
                if action in tool_map:
                    selected_tool = tool_map[action]
                    tool_input = None
                    action_input_str = ""

                    if selected_tool.name in no_input_tools:
                        if action_input_match and action_input_match.group(1) and action_input_match.group(1).strip():
                            actual_input_provided = action_input_match.group(1).strip()
                            logger.warning(f"LLM attempted to provide input '{actual_input_provided}' for no-input tool '{selected_tool.name}'. Ignoring input.")
                        tool_input = None
                        action_input_str = ""
                    elif selected_tool.name == "BookAppointment":
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
                                result_content = f"Error: Expected JSON input for BookAppointment but received: {action_input_str}"
                                state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                                logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to non-JSON input for BookAppointment.")
                        else:
                            logger.error("Error: Missing input for BookAppointment.")
                            result_content = "Error: Missing Action Input for BookAppointment."
                            state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                            logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to missing input for BookAppointment.")

                    if not result_content:
                        try:
                            logger.info(f"Executing {selected_tool.name} tool with input: {tool_input}")
                            result = selected_tool.func(tool_input)
                            logger.info(f"Tool result: {result}")

                            if selected_tool.name == "BookAppointment":
                                if "error" in result.lower() or "failed" in result.lower():
                                    state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                                    logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to BookAppointment operational error: {result}")
                                elif "successfully booked" in result.lower():
                                    state["booking_info"] = {"name": None, "date": None, "time": None, "purpose": None}
                                    state["last_action"] = None
                                    state["action_count"] = 0
                                    state["conversation_state"] = CS_POST_BOOKING_FEEDBACK
                                    logger.info(f"Transitioning state to {CS_POST_BOOKING_FEEDBACK} after successful booking.")
                            elif selected_tool.name == "HandleGreeting":
                                state["conversation_state"] = CS_GENERAL_INQUIRY
                                logger.info(f"Transitioning state to {CS_GENERAL_INQUIRY} after HandleGreeting.")
                            elif selected_tool.name in ["ViewAppointments", "GetPurposeExamples"]:
                                state["conversation_state"] = CS_GENERAL_INQUIRY
                                logger.info(f"Transitioning state to {CS_GENERAL_INQUIRY} after {selected_tool.name}.")

                            if action_input_str:
                                result_content = f"Action: {action}\nAction Input: {action_input_str}\nObservation: {result}"
                            else:
                                result_content = f"Action: {action}\nObservation: {result}"

                        except Exception as e:
                            logger.error(f"Tool execution error for {selected_tool.name}: {e}")
                            result_content = f"Error executing {action}: {str(e)}"
                            state["conversation_state"] = CS_HANDLING_TOOL_ERROR
                            logger.info(f"Transitioning state to {CS_HANDLING_TOOL_ERROR} due to tool execution exception.")
                else:
                    logger.error(f"Error: Tool '{action}' not found.")
                    result_content = f"Error: Tool '{action}' not found."
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

