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
CS_CONFIRMING_NAME_SPELLING = "CONFIRMING_NAME_SPELLING" # New state
CS_CONFIRMING_BOOKING_INFO = "CONFIRMING_BOOKING_INFO"
CS_VIEWING_APPOINTMENTS = "VIEWING_APPOINTMENTS"
CS_PROVIDING_EXAMPLES = "PROVIDING_EXAMPLES"
CS_AWAITING_RESPONSE_TO_OPTIONS = "AWAITING_RESPONSE_TO_OPTIONS"
CS_CLARIFYING_INPUT = "CLARIFYING_INPUT"
CS_HANDLING_TOOL_ERROR = "HANDLING_TOOL_ERROR"
CS_POST_BOOKING_FEEDBACK = "POST_BOOKING_FEEDBACK"
CS_ENDING_CONVERSATION = "ENDING_CONVERSATION"

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    next: str
    current_step: str
    booking_info: Dict[str, Any] # Will include name, date, time, purpose, email
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
            "name": None, "date": None, "time": None, "purpose": None, "email": None
        }
        self.prompt_segments = {
            CS_INITIAL_GREETING: (
                "You are in the INITIAL_GREETING state. This is the user's first interaction. "
                "Analyze the user's first message ({input}). "
                "1. If the user's message is ONLY a simple greeting (e.g., 'hi', 'hello'): Use HandleGreeting tool. "
                "2. If the user asks for general help or what you can do: Provide a welcoming, informative response (e.g., 'Hello! I can book, view, or show examples for appointments. What would you like to do?'). Do NOT use HandleGreeting. "
                "3. If the user's first message clearly indicates an intent to book an appointment OR provides any booking details (like name, date, time, or purpose, e.g., 'I want to book an appointment for August 2nd' or 'My name is John, I need an appointment'): "
                "   Your Thought must be: 'User wants to book and may have provided some info. I will transition to collecting booking information. I will check what info is provided ({input}) and what is in {booking_info}, then ask for the *first* piece of missing information (Name -> Date -> Time -> Purpose -> Email). If they provided a name like 'My name is X', my thought should include `Extracted Name: X`. If they provided a date like 'for August 2nd', my thought should include `Extracted Date: YYYY-08-02` (ensure YYYY-MM-DD). If they provided a purpose like 'for a checkup', my thought should include `Extracted Purpose: checkup`.' "
                "   Then, set conversation_state to CS_COLLECTING_BOOKING_INFO. "
                "   Your Final Answer should be to ask for the *first* missing piece of information based on the sequence: Name, Date, Time, Purpose, Email. For example, if they only said 'book appointment', ask for name. If they said 'book for August 2nd', ask for name. If they said 'My name is John, book for August 2nd', ask for time. "
                "Your goal is to quickly move to `CS_COLLECTING_BOOKING_INFO` if booking is intended."
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

                "If the user previously indicated they are new or asked for general help, and you haven't yet provided substantial guidance: "
                "Be proactive. If the user seems unsure what to do, you can offer to explain the booking process. If they seem unsure about the reason for their appointment after you've engaged them, try to ask a clarifying question like 'What is the main reason for your visit?' before offering generic examples. "
                "Example for a new user who seems unsure: 'Since you're new, I can quickly explain how booking works. What would you like to do?' "

                "For other general inquiries, or if the user seems to be exploring options after declining a specific offer: "
                "Listen carefully, politely ask clarifying questions to guide them towards a task you can perform (booking, viewing). "
                "If they are trying to book but are unsure about the purpose, first ask them directly, e.g., 'What's the reason for your appointment today?'. Only if they express difficulty in naming a purpose or ask for types of appointments should you then consider using GetPurposeExamples. Focus on their last statement. "
                "Your aim is to understand their current need and help them navigate to a solution."
            ),
            CS_COLLECTING_BOOKING_INFO: (
                "You are in the COLLECTING_BOOKING_INFO state. Your goal is to fill all fields in {booking_info}: name, date, time, purpose, and email. "
                "ALWAYS check the current {booking_info} to see what is already filled and what is still None. Ask for details sequentially. "
                "If the user provides multiple pieces of information in their response (e.g., 'My name is John and I want an appointment for tomorrow for a checkup'), your Thought MUST try to extract all of them using the `Extracted ...: [value]` format for each (e.g., 'Thought: User provided name, date, and purpose. Extracted Name: John. Extracted Date: YYYY-MM-DD (for tomorrow). Extracted Purpose: checkup. I will now ask for the time...'). Then, ask for the NEXT missing piece of information in the sequence. "
                "Sequence of collection: Name -> Date -> Time -> Purpose -> Email. "
                "1. If {booking_info[name]} is None: Ask for the name (e.g., 'May I have your name for the booking?'). If the user provides a name, your Thought must include `Extracted Name: [User's Response]`. Then, your Thought should be: 'I will assume this is the name. I must now confirm the spelling.' Set conversation_state to CS_CONFIRMING_NAME_SPELLING. Your Final Answer should be to ask for spelling confirmation: 'Thank you. I have your name as [User's Response]. Is that spelled correctly?' "
                "2. If {booking_info[name]} is not None (spelling confirmed), and {booking_info[date]} is None: Ask for the date (e.g., 'What date would you like for this appointment?'). If the user provides a date, your Thought must include `Extracted Date: [YYYY-MM-DD format of date]`. "
                "3. If name and date are not None, and {booking_info[time]} is None: Ask for the time (e.g., 'What time would you like?'). If the user provides a time, your Thought must include `Extracted Time: [HH:MM format of time]`. "
                "4. If name, date, and time are not None, and {booking_info[purpose]} is None: Ask for the purpose (e.g., 'What is the purpose of this appointment?'). If the user provides a purpose, your Thought must include `Extracted Purpose: [purpose provided]`. "
                "5. If name, date, time, and purpose are not None, and {booking_info[email]} is None: Ask for the email (e.g., 'May I have your email address for this booking?'). If the user provides an email, your Thought must include `Extracted Email: [user's email address]`. "
                "Acknowledge information briefly if you are asking for the next piece in the same turn. "
                "Once {booking_info[name]}, {booking_info[date]}, {booking_info[time]}, {booking_info[purpose]}, and {booking_info[email]} are ALL FILLED (not None), your Thought must be: 'All required information collected: Name: {booking_info[name]}, Date: {booking_info[date]}, Time: {booking_info[time]}, Purpose: {booking_info[purpose]}, Email: {booking_info[email]}. I will now summarize for final confirmation.' Then, set conversation_state to CS_CONFIRMING_BOOKING_INFO. Your Final Answer should summarize these details and asks 'Is this all correct?'"
            ),
            CS_CONFIRMING_NAME_SPELLING: (
                "You are in the CONFIRMING_NAME_SPELLING state. The current name recorded is {booking_info[name]}. You have just asked the user if this spelling is correct. "
                "Now, evaluate the user's response ({input}). "
                "If the user confirms (e.g., 'yes', 'that's right'): Your Thought should be: 'Name spelling confirmed for {booking_info[name]}. I will check if all other booking info (date, time, purpose) is collected. If so, transition to CS_CONFIRMING_BOOKING_INFO. Otherwise, transition to CS_COLLECTING_BOOKING_INFO to ask for the next missing field.' Your Final Answer should be to proceed (e.g., 'Great! Now, what date...?' or the full summary if all info is now present). "
                "If the user denies or indicates the spelling is wrong (e.g., 'no', 'that's incorrect'): Your Thought must be: 'User says name {booking_info[name]} is incorrect. I need to ask for the correct name again. RESET_NAME_FLAG'. Set conversation_state to CS_COLLECTING_BOOKING_INFO. Your Final Answer should be: 'My apologies. Could you please provide me with the correct spelling of your name?' "
            ),
            CS_CONFIRMING_BOOKING_INFO: (
                "You are in the CONFIRMING_BOOKING_INFO state. You have summarized all details: Name: {booking_info[name]}, Date: {booking_info[date]}, Time: {booking_info[time]}, Purpose: {booking_info[purpose]}, Email: {booking_info[email]}. You asked 'Is this all correct?'. "
                "Now, evaluate the user's response ({input}). "
                "If the user confirms (e.g., 'yes', 'correct', 'all good', 'that's right', 'okay', 'ok', 'confirmed'): "
                "Your Thought should be: 'All details confirmed by user. I will now book the appointment.' "
                "You MUST output: Action: BookAppointment Action Input: {{\"name\": \"{booking_info[name]}\", \"date\": \"{booking_info[date]}\", \"time\": \"{booking_info[time]}\", \"purpose\": \"{booking_info[purpose]}\", \"email\": \"{booking_info[email]}\"}}. "
                "Do NOT output a Final Answer until after the booking action is complete. "
                "If the user denies or wants to change something (e.g., 'no, the date is wrong', 'actually, can we change the time?', 'my email is incorrect'): "
                "Your Thought must identify the field they want to change (name, date, time, purpose, or email). Then, include the appropriate reset flag in your thought: `RESET_NAME_FLAG` for name, `RESET_DATE_FLAG` for date, `RESET_TIME_FLAG` for time, `RESET_PURPOSE_FLAG` for purpose, or `RESET_EMAIL_FLAG` for email. "
                "Then, set conversation_state to CS_COLLECTING_BOOKING_INFO. Your Final Answer should ask for the corrected information for that specific field. For example, if they said 'the email is wrong', your Final Answer could be 'Okay, what is the correct email address?'."
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
        You are a friendly and helpful appointment booking assistant. Your goal is to assist users efficiently.
        **Keep your responses concise, clear, and use natural, everyday language. Avoid being overly formal or verbose unless necessary for clarification.**
        You have access to the following tools:
        
        {tools}
        
        Current booking information:
        Name: {name}
        Date: {date}
        Time: {time}
        Purpose: {purpose}
        Email: {email}
        
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
        20. When asking for information, be specific about what you need (e.g., name, date, time, purpose). For purpose, you can say: "Please provide the purpose of your appointment"
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
                "name", "date", "time", "purpose", "email",
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

            # --- BEGIN SLOT FILLING EXTRACTION ---
            # Extract booking info from agent response (LLM output)
            agent_response = messages[-1]["content"]
            # Use regex to extract cues
            name_match = re.search(r'Extracted Name: ([^\n]+)', agent_response)
            date_match = re.search(r'Extracted Date: ([0-9]{4}-[0-9]{2}-[0-9]{2})', agent_response)
            time_match = re.search(r'Extracted Time: ([0-9]{2}:[0-9]{2})', agent_response)
            purpose_match = re.search(r'Extracted Purpose: ([^\n]+)', agent_response)
            email_match = re.search(r'Extracted Email: ([^\s@]+@[^\s@]+\.[^\s@]+)', agent_response) # Basic email regex

            if name_match:
                state["booking_info"]["name"] = name_match.group(1).strip()
                logger.info(f"Slot-filling: Extracted name: {state['booking_info']['name']}")
            if date_match:
                state["booking_info"]["date"] = date_match.group(1).strip()
                logger.info(f"Slot-filling: Extracted date: {state['booking_info']['date']}")
            if time_match:
                state["booking_info"]["time"] = time_match.group(1).strip()
                logger.info(f"Slot-filling: Extracted time: {state['booking_info']['time']}")
            if purpose_match:
                state["booking_info"]["purpose"] = purpose_match.group(1).strip()
                logger.info(f"Slot-filling: Extracted purpose: {state['booking_info']['purpose']}")
            if email_match:
                state["booking_info"]["email"] = email_match.group(1).strip()
                logger.info(f"Slot-filling: Extracted email: {state['booking_info']['email']}")
            # --- END SLOT FILLING EXTRACTION ---

            # --- START: More Robust State Transition Logic ---
            thought_content_lower = ""
            response = self.llm.invoke(prompt.format(
                input=messages[-1]["content"],
                history="\n".join([m["content"] for m in messages[:-1]]),
                agent_scratchpad="",
                name=state["booking_info"]["name"], date=state["booking_info"]["date"], time=state["booking_info"]["time"], purpose=state["booking_info"]["purpose"], email=state["booking_info"]["email"],
                last_action=state["last_action"], action_count=state["action_count"], has_all_info=all(val is not None for val in state["booking_info"].values()),
                current_conversation_state=current_conversation_state_for_prompt,
                prompt_segments=self.prompt_segments
            ))

            processed_content = response.content
            if "RESET_NAME_FLAG" in processed_content:
                logger.info("RESET_NAME_FLAG found in LLM response. Clearing booking_info['name'] for re-collection.")
                state["booking_info"]["name"] = None
                # Remove the flag from the content that goes into current_step to keep agent_scratchpad clean
                processed_content = processed_content.replace("RESET_NAME_FLAG", "").strip()
            if "RESET_DATE_FLAG" in processed_content: # Assuming similar flags for other fields might be used
                logger.info("RESET_DATE_FLAG found in LLM response. Clearing booking_info['date'].")
                state["booking_info"]["date"] = None
                processed_content = processed_content.replace("RESET_DATE_FLAG", "").strip()
            if "RESET_TIME_FLAG" in processed_content:
                logger.info("RESET_TIME_FLAG found in LLM response. Clearing booking_info['time'].")
                state["booking_info"]["time"] = None
                processed_content = processed_content.replace("RESET_TIME_FLAG", "").strip()
            if "RESET_PURPOSE_FLAG" in processed_content:
                logger.info("RESET_PURPOSE_FLAG found in LLM response. Clearing booking_info['purpose'].")
                state["booking_info"]["purpose"] = None
                processed_content = processed_content.replace("RESET_PURPOSE_FLAG", "").strip()
            if "RESET_EMAIL_FLAG" in processed_content:
                logger.info("RESET_EMAIL_FLAG found in LLM response. Clearing booking_info['email'].")
                state["booking_info"]["email"] = None
                processed_content = processed_content.replace("RESET_EMAIL_FLAG", "").strip()

            logger.info(f"Agent response (raw): {response.content}") # Log raw response
            logger.info(f"Agent response (processed for current_step): {processed_content}")
            state["current_step"] = processed_content

            thought_match_for_state = re.search(r"Thought:(.*?)(?:\nAction:|\nFinal Answer:)", response.content, re.DOTALL)
            if thought_match_for_state:
                thought_content_lower = (thought_match_for_state.group(1) or "").strip().lower()

            # Explicitly check for state transition cues from LLM's thought
            llm_signaled_collecting = "transition to collecting booking information" in thought_content_lower or \
                                      "set conversation_state to cs_collecting_booking_info" in thought_content_lower or \
                                      any(cue in response.content for cue in ["Extracted Name:", "Extracted Date:", "Extracted Time:", "Extracted Purpose:", "Extracted Email:"])

            if state.get("conversation_state") == CS_INITIAL_GREETING:
                user_input_lower = messages[-1]["content"].lower()
                is_first_turn = len(messages) <= 2 # Approx first user input after initial agent greeting

                booking_keywords = ["book", "appointment", "schedule", "meeting"]
                simple_greeting_keywords = ["hi", "hello", "hey"]

                contains_booking_intent = any(keyword in user_input_lower for keyword in booking_keywords)
                is_just_simple_greeting = any(keyword in user_input_lower.split() for keyword in simple_greeting_keywords) and not contains_booking_intent

                if is_first_turn and not is_just_simple_greeting: # If first user message is not just a greeting
                    if llm_signaled_collecting:
                        state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                        logger.info(f"LLM signaled transition from {CS_INITIAL_GREETING} to {CS_COLLECTING_BOOKING_INFO}.")
                    elif contains_booking_intent: # Force transition if booking intent detected, even if LLM didn't signal
                        state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                        logger.info(f"Booking intent detected in first user message. Forcefully transitioning from {CS_INITIAL_GREETING} to {CS_COLLECTING_BOOKING_INFO}.")
                    # If it was a general question, it will stay in INITIAL_GREETING for LLM to handle as per prompt point 2.
                elif llm_signaled_collecting: # Also transition if LLM signals it even if not first turn (e.g. user clarifies after greeting)
                    state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                    logger.info(f"LLM signaled transition from {CS_INITIAL_GREETING} to {CS_COLLECTING_BOOKING_INFO} (not first turn).")

            # If in CS_COLLECTING_BOOKING_INFO, generally stay there until all info is collected or LLM signals otherwise
            if state.get("conversation_state") == CS_COLLECTING_BOOKING_INFO:
                # Check if all info is now filled (name, date, time, purpose, email)
                # Note: The 'email' field was added to self.booking_info.
                # The has_all_info check needs to be aware of all required fields.
                required_fields = ["name", "date", "time", "purpose", "email"]
                has_all_info_now = all(state["booking_info"].get(field) is not None for field in required_fields)

                if has_all_info_now:
                    # This was the logic for CS_COLLECTING_BOOKING_INFO prompt:
                    # "Once {booking_info[name]}, {booking_info[date]}, {booking_info[time]}, {booking_info[purpose]}, and {booking_info[email]} are ALL FILLED (not None),
                    # your Thought must be: 'All required information collected: ... I will now summarize for final confirmation.'
                    # Then, set conversation_state to CS_CONFIRMING_BOOKING_INFO."
                    # So, if LLM's thought reflects this, it will set the state.
                    # We might not need to force it here if LLM follows prompt.
                    pass # LLM should handle transition to CS_CONFIRMING_BOOKING_INFO as per its prompt
                else:
                    # If still collecting, ensure the state remains CS_COLLECTING_BOOKING_INFO
                    # unless LLM explicitly signals a different state (e.g. user asks to view examples).
                    # The llm_signaled_next_state logic later will handle explicit signals.
                    if not state.get("llm_signaled_next_state"): # placeholder for more advanced check
                         state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
            # --- END: State Transition Logic ---

            has_all_info = all(val is not None for val in state["booking_info"].values()) # Original check, might need adjustment based on required_fields

            # If all info is collected and user confirms, insert to DB directly (This was specific to CS_CONFIRMING_BOOKING_INFO)
            # This direct DB insert might be too aggressive here. Let LLM drive via BookAppointment tool.
            # Consider if this block is still needed or if it should only be triggered from CS_CONFIRMING_BOOKING_INFO state + user affirmation.
            # For now, let's assume this user_affirmed check is for when the agent has ALREADY presented all info and is in CS_CONFIRMING_BOOKING_INFO.
            if current_conversation_state_for_prompt == CS_CONFIRMING_BOOKING_INFO and has_all_info and user_affirmed:
                from .database import AppointmentDB
                db = AppointmentDB()
                result = db.book_appointment(
                    state["booking_info"]["name"],
                    state["booking_info"]["date"],
                    state["booking_info"]["time"],
                    state["booking_info"]["purpose"],
                    email=state["booking_info"]["email"]
                )
                logger.info(f"Direct DB insert result: {result}")
                # Reset booking_info for next booking
                state["booking_info"] = {"name": None, "date": None, "time": None, "purpose": None, "email": None}
                # Add a message to the conversation
                state["messages"].append({"role": "assistant", "content": f"Appointment booked successfully! {result}"})
                # Optionally, set state to POST_BOOKING_FEEDBACK or similar
                state["conversation_state"] = CS_POST_BOOKING_FEEDBACK
                # Return early to avoid LLM call
                return state

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

            # Conditional auto-filling of booking_info based on user's last message
            # Only attempt to auto-fill if we are in a state where user is providing them,
            # or if it's a general inquiry where they might offer info spontaneously.
            # Avoid auto-filling if we are in CS_CONFIRMING_BOOKING_INFO.
            # For CS_CONFIRMING_NAME_SPELLING, name extraction is handled by LLM logic.
            # If name was reset (is None) and we are in CS_COLLECTING_BOOKING_INFO, it's okay to try auto-fill name.
            # Only handle RESET_..._FLAGs below

            if current_conversation_state_for_prompt == CS_COLLECTING_BOOKING_INFO and has_all_info:
                state["conversation_state"] = CS_CONFIRMING_BOOKING_INFO
                logger.info(f"Transitioning state from {CS_COLLECTING_BOOKING_INFO} to {CS_CONFIRMING_BOOKING_INFO} as all info is collected.")
            elif current_conversation_state_for_prompt == CS_CONFIRMING_BOOKING_INFO:
                if any(kw in last_message.lower() for kw in ["no", "change", "wrong", "don't", "alter", "modify"]):
                    state["conversation_state"] = CS_COLLECTING_BOOKING_INFO
                    logger.info(f"User wants to change booking info. Transitioning from {CS_CONFIRMING_BOOKING_INFO} to {CS_COLLECTING_BOOKING_INFO}.")

            logger.info(f"Conversational state for LLM prompt: {current_conversation_state_for_prompt}")

            llm_signaled_next_state = False
            thought_match = re.search(r"Thought:(.*?)Action:|Thought:(.*?)Final Answer:", response.content, re.DOTALL)
            if thought_match:
                thought_text = (thought_match.group(1) or thought_match.group(2) or "").strip().lower()
                if "set next state to cs_awaiting_response_to_options" in thought_text:
                    state["conversation_state"] = CS_AWAITING_RESPONSE_TO_OPTIONS
                    logger.info(f"LLM signaled to set next state to {CS_AWAITING_RESPONSE_TO_OPTIONS}.")
                    llm_signaled_next_state = True

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
                                    state["booking_info"] = {"name": None, "date": None, "time": None, "purpose": None, "email": None}
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

