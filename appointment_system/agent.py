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
        self.last_action = None
        self.action_count = 0
        
    def create_agent(self):
        # Get the tools
        tools = [Tool(**tool) for tool in self.tools_handler.get_tools()]
        logger.info(f"Available tools: {[tool.name for tool in tools]}")
        
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
        {{"name": "person name", "date": "dd/mm/yyyy", "time": "number with am/pm", "purpose": "appointment purpose"}}
        
        Important Instructions:
        1. Be friendly and conversational in your responses.
        2. If the user's input is unclear or too short (like "hi", "hello", etc.), use the HandleGreeting tool ONCE.
        3. When booking appointments:
           - First, ask for the person's name if not already provided
           - Then, ask for their preferred date (in dd/mm/yyyy format) if not already provided
           - Next, ask for their preferred time (in number with am/pm format) if not already provided
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
            - For dates: "Please provide the date in dd/mm/yyyy format (e.g., 11/09/2025)"
            - For time: "Please provide the time in number with am/pm format (e.g., 11am or 2:30pm)"
            - For purpose: "Please provide the purpose of your appointment"
        21. If the user provides information in the wrong format, politely ask them to provide it in the correct format.
        22. If the user asks for examples of valid purposes, use the GetPurposeExamples tool.
        23. When you have all required information (has_all_info is true), use the BookAppointment tool to book the appointment.
        24. After successfully booking an appointment, confirm the details with the user and ask if they need anything else.
        25. If the user provides information that's already in the booking_info, acknowledge it and ask for the next missing piece of information.
        
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
            
            # Check for date (dd/mm/yyyy format)
            date_match = re.search(r'\d{2}/\d{2}/\d{4}', last_message)
            if date_match:
                state["booking_info"]["date"] = date_match.group(0)
            
            # Check for time (number with am/pm format)
            time_match = re.search(r'\d{1,2}(?::\d{2})?\s*(?:am|pm)', last_message.lower())
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
                last_action=self.last_action,
                action_count=self.action_count,
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
                if action == self.last_action:
                    self.action_count += 1
                else:
                    self.last_action = action
                    self.action_count = 1
                
                # Force final answer if too many consecutive actions
                if self.action_count >= 3:
                    logger.info("Too many consecutive actions, forcing final answer")
                    state["current_step"] = f"Thought: I've used the same tool too many times. I should provide a final answer.\nFinal Answer: I apologize for the confusion. Let me help you with that directly. What would you like to do?"
                    return state
                
                # Handle tools that don't need input
                if action in ["ViewAppointments", "HandleGreeting", "GetPurposeExamples"]:
                    try:
                        logger.info(f"Executing {action} tool")
                        tool_index = {
                            "ViewAppointments": 1,
                            "HandleGreeting": 2,
                            "GetPurposeExamples": 3
                        }[action]
                        result = tools[tool_index].func(None)
                        logger.info(f"Tool result: {result}")
                        state["messages"].append({
                            "role": "assistant",
                            "content": f"Action: {action}\nObservation: {result}"
                        })
                    except Exception as e:
                        logger.error(f"Tool execution error: {e}")
                        state["messages"].append({
                            "role": "assistant",
                            "content": f"Error executing {action}: {str(e)}"
                        })
                else:
                    # Handle BookAppointment tool
                    if action_input_match:
                        action_input = action_input_match.group(1).strip()
                        logger.info(f"Tool input: {action_input}")
                        
                        # Try to parse action input as JSON
                        if action_input.startswith('{') and action_input.endswith('}'):
                            try:
                                action_input = json.loads(action_input)
                                logger.info(f"Parsed JSON input: {action_input}")
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON: {e}")
                        
                        # Execute the tool
                        try:
                            logger.info(f"Executing tool: {action}")
                            result = tools[0].func(action_input)  # BookAppointment is the first tool
                            logger.info(f"Tool result: {result}")
                            state["messages"].append({
                                "role": "assistant",
                                "content": f"Action: {action}\nAction Input: {action_input}\nObservation: {result}"
                            })
                            # Reset booking info after successful booking
                            state["booking_info"] = {
                                "name": None,
                                "date": None,
                                "time": None,
                                "purpose": None
                            }
                            # Reset action tracking
                            self.last_action = None
                            self.action_count = 0
                        except Exception as e:
                            logger.error(f"Tool execution error: {e}")
                            state["messages"].append({
                                "role": "assistant",
                                "content": f"Error executing {action}: {str(e)}"
                            })
            
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
