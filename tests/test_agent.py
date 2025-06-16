import unittest
from unittest.mock import patch, MagicMock, call
import re
import json
import os

# Attempt to import from the project structure
# This assumes 'appointment_system' is in the Python path or PYTHONPATH is set up
from datetime import datetime as dt, date as test_date, time as test_time, timedelta # Use 'dt' to avoid conflict
import json # Added for json.dumps in confirmation flow test
try:
    from appointment_system.agent import (
        AppointmentAgent, AgentState, CustomPromptTemplate,
        CS_GENERAL_INQUIRY, CS_COLLECTING_BOOKING_INFO, CS_INITIAL_GREETING,
        CS_CONFIRMING_BOOKING_INFO, CS_AWAITING_FINAL_CONFIRMATION # New states
    )
    from appointment_system.tools import AppointmentTools
except ImportError:
    # Fallback for environments where direct import might be tricky
    print("Warning: Could not import full agent classes, using stubs for testing where necessary.")
    class AgentState(dict): pass
    class CustomPromptTemplate:
        def __init__(self, template, tools, input_variables): self.template = template; self.tools = tools; self.input_variables = input_variables
        def format(self, **kwargs): return self.template.format(**kwargs)
    class AppointmentAgent: # Stub
        def __init__(self): self.llm = MagicMock(); self.tools_handler = MagicMock(); self.tool_map = {}
        def create_agent(self): return MagicMock() # Returns a mock graph
    class AppointmentTools: # Stub
        def get_tools(self): return []
    CS_INITIAL_GREETING = "INITIAL_GREETING"
    CS_GENERAL_INQUIRY = "GENERAL_INQUIRY"
    CS_COLLECTING_BOOKING_INFO = "COLLECTING_BOOKING_INFO"
    CS_CONFIRMING_BOOKING_INFO = "CONFIRMING_BOOKING_INFO" # New stub
    CS_AWAITING_FINAL_CONFIRMATION = "AWAITING_FINAL_CONFIRMATION" # New stub


# It's important that the subtask can actually access and import these.
# The following path manipulations are attempts to help the subtask find modules if they are not in PYTHONPATH.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# This line assumes tests/test_agent.py is one level down from the project root where appointment_system/ is.


class TestAgentLogic(unittest.TestCase):

    def setUp(self):
        # Mock environment variables if necessary (e.g., for GOOGLE_API_KEY)
        self.env_patch = patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_api_key'})
        self.env_patch.start()

        # Mock AppointmentTools and its get_tools method
        self.mock_tools_instance = MagicMock(spec=AppointmentTools)
        self.mock_book_appointment_tool = MagicMock(name="BookAppointment_func")
        self.mock_view_appointments_tool = MagicMock(name="ViewAppointments_func")

        # Define what get_tools returns
        # The 'func' needs to be callable.
        # The 'description' is used by CustomPromptTemplate.
        self.tools_config = [
            {'name': 'BookAppointment', 'func': self.mock_book_appointment_tool, 'description': 'Books an appt'},
            {'name': 'ViewAppointments', 'func': self.mock_view_appointments_tool, 'description': 'Views appts'},
        ]
        self.mock_tools_instance.get_tools.return_value = self.tools_config

        # Patch the __init__ of AppointmentTools to return our mock instance
        self.tools_patcher = patch('appointment_system.agent.AppointmentTools', return_value=self.mock_tools_instance)
        self.MockAppointmentTools = self.tools_patcher.start()

        # Mock ChatGoogleGenerativeAI
        self.llm_patcher = patch('appointment_system.agent.ChatGoogleGenerativeAI')
        self.MockChatGoogleGenerativeAI = self.llm_patcher.start()
        self.mock_llm_instance = self.MockChatGoogleGenerativeAI.return_value
        self.mock_llm_instance.invoke.return_value = MagicMock(content="Final Answer: Default mock response")

        # Now instantiate the agent - it will use the mocked classes
        self.agent_wrapper = AppointmentAgent()
        self.graph = self.agent_wrapper.create_agent()

        # Initial state for invoking the graph
        self.initial_state = {
            "messages": [{"role": "user", "content": "Hello"}],
            "next": "agent",
            "current_step": "",
            "booking_info": {"name": None, "date": None, "time": None, "purpose": None},
            "last_action": None,
            "action_count": 0
        }

    def tearDown(self):
        self.env_patch.stop()
        self.tools_patcher.stop()
        self.llm_patcher.stop()

    def test_datetime_regex_extraction_in_call_agent(self):
        # This tests the regex as used in call_agent, so we need to simulate its invocation
        # by controlling the LLM's output if call_agent directly uses it, or by directly testing.
        # The regexes are: date_regex = r'\d{4}-\d{2}-\d{2}' and time_regex = r'\d{2}:\d{2}'
        # For simplicity, we'll test the regexes directly here.

        # Access the actual regex from the agent's implementation if possible or re-define
        date_regex = r'\d{4}-\d{2}-\d{2}' # As defined in agent's call_agent
        time_regex = r'\d{2}:\d{2}' # As defined in agent's call_agent

        self.assertIsNotNone(re.search(date_regex, "Please book for 2024-07-25"))
        self.assertIsNone(re.search(date_regex, "Please book for 25/07/2024"))
        self.assertEqual(re.search(date_regex, "Date is 2024-03-10.").group(0), "2024-03-10")

        self.assertIsNotNone(re.search(time_regex, "at 14:30 thanks"))
        self.assertIsNone(re.search(time_regex, "at 2:30pm"))
        self.assertEqual(re.search(time_regex, "Time: 09:00.").group(0), "09:00")

    def test_prompt_formatting_includes_date_time_instructions(self):
        # Get the prompt template from the agent
        # This requires CustomPromptTemplate to be correctly imported and used by AppointmentAgent
        # We check the formatted prompt string passed to the LLM

        # Simulate a call that would format the prompt
        _ = self.graph.invoke(self.initial_state)

        # Assuming the prompt is the first arg to llm.invoke
        # The actual prompt string is complex, we're checking for substrings
        formatted_prompt_args = self.mock_llm_instance.invoke.call_args[0]
        prompt_string = formatted_prompt_args[0] # The formatted prompt string

        self.assertIn("YYYY-MM-DD format", prompt_string)
        self.assertIn("HH:MM 24-hour format", prompt_string)
        self.assertIn('{"name": "person name", "date": "YYYY-MM-DD", "time": "HH:MM", "purpose": "appointment purpose"}', prompt_string)
        self.assertIn("summarize all currently known booking details", prompt_string)

    def test_state_variables_in_prompt(self):
        state = {**self.initial_state, "last_action": "ViewAppointments", "action_count": 1}
        _ = self.graph.invoke(state)

        formatted_prompt_args = self.mock_llm_instance.invoke.call_args[0]
        prompt_string = formatted_prompt_args[0]

        self.assertIn("Last action taken: ViewAppointments", prompt_string)
        self.assertIn("Number of consecutive actions: 1", prompt_string)

    def test_tool_calling_logic_viewappointments(self):
        # Simulate LLM output that asks to call ViewAppointments
        self.mock_llm_instance.invoke.return_value = MagicMock(
            content="Thought: I need to view appointments.\nAction: ViewAppointments\nAction Input: None"
        )

        state = {**self.initial_state, "messages": [{"role": "user", "content": "View my appointments"}]}
        final_state = self.graph.invoke(state)

        self.mock_view_appointments_tool.assert_called_once_with(None)
        # Check if state['last_action'] and state['action_count'] are updated
        # Note: The graph invoke will run agent, then tool, then agent again.
        # The final state's last_action might be from the second agent call.
        # To check intermediate state is harder without graph inspection tools.
        # For this test, we focus on the tool being called.
        # The prompt for the *next* LLM call would contain the updated last_action.

        # Verify that the prompt for the next LLM call (after tool execution) includes the updated last_action
        # The mock_llm_instance is called multiple times in one graph.invoke for tool use.
        # First call: To decide to use a tool. Second call: After observing tool output.
        second_llm_call_args = self.mock_llm_instance.invoke.call_args_list[1][0]
        prompt_string_after_tool = second_llm_call_args[0]
        self.assertIn("Last action taken: ViewAppointments", prompt_string_after_tool)
        self.assertIn("Number of consecutive actions: 1", prompt_string_after_tool)


    def test_tool_calling_logic_bookappointment(self):
        action_input_dict = {"name": "Test User", "date": "2024-03-15", "time": "10:00", "purpose": "Checkup"}
        action_input_json = json.dumps(action_input_dict)

        self.mock_llm_instance.invoke.return_value = MagicMock(
            content=f"Thought: I have all info, book it.\nAction: BookAppointment\nAction Input: {action_input_json}"
        )
        self.mock_book_appointment_tool.return_value = "Successfully booked." # Simulate tool success

        state = {**self.initial_state, "messages": [{"role": "user", "content": "Book for Test User..."}]}
        final_state = self.graph.invoke(state)

        self.mock_book_appointment_tool.assert_called_once_with(action_input_dict)

        # Check that last_action and action_count are reset after successful booking
        # This would be reflected in the prompt for the LLM call *after* booking.
        second_llm_call_args = self.mock_llm_instance.invoke.call_args_list[1][0]
        prompt_string_after_booking = second_llm_call_args[0]
        self.assertIn("Last action taken: None", prompt_string_after_booking) # Assuming it's reset to None
        self.assertIn("Number of consecutive actions: 0", prompt_string_after_booking)


    def test_action_count_increment_and_reset(self):
        # Simulate LLM deciding to use ViewAppointments three times
        llm_response_view = "Thought: View again.\nAction: ViewAppointments\nAction Input: None"
        self.mock_llm_instance.invoke.return_value = MagicMock(content=llm_response_view)
        self.mock_view_appointments_tool.return_value = "Some appointments."

        # First call
        state = self.graph.invoke(self.initial_state)
        # Prompt for 2nd agent turn (after 1st ViewAppointments)
        prompt_after_1st_call = self.mock_llm_instance.invoke.call_args_list[1][0][0]
        self.assertIn("Last action taken: ViewAppointments", prompt_after_1st_call)
        self.assertIn("Number of consecutive actions: 1", prompt_after_1st_call)

        # Second call
        # Need to feed the output state of the first call (which includes agent's response)
        # back into the graph. The 'messages' list in state would grow.
        # For this specific test, we'll focus on the prompt generation part.
        # We need to manually construct the state that call_agent would receive.
        current_state_for_prompt = {
            **self.initial_state,
            "last_action": "ViewAppointments",
            "action_count": 1 # This is what call_agent would receive for the 2nd decision
        }
        # This doesn't test the graph's state flow, but the prompt formatting directly.
        # For simplicity, testing the prompt formatting part:
        agent_instance_for_prompt_test = AppointmentAgent() # Fresh instance for CustomPromptTemplate
        prompt_template_str = agent_instance_for_prompt_test.create_agent.__closure__[1].cell_contents.template # Accessing template string via closure (fragile)

        test_prompt = CustomPromptTemplate(
            template=prompt_template_str,
            tools=[],
            input_variables=["last_action", "action_count", "name", "date", "time", "purpose", "input", "history", "agent_scratchpad", "tools", "tool_names", "has_all_info"]
        )

        formatted_prompt = test_prompt.format(
            input="hello",
            history="",
            agent_scratchpad="",
            name=None,
            date=None,
            time=None,
            purpose=None,
            has_all_info=False,
            tools="",
            tool_names=[],
            last_action="ViewAppointments",
            action_count=1
        )
        self.assertIn("Last action taken: ViewAppointments", formatted_prompt)
        self.assertIn("Number of consecutive actions: 1", formatted_prompt)

        formatted_prompt_2 = test_prompt.format(
            input="hello",
            history="",
            agent_scratchpad="",
            name=None,
            date=None,
            time=None,
            purpose=None,
            has_all_info=False,
            tools="",
            tool_names=[],
            last_action="ViewAppointments",
            action_count=2 # If it were the 3rd decision to use ViewAppointments
        )
        self.assertIn("Last action taken: ViewAppointments", formatted_prompt_2)
        self.assertIn("Number of consecutive actions: 2", formatted_prompt_2)


    def test_tool_not_found(self):
        self.mock_llm_instance.invoke.return_value = MagicMock(
            content="Thought: I need to use a tool that doesn't exist.\nAction: NonExistentTool\nAction Input: None"
        )
        state = {**self.initial_state, "messages": [{"role": "user", "content": "Use NonExistentTool"}]}
        final_state = self.graph.invoke(state)

        # The error message "Error: Tool 'NonExistentTool' not found." would be part of the agent's scratchpad,
        # and then the LLM would be invoked again. The final answer might reflect this.
        # We check the 'messages' in the final state for an assistant message containing the error.
        assistant_messages = [m["content"] for m in final_state["messages"] if m["role"] == "assistant"]
        self.assertTrue(any("Error executing NonExistentTool" in msg or "Error: Tool 'NonExistentTool' not found." in msg for msg in assistant_messages))

    def test_handle_misidentified_name_and_confirmation(self):
        # --- Setup: Agent has previously misidentified name and asked for confirmation + details ---
        # This simulates the conversation history leading up to the user saying "yeah sure".
        initial_user_input_that_led_to_bad_name = "i am new here please help to process"
        agent_previous_question_with_bad_name = "Okay, I understand you're new and need a consultation. First, is the name 'i am new here please help to process' correct? Also, please provide the date (YYYY-MM-DD) and time (HH:MM)?"

        # Mock the LLM's response for when the input is "yeah sure" under these conditions.
        # The agent's internal Python logic should have already cleared the suspect name
        # and set the state to CS_COLLECTING_BOOKING_INFO before this LLM call.
        # So, the LLM's task is to formulate a response based on a state where 'name' is None
        # and the goal is to collect it.
        self.mock_llm_instance.invoke.return_value = MagicMock(
            content="Thought: The user affirmed, but the previously stored name was suspect and has been cleared by prior logic. The current state is CS_COLLECTING_BOOKING_INFO and name is missing. I must ask for the name now.\nFinal Answer: Okay, great! To get started, could you please tell me your name?"
        )

        # --- User says "yeah sure" ---
        user_confirmation_input = "yeah sure"

        # Define the state of the conversation just before processing "yeah sure".
        # Crucially, `booking_info["name"]` contains the suspect phrase.
        state_before_confirmation_processed = {
            "messages": [
                {"role": "user", "content": initial_user_input_that_led_to_bad_name},
                {"role": "assistant", "content": agent_previous_question_with_bad_name},
                {"role": "user", "content": user_confirmation_input}
            ],
            "booking_info": {"name": "i am new here please help to process", "date": None, "time": None, "purpose": "consultation"}, # Bad name is present here
            "last_action": None,
            "action_count": 0,
            # When "yeah sure" is processed, the agent's `call_agent` function should:
            # 1. Detect `user_affirmed` is true.
            # 2. Check `made_booking_offer` (agent_previous_question_with_bad_name implies an offer to book).
            # 3. Identify `booking_info.name` as suspect and clear it to None.
            # 4. Set `current_conversation_state_for_prompt` to CS_COLLECTING_BOOKING_INFO.
            # 5. Call the LLM with this updated state (name=None, state=CS_COLLECTING_BOOKING_INFO).
            "conversation_state": CS_GENERAL_INQUIRY # Initial state before "yeah sure" is processed by call_agent
        }

        # Invoke the graph. The `call_agent` method will execute its Python logic,
        # then call the mocked LLM.
        final_agent_output_obj = self.graph.invoke(state_before_confirmation_processed)

        # The agent's response content is taken from the mocked LLM's `content` field.
        final_response_content = final_agent_output_obj['current_step']

        # Assertions:
        # 1. The mocked LLM was called exactly once during this invocation.
        self.mock_llm_instance.invoke.assert_called_once()

        # 2. The agent's final response (from the LLM) should be asking for the name.
        self.assertIn("could you please tell me your name?", final_response_content)
        # Ensure the bad name is not part of the agent's response.
        self.assertNotIn("i am new here please help to process", final_response_content)

        # 3. The `booking_info.name` in the final state returned by the graph should be None,
        #    indicating it was cleared by the agent's internal logic.
        self.assertIsNone(final_agent_output_obj['booking_info']['name'],
                          "Booking_info.name should have been cleared by the agent's logic before calling the LLM.")

    def test_transition_to_collecting_info_after_name_provided(self):
        # Simulate conversation history:
        # 1. User: asks to explain booking
        # 2. Agent: explains booking
        # 3. User: provides their name ("i am Test User")

        history_messages = [
            {"role": "user", "content": "Can you explain how booking works?"},
            {"role": "assistant", "content": "Final Answer: To book, I need name, date, time, purpose. For example..."} # Simplified agent explanation
        ]

        user_provides_name_input = "i am Test User" # 4 words, fits extraction rules

        # Expected LLM behavior after name is provided:
        # The LLM should be guided by the modified CS_GENERAL_INQUIRY prompt (for handling newly provided name)
        # to acknowledge the name, ask for the next detail, and signal transition to CS_COLLECTING_BOOKING_INFO.
        self.mock_llm_instance.invoke.return_value = MagicMock(
            content="Thought: User provided name 'i am Test User'. Acknowledge and ask for date. Set next state to CS_COLLECTING_BOOKING_INFO.\nFinal Answer: Thanks, i am Test User! What date would you like for your appointment (YYYY-MM-DD)?"
        )

        # State before processing the user's name input
        state_before_name_provided = {
            "messages": history_messages + [{"role": "user", "content": user_provides_name_input}],
            "booking_info": {"name": None, "date": None, "time": None, "purpose": None}, # Name is None initially
            "last_action": None,
            "action_count": 0,
            "conversation_state": CS_GENERAL_INQUIRY # Agent is in general inquiry before processing this new input
        }

        # Invoke the graph
        final_agent_output_obj = self.graph.invoke(state_before_name_provided)
        final_response_content = final_agent_output_obj['current_step']

        # Assertions:
        # 1. LLM was called once for this interaction.
        self.mock_llm_instance.invoke.assert_called_once()

        # 2. The agent's response should acknowledge the name and ask for the next detail.
        self.assertIn("Thanks, i am Test User!", final_response_content)
        self.assertIn("What date would you like", final_response_content)

        # 3. The booking_info.name in the final state should be the extracted name.
        #    The name extraction logic: `state["booking_info"]["name"] = last_message` if rules pass.
        self.assertEqual(final_agent_output_obj['booking_info']['name'], "i am Test User")

        # 4. The conversation state in the *final_agent_output_obj* should be CS_COLLECTING_BOOKING_INFO,
        #    as signaled by the LLM and processed by the updated call_agent logic.
        self.assertEqual(final_agent_output_obj['conversation_state'], CS_COLLECTING_BOOKING_INFO)

        # 5. (Optional but good) Check the arguments passed to the LLM prompt formatting.
        #    The `name` should be "i am Test User" and `current_conversation_state` should be CS_GENERAL_INQUIRY
        #    when the LLM was called for *this specific turn*.
        #    The `call_agent` function uses `current_conversation_state_for_prompt` for the LLM call.
        #    This state is CS_GENERAL_INQUIRY. After the LLM responds with the signal to change state,
        #    `state['conversation_state']` becomes CS_COLLECTING_BOOKING_INFO. So Assertion 4 is key.

        # llm_call_kwargs = self.mock_llm_instance.invoke.call_args.kwargs
        # This part is more for deeper debugging if needed and can be omitted for brevity
        # if the primary assertions (1-4) cover the main functionality.

    def test_general_help_for_clueless_user(self):
        user_input = "i am new i dont know anything can you please help me"

        # Expected LLM response based on the refined CS_GENERAL_INQUIRY prompt:
        # The agent should provide an overview of its capabilities.
        expected_llm_thought = "Thought: The user is new and states they don't know anything. I should provide a concise overview of my main functions as per CS_GENERAL_INQUIRY guidance, then ask how they'd like to proceed."
        expected_final_answer = "I'm an appointment booking assistant. I can help you schedule new appointments, check your existing ones, or show you examples of typical appointment reasons. To get started, you can tell me what you'd like to do, like saying 'book an appointment' or 'view my appointments'. What would you like to do, or do you need more details on any of these functions?"

        self.mock_llm_instance.invoke.return_value = MagicMock(
            content=f"{expected_llm_thought}\nFinal Answer: {expected_final_answer}"
        )

        # Initial state for this user input
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "booking_info": {"name": None, "date": None, "time": None, "purpose": None},
            "last_action": None,
            "action_count": 0,
            # The user's input "i am new..." should transition from INITIAL_GREETING to GENERAL_INQUIRY
            # in the call_agent logic before the LLM is invoked.
            "conversation_state": CS_INITIAL_GREETING
        }

        # Invoke the graph
        final_agent_output_obj = self.graph.invoke(initial_state)
        final_response_content = final_agent_output_obj['current_step'] # This is the raw LLM output

        # Assertions:
        # 1. LLM was called once.
        self.mock_llm_instance.invoke.assert_called_once()

        # 2. The agent's response (which is the LLM's Final Answer) should contain the overview.
        self.assertIn("I'm an appointment booking assistant.", final_response_content)
        self.assertIn("schedule new appointments, check your existing ones, or show you examples", final_response_content)
        self.assertIn("To get started, you can tell me what you'd like to do", final_response_content)

        # 3. Verify that the conversation_state for the LLM prompt was CS_GENERAL_INQUIRY.
        #    The call_agent logic should transition from CS_INITIAL_GREETING to CS_GENERAL_INQUIRY
        #    for this type of input.
        #    The state recorded in final_agent_output_obj is the state *after* the LLM call and any
        #    LLM-signaled state changes.
        #    Our mocked LLM doesn't signal a state change here.
        #    The call_agent logic transitions CS_INITIAL_GREETING -> CS_GENERAL_INQUIRY for such inputs.
        #    Then, when the LLM is called with CS_GENERAL_INQUIRY, and if it doesn't signal a state change,
        #    the state in final_agent_output_obj['conversation_state'] should be CS_GENERAL_INQUIRY.
        self.assertEqual(final_agent_output_obj['conversation_state'], CS_GENERAL_INQUIRY,
                         "The conversation state after the LLM response should be CS_GENERAL_INQUIRY as no further transition was signaled.")

    def test_extract_time_am_pm_정확히_on_the_hour(self): # Using Korean for unique name
        mock_now = dt(2024, 7, 15, 10, 31) # Current time has non-zero minutes
        user_input = "How about 11am?"
        # Expected: time should be 11:00 due to heuristic, date should be None (if not intended)
        expected_changes = {"time": "11:00"}
        final_info = self._run_datetime_extraction_test(user_input, mock_now, expected_changes)
        self.assertIsNone(final_info.get("date"), "Date should not be set if only '11am' is mentioned.")

    def test_purpose_update_from_generic_to_specific(self):
        initial_generic_purpose = "user needs help"
        user_clarifies_symptoms = "actually i have a bad headache"
        agent_suggests_specific = "Perhaps a 'Consultation' for your headache?" # Part of LLM1 output
        user_confirms_specific = "yes, consultation sounds right"

        # LLM Responses:
        # 1. After user clarifies symptoms (LLM suggests specific purpose)
        llm_response_1_content = f"Thought: User has a headache. Suggest 'Consultation'. Effective current purpose: Consultation.\nFinal Answer: {agent_suggests_specific}"
        # 2. After user confirms specific purpose (LLM acknowledges and asks next q, e.g., name)
        llm_response_2_content = "Thought: User confirmed 'Consultation'. Name is missing. Set next state to CS_COLLECTING_BOOKING_INFO. Effective current purpose: Consultation.\nFinal Answer: Okay, a 'Consultation' it is. What's your name?"

        self.mock_llm_instance.invoke.side_effect = [
            MagicMock(content=llm_response_1_content),
            MagicMock(content=llm_response_2_content)
        ]

        # Initial state: generic purpose already set somehow
        state_v1 = {
            "messages": [{"role": "user", "content": user_clarifies_symptoms}],
            "booking_info": {"name": None, "date": None, "time": None, "purpose": initial_generic_purpose},
            "conversation_state": CS_GENERAL_INQUIRY, # Assuming this state allows purpose refinement via LLM
            "last_action": None, "action_count": 0,
        }
        intermediate_state = self.graph.invoke(state_v1)
        # Assert that after LLM1, purpose in booking_info is "Consultation"
        self.assertEqual(intermediate_state['booking_info']['purpose'], "Consultation", "Purpose should be updated by LLM signal 1")

        # Ensure the LLM mock is reset for the next call if not using side_effect for all calls in a test
        # self.mock_llm_instance.reset_mock() # Not needed here due to fresh side_effect

        state_v2 = {
            "messages": [
                {"role": "user", "content": user_clarifies_symptoms},
                {"role": "assistant", "content": agent_suggests_specific}, # from llm_response_1
                {"role": "user", "content": user_confirms_specific}
            ],
            "booking_info": intermediate_state['booking_info'], # Carry over booking_info
            "conversation_state": CS_GENERAL_INQUIRY, # Or whatever state LLM1 led to, assume general for now
            "last_action": None, "action_count": 0,
        }
        final_state = self.graph.invoke(state_v2)
        # Assert that after LLM2, purpose is still "Consultation" and agent asks for name
        self.assertEqual(final_state['booking_info']['purpose'], "Consultation", "Purpose should remain updated")
        self.assertIn("What's your name?", final_state['current_step'])
        self.assertEqual(final_state['conversation_state'], CS_COLLECTING_BOOKING_INFO)


    def test_confirmation_flow_user_confirms_yes(self):
        # State: All info collected, agent is about to ask for confirmation
        all_info = {"name": "Test User", "date": "2024-09-15", "time": "14:00", "purpose": "Checkup"}
        state_before_confirm_prompt = {
            "messages": [{"role": "user", "content": "some prior message that led to all info"}],
            "booking_info": all_info,
            # "has_all_info": True, # Agent recalculates this
            "last_action": None, "action_count": 0,
            "conversation_state": CS_COLLECTING_BOOKING_INFO
        }

        # LLM Responses:
        # 1. Agent asks for confirmation (in CS_CONFIRMING_BOOKING_INFO)
        confirm_prompt_text = f"So, I have an appointment for {all_info['name']} on {all_info['date']} at {all_info['time']} for {all_info['purpose']}. Is that all correct?"
        llm_asks_confirmation = MagicMock(content=f"Thought: All info present. Ask for confirmation. Set next state to CS_AWAITING_FINAL_CONFIRMATION.\nFinal Answer: {confirm_prompt_text}")

        # 2. User says "yes", agent calls BookAppointment (in CS_AWAITING_FINAL_CONFIRMATION)
        self.mock_book_appointment_tool.return_value = "Appointment booked successfully!"
        llm_calls_bookappointment = MagicMock(content=f"Thought: User confirmed. Book it. Effective current purpose: {all_info['purpose']}.\nAction: BookAppointment\nAction Input: {json.dumps(all_info)}")

        self.mock_llm_instance.invoke.side_effect = [llm_asks_confirmation, llm_calls_bookappointment]

        # First invocation: Agent asks for confirmation
        state_after_confirm_prompt = self.graph.invoke(state_before_confirm_prompt)
        self.assertIn(confirm_prompt_text, state_after_confirm_prompt['current_step'])
        self.assertEqual(state_after_confirm_prompt['conversation_state'], CS_AWAITING_FINAL_CONFIRMATION)

        # Second invocation: User confirms "yes"
        state_after_user_yes = {
            "messages": state_after_confirm_prompt['messages'] + [{"role": "user", "content": "yes, that's correct"}],
            "booking_info": all_info,
            "last_action": None, "action_count": 0, # Reset for this new user turn
            "conversation_state": CS_AWAITING_FINAL_CONFIRMATION
        }
        final_state = self.graph.invoke(state_after_user_yes)
        self.mock_book_appointment_tool.assert_called_once_with(all_info)

        # Check messages for tool output being included before next LLM call (if any)
        # The current graph structure: agent -> tool -> agent. So after BookAppointment, LLM is called again.
        # The "Appointment booked successfully!" would be an observation.
        # The final_state['current_step'] would be the LLM's response *after* the booking.
        # We need to check that the booking tool's result is in the message history.
        assistant_messages_after_booking = [m['content'] for m in final_state['messages'] if m['role'] == 'assistant']
        self.assertTrue(any("Observation: Appointment booked successfully!" in msg for msg in assistant_messages_after_booking),
                        "Tool observation for successful booking not found in assistant messages.")


    def test_confirmation_flow_user_says_no(self):
        all_info = {"name": "Test User", "date": "2024-09-15", "time": "14:00", "purpose": "Checkup"}
        state_before_confirm_prompt = {
            "messages": [{"role": "user", "content": "some prior message"}],
            "booking_info": all_info,
            "last_action": None, "action_count": 0,
            "conversation_state": CS_COLLECTING_BOOKING_INFO
        }

        # LLM Responses:
        # 1. Agent asks for confirmation
        confirm_prompt_text = f"So, I have an appointment for {all_info['name']} on {all_info['date']} at {all_info['time']} for {all_info['purpose']}. Is that all correct?"
        llm_asks_confirmation = MagicMock(content=f"Thought: Ask for confirmation. Set next state to CS_AWAITING_FINAL_CONFIRMATION.\nFinal Answer: {confirm_prompt_text}")

        # 2. User says "no", agent asks what to change (in CS_AWAITING_FINAL_CONFIRMATION)
        llm_asks_what_to_change = MagicMock(content="Thought: User wants to change. Ask what. Set next state to CS_COLLECTING_BOOKING_INFO.\nFinal Answer: Okay, what information isn't correct or what would you like to change?")

        self.mock_llm_instance.invoke.side_effect = [llm_asks_confirmation, llm_asks_what_to_change]

        # First invocation: Agent asks for confirmation
        state_after_confirm_prompt = self.graph.invoke(state_before_confirm_prompt)
        self.assertEqual(state_after_confirm_prompt['conversation_state'], CS_AWAITING_FINAL_CONFIRMATION)

        # Second invocation: User says "no"
        state_after_user_no = {
            "messages": state_after_confirm_prompt['messages'] + [{"role": "user", "content": "no, the time is wrong"}],
            "booking_info": all_info,
            "last_action": None, "action_count": 0,
            "conversation_state": CS_AWAITING_FINAL_CONFIRMATION
        }
        final_state = self.graph.invoke(state_after_user_no)

        self.assertIn("what information isn't correct", final_state['current_step'])
        self.assertEqual(final_state['conversation_state'], CS_COLLECTING_BOOKING_INFO)
        self.mock_book_appointment_tool.assert_not_called()

    # Helper method to set up state and mock LLM for date/time tests
    def _run_datetime_extraction_test(self, user_input, mock_current_datetime, expected_booking_info_changes):
        # Mock datetime.now() within the agent's module
        # The agent uses `from datetime import datetime` and calls `datetime.now()`
        with patch('appointment_system.agent.datetime', autospec=True) as mock_datetime_class:
            mock_datetime_class.now.return_value = mock_current_datetime

            # Mock LLM to provide a simple final answer after extraction
            self.mock_llm_instance.reset_mock() # Reset mock before each call in helper
            self.mock_llm_instance.invoke.return_value = MagicMock(
                content="Thought: Extracted info, will confirm or ask for more.\nFinal Answer: Okay, processing that."
            )

            initial_state = {
                "messages": [{"role": "user", "content": user_input}],
                "booking_info": {"name": "TestUser", "date": None, "time": None, "purpose": "Checkup"}, # Assume name and purpose are somehow known
                "last_action": None,
                "action_count": 0,
                "conversation_state": CS_COLLECTING_BOOKING_INFO # Test extraction in this state
            }

            final_agent_output_obj = self.graph.invoke(initial_state)
            final_booking_info = final_agent_output_obj['booking_info']

            for key, expected_value in expected_booking_info_changes.items():
                self.assertEqual(final_booking_info.get(key), expected_value, f"Booking info for '{key}' not as expected.")

            return final_booking_info

    def test_extract_full_date_time_natural_language(self):
        mock_now = dt(2024, 7, 15, 10, 0) # July 15, 2024, 10:00 AM
        user_input = "How about August 20 at 11am?"
        expected_changes = {"date": "2024-08-20", "time": "11:00"}
        self._run_datetime_extraction_test(user_input, mock_now, expected_changes)

    def test_extract_tomorrow_and_time(self):
        mock_now = dt(2024, 7, 15, 10, 0) # July 15, 2024
        user_input = "tomorrow 3pm please"
        expected_date = (mock_now + timedelta(days=1)).strftime("%Y-%m-%d")
        expected_changes = {"date": expected_date, "time": "15:00"}
        self._run_datetime_extraction_test(user_input, mock_now, expected_changes)

    def test_extract_month_day_defaults_current_year(self):
        mock_now = dt(2024, 7, 15, 10, 0) # July 15, 2024
        user_input = "Is September 10th okay?"
        expected_changes = {"date": "2024-09-10"}
        final_info = self._run_datetime_extraction_test(user_input, mock_now, expected_changes)
        self.assertIsNone(final_info.get("time"), "Time should not be set if only date is mentioned.")


    def test_extract_time_defaults_current_date(self):
        mock_now = dt(2024, 7, 15, 10, 0) # July 15, 2024
        user_input = "Can we do 2:30 PM?"
        expected_changes = {"time": "14:30"}
        final_info = self._run_datetime_extraction_test(user_input, mock_now, expected_changes)
        self.assertIsNone(final_info.get("date"), "Date should not be set if only time is mentioned and date was initially None.")


    def test_extract_month_day_past_date_uses_next_year(self):
        mock_now = dt(2024, 12, 15, 10, 0) # December 15, 2024
        user_input = "How about January 10?" # Should be for 2025
        expected_changes = {"date": "2025-01-10"}
        final_info = self._run_datetime_extraction_test(user_input, mock_now, expected_changes)
        self.assertIsNone(final_info.get("time"), "Time should not be set if only date is mentioned for next year.")

    def test_extract_date_with_fuzzy_input(self):
        mock_now = dt(2024, 7, 15, 10, 0)
        user_input = "I'd like to book for July 25th this year, maybe around 4pm?"
        expected_changes = {"date": "2024-07-25", "time": "16:00"}
        self._run_datetime_extraction_test(user_input, mock_now, expected_changes)

    def test_input_without_date_time_does_not_crash(self):
        user_input = "i am new here and need help with the process" # Input that contains no date/time
        # This input would cause UnboundLocalError if heuristic flags are not pre-initialized

        self.mock_llm_instance.invoke.return_value = MagicMock(
            content="Thought: User is new and needs help. Provide overview.\nFinal Answer: I can help with booking, viewing, or examples. What would you like?"
        )

        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "booking_info": {"name": None, "date": None, "time": None, "purpose": None},
            "conversation_state": CS_GENERAL_INQUIRY, # This state is where the flags are used
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        }

        try:
            final_state = self.graph.invoke(initial_state)
            # We mainly care that it doesn't crash. Content of final_state['current_step'] can be checked too.
            self.assertIn("What would you like?", final_state['current_step'])
        except UnboundLocalError:
            self.fail("UnboundLocalError was raised, meaning date/time heuristic flags were not properly initialized.")

    def test_extract_relative_date_tomorrow(self):
        mock_now = dt(2024, 7, 15, 10, 0) # July 15, 2024
        user_input = "ok then tommorow" # Note: "tommorow" spelling as per user log
        expected_date = (mock_now + timedelta(days=1)).strftime("%Y-%m-%d")
        expected_changes = {"date": expected_date}
        # This test also implicitly checks if "ok then" is handled by fuzzy parsing or pre-processing
        final_info = self._run_datetime_extraction_test(user_input, mock_now, expected_changes)
        self.assertIsNone(final_info.get("time"), "Time should not be set for 'ok then tommorow'")

    def test_extract_relative_date_day_after_tomorrow(self):
        mock_now = dt(2024, 7, 15, 10, 0)
        user_input = "i just need day after tommorow" # Note: "tommorow" spelling
        # Pre-processing should change "day after tommorow" to "in 2 days"
        expected_date = (mock_now + timedelta(days=2)).strftime("%Y-%m-%d")
        expected_changes = {"date": expected_date}
        final_info = self._run_datetime_extraction_test(user_input, mock_now, expected_changes)
        self.assertIsNone(final_info.get("time"), "Time should not be set for 'day after tommorow'")

    def test_confirmation_message_uses_correct_specific_details(self):
        # Setup: name is "Suhail Shaji", purpose is "Consultation for back pain"
        # Date and time are also known.
        specific_booking_info = {
            "name": "Suhail Shaji",
            "date": "2024-09-20",
            "time": "11:00",
            "purpose": "Consultation for back pain"
        }

        # State where agent is about to generate the confirmation prompt
        state_entering_confirmation = {
            "messages": [
                {"role": "user", "content": "some message that led to all info being collected"},
            ],
            "booking_info": specific_booking_info,
            "conversation_state": CS_COLLECTING_BOOKING_INFO, # This, with all info, will trigger CS_CONFIRMING_BOOKING_INFO for LLM prompt
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        }

        # Mock LLM response when CS_CONFIRMING_BOOKING_INFO is the prompt state
        # The LLM should use the {name}, {date}, {time}, {purpose} from its context.
        expected_confirmation_question = "So, I have an appointment for Suhail Shaji on 2024-09-20 at 11:00 for Consultation for back pain. Is that all correct?"
        self.mock_llm_instance.invoke.return_value = MagicMock(
            content=f"Thought: All details present. Confirm them. Effective current purpose: Consultation for back pain. Set next state to CS_AWAITING_FINAL_CONFIRMATION.\nFinal Answer: {expected_confirmation_question}"
        )

        final_state = self.graph.invoke(state_entering_confirmation)

        # Check that the LLM's output (the confirmation question) contains the correct specific details
        self.assertIn("Suhail Shaji", final_state['current_step'])
        self.assertIn("2024-09-20", final_state['current_step'])
        self.assertIn("11:00", final_state['current_step'])
        self.assertIn("Consultation for back pain", final_state['current_step'])
        self.assertNotIn("book an appointment", final_state['current_step'], "Generic purpose should not be in confirmation")

        self.assertEqual(final_state['conversation_state'], CS_AWAITING_FINAL_CONFIRMATION)

    def test_date_only_input_does_not_set_random_time(self):
        mock_now = dt(2024, 8, 1, 14, 59) # Current time is 14:59
        user_input = "august 2 this year"
        expected_date = "2024-08-02" # Assuming mock_now is in 2024

        # We expect date to be set, but time to remain None because no time was mentioned.
        expected_changes = {"date": expected_date}
        final_info = self._run_datetime_extraction_test(user_input, mock_now, expected_changes)

        self.assertIsNone(final_info.get("time"),
                          f"Time should be None for date-only input, but was {final_info.get('time')}")

    def test_affirm_booking_offer_asks_next_detail_directly(self):
        # Scenario:
        # 1. Agent has some context (e.g., user mentioned back pain, purpose is 'Consultation for back pain').
        # 2. Agent makes a booking offer: "Okay, [User Name], for the 'Consultation for back pain', shall I proceed to get date and time?"
        # 3. User says: "yes"
        # 4. Agent should respond: "Great, [User Name]! For your 'Consultation for back pain', what date would you like (YYYY-MM-DD)?" (assuming name is known, date is next)

        user_name = "Test User"
        confirmed_purpose = "Consultation for back pain"

        # Setup initial state: name and purpose are known. Agent is about to make an offer.
        # For this test, we'll start from the user's "yes" after an offer was made.
        # The agent's last message (in history) would be the offer.
        agent_offer_message = f"Okay, {user_name}, for the '{confirmed_purpose}', shall I proceed to get date and time?"

        current_messages = [
            {"role": "assistant", "content": "Some prior conversation..."}, # Simulating some history
            {"role": "user", "content": "My name is Test User and I have back pain."},
            {"role": "assistant", "content": agent_offer_message} # Agent's offer
        ]
        current_booking_info = {"name": user_name, "date": None, "time": None, "purpose": confirmed_purpose}

        # LLM mock for the turn AFTER user says "yes".
        # This LLM call will be made with current_conversation_state_for_prompt = CS_COLLECTING_BOOKING_INFO
        # due to the made_booking_offer & user_affirmed logic in call_agent.
        # The refined CS_COLLECTING_BOOKING_INFO prompt should then guide this response.
        expected_agent_response_content = f"Thought: User affirmed booking offer for '{confirmed_purpose}' for '{user_name}'. Date is missing. Ask for date. Effective current purpose: {confirmed_purpose}.\nFinal Answer: Great, {user_name}! For your '{confirmed_purpose}', what date would you like (YYYY-MM-DD)?"
        self.mock_llm_instance.invoke.return_value = MagicMock(content=expected_agent_response_content)

        user_input_affirmation = "yes"
        current_messages.append({"role": "user", "content": user_input_affirmation})

        # The state passed to invoke for this turn.
        # call_agent will identify made_booking_offer=True from agent_offer_message and user_affirmed=True.
        # It will then set current_conversation_state_for_prompt to CS_COLLECTING_BOOKING_INFO.
        state_for_affirmation_processing = {
            "messages": current_messages,
            "booking_info": current_booking_info,
            "conversation_state": CS_GENERAL_INQUIRY, # Or whatever state agent was in when it made the offer
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        }

        final_state = self.graph.invoke(state_for_affirmation_processing)

        # 1. Check LLM was called once for this "yes" turn.
        self.mock_llm_instance.invoke.assert_called_once()

        # 2. Agent's response should ask for the date, not repeat the offer.
        self.assertIn(f"Great, {user_name}!", final_state['current_step'])
        self.assertIn(f"For your '{confirmed_purpose}', what date would you like", final_state['current_step'])
        self.assertNotIn("Shall I proceed", final_state['current_step'].lower(), "Agent should not repeat the offer.")

        # 3. Booking info (name, purpose) should be preserved.
        self.assertEqual(final_state['booking_info']['name'], user_name)
        self.assertEqual(final_state['booking_info']['purpose'], confirmed_purpose)

        # 4. Final conversation state should be CS_COLLECTING_BOOKING_INFO (as set by call_agent logic,
        #    and LLM didn't signal a change from it in this mocked response).
        self.assertEqual(final_state['conversation_state'], CS_COLLECTING_BOOKING_INFO)

    def test_correction_during_final_confirmation_updates_and_reconfirms(self):
        # Scenario:
        # 1. Agent has all info (Name A, Date A, Time A_initial, Purpose A) and asks for confirmation.
        #    (Agent is in CS_CONFIRMING_BOOKING_INFO, then transitions to CS_AWAITING_FINAL_CONFIRMATION)
        # 2. User: "no, the time is actually 11am" (provides correction)
        # 3. Agent (LLM mock): Should acknowledge, re-parse. Python logic updates booking_info.time to "11:00".
        #    Agent then transitions back to CS_CONFIRMING_BOOKING_INFO.
        # 4. Agent (LLM mock): Re-confirms ALL details, now with Time A_corrected ("11:00").
        #    Transitions again to CS_AWAITING_FINAL_CONFIRMATION.
        # 5. User: "yes"
        # 6. Agent (LLM mock): Calls BookAppointment with corrected time.

        initial_booking_info = {"name": "TestUser", "date": "2024-09-25", "time": "14:30", "purpose": "Dental Checkup"}
        corrected_time = "11:00" # User wants to change to 11am

        # This will be the booking_info after user correction and Python re-parsing
        updated_booking_info_after_correction = initial_booking_info.copy()
        updated_booking_info_after_correction["time"] = corrected_time

        # --- Mock LLM Responses ---
        # 1. LLM asks for initial confirmation
        llm_response_ask_confirmation_content = (
            f"Thought: All details collected. Confirm them. Effective current purpose: {initial_booking_info['purpose']}. "
            f"Set next state to CS_AWAITING_FINAL_CONFIRMATION.\n"
            f"Final Answer: So, I have an appointment for {initial_booking_info['name']} on {initial_booking_info['date']} "
            f"at {initial_booking_info['time']} for {initial_booking_info['purpose']}. Is that all correct?"
        )

        # 2. LLM acknowledges correction (after user says "no, the time is 11am")
        #    Prompt for CS_AWAITING_FINAL_CONFIRMATION (non-affirmative branch) guides this.
        llm_response_acknowledge_correction_content = (
            "Thought: User input is not 'yes', treating as correction. System will re-parse this input. "
            "I must then re-confirm. Set next state to CS_CONFIRMING_BOOKING_INFO.\n"
            "Final Answer: Okay, let me update that for you."
        )

        # 3. LLM re-confirms with updated details
        #    This LLM call is made when state is CS_CONFIRMING_BOOKING_INFO and booking_info is updated.
        llm_response_reconfirm_details_content = (
            f"Thought: All details (now updated with time {corrected_time}) present. Confirm them. Effective current purpose: {updated_booking_info_after_correction['purpose']}. "
            f"Set next state to CS_AWAITING_FINAL_CONFIRMATION.\n"
            f"Final Answer: So, I have an appointment for {updated_booking_info_after_correction['name']} on {updated_booking_info_after_correction['date']} "
            f"at {updated_booking_info_after_correction['time']} for {updated_booking_info_after_correction['purpose']}. Is that all correct?"
        )

        # 4. LLM books appointment after user confirms updated details
        self.mock_book_appointment_tool.return_value = "Appointment booked successfully!" # Setup mock for tool
        llm_response_book_appointment_content = (
            f"Thought: User confirmed updated details. Book it. Effective current purpose: {updated_booking_info_after_correction['purpose']}.\n"
            f"Action: BookAppointment\nAction Input: {json.dumps(updated_booking_info_after_correction)}"
        )

        self.mock_llm_instance.invoke.side_effect = [
            MagicMock(content=llm_response_ask_confirmation_content),
            MagicMock(content=llm_response_acknowledge_correction_content),
            MagicMock(content=llm_response_reconfirm_details_content),
            MagicMock(content=llm_response_book_appointment_content)
        ]

        # --- Simulate Conversation Flow ---

        # Invoke 1: Agent asks for initial confirmation
        current_messages = [{"role": "user", "content": "Details were just collected"}] # Placeholder history
        current_booking_info = initial_booking_info.copy()
        # Agent's call_agent logic will transition from CS_COLLECTING_BOOKING_INFO to CS_CONFIRMING_BOOKING_INFO for the LLM prompt
        state_after_ask_confirmation = self.graph.invoke({
            "messages": current_messages, "booking_info": current_booking_info,
            "conversation_state": CS_COLLECTING_BOOKING_INFO,
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        })
        self.assertIn(initial_booking_info['time'], state_after_ask_confirmation['current_step'])
        self.assertEqual(state_after_ask_confirmation['conversation_state'], CS_AWAITING_FINAL_CONFIRMATION)
        current_messages = state_after_ask_confirmation['messages']
        current_booking_info = state_after_ask_confirmation['booking_info']

        # Invoke 2: User provides correction "no, the time is actually 11am"
        user_correction_input = "no, the time is actually 11am"
        current_messages.append({"role": "user", "content": user_correction_input})

        state_after_ack_correction = self.graph.invoke({
            "messages": current_messages, "booking_info": current_booking_info,
            "conversation_state": CS_AWAITING_FINAL_CONFIRMATION,
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        })
        self.assertIn("Okay, let me update", state_after_ack_correction['current_step'])
        self.assertEqual(state_after_ack_correction['booking_info']['time'], corrected_time, "Time should be updated in booking_info by Python logic")
        self.assertEqual(state_after_ack_correction['conversation_state'], CS_CONFIRMING_BOOKING_INFO)
        current_messages = state_after_ack_correction['messages']
        current_booking_info = state_after_ack_correction['booking_info']

        # Invoke 3: Agent re-confirms with updated time
        state_after_reconfirm = self.graph.invoke({
            "messages": current_messages, "booking_info": current_booking_info,
            "conversation_state": CS_CONFIRMING_BOOKING_INFO,
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        })
        self.assertIn(f"at {corrected_time} for", state_after_reconfirm['current_step'], "Confirmation should use new time")
        self.assertIn(initial_booking_info['name'], state_after_reconfirm['current_step'])
        self.assertEqual(state_after_reconfirm['conversation_state'], CS_AWAITING_FINAL_CONFIRMATION)
        current_messages = state_after_reconfirm['messages']
        current_booking_info = state_after_reconfirm['booking_info']

        # Invoke 4: User says "yes" to updated details
        user_final_affirmation = "yes, that's perfect now"
        current_messages.append({"role": "user", "content": user_final_affirmation})
        final_state = self.graph.invoke({
            "messages": current_messages, "booking_info": current_booking_info,
            "conversation_state": CS_AWAITING_FINAL_CONFIRMATION,
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        })
        self.mock_book_appointment_tool.assert_called_once_with(updated_booking_info_after_correction)
        self.assertTrue(any("Appointment booked successfully!" in msg['content'] for msg in final_state['messages'] if msg['role'] == 'assistant'))

        self.assertEqual(self.mock_llm_instance.invoke.call_count, 4)

    def test_extract_specific_time_date_combo_from_log(self):
        mock_now = dt(2024, 6, 15, 10, 53) # June 15, 2024
        user_input = "11am july 20"
        expected_changes = {"date": "2024-07-20", "time": "11:00"}
        self._run_datetime_extraction_test(user_input, mock_now, expected_changes)

    def test_name_input_does_not_become_purpose(self):
        user_provides_name = "Lijo Jose"

        expected_llm_response = "Thought: Name 'Lijo Jose' is set. Purpose is not set. Date is missing. Ask for date first. Effective current purpose: None.\nFinal Answer: Thanks, Lijo Jose! What date would you like for your appointment (YYYY-MM-DD)?"
        self.mock_llm_instance.invoke.return_value = MagicMock(content=expected_llm_response)

        initial_state = {
            "messages": [{"role": "user", "content": user_provides_name}],
            "booking_info": {"name": None, "date": None, "time": None, "purpose": None},
            "conversation_state": CS_GENERAL_INQUIRY,
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        }

        final_state = self.graph.invoke(initial_state)

        self.assertEqual(final_state['booking_info']['name'], user_provides_name, "Name was not extracted correctly.")
        self.assertIsNone(final_state['booking_info']['purpose'],
                          f"Purpose should be None, but was '{final_state['booking_info']['purpose']}'. Name should not become purpose.")
        self.assertIn("What date would you like", final_state['current_step'])
        self.assertEqual(final_state['conversation_state'], CS_COLLECTING_BOOKING_INFO)

    def test_confirmation_message_with_none_purpose(self):
        booking_info_no_purpose = {
            "name": "Sachi M",
            "date": "2024-10-05",
            "time": "15:00",
            "purpose": None
        }

        state_entering_confirmation = {
            "messages": [{"role": "user", "content": "Last detail provided"}],
            "booking_info": booking_info_no_purpose,
            "conversation_state": CS_COLLECTING_BOOKING_INFO,
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        }

        expected_confirmation_message = "So, I have an appointment for Sachi M on 2024-10-05 at 15:00 for Purpose: Not Specified. Is that all correct?"
        self.mock_llm_instance.invoke.return_value = MagicMock(
            content=f"Thought: Confirming details. Purpose is None, so state 'Not Specified'. Effective current purpose: None. Set next state to CS_AWAITING_FINAL_CONFIRMATION.\nFinal Answer: {expected_confirmation_message}"
        )

        final_state = self.graph.invoke(state_entering_confirmation)

        self.assertIn("Purpose: Not Specified", final_state['current_step'])
        self.assertNotIn("for None.", final_state['current_step'])
        self.assertEqual(final_state['conversation_state'], CS_AWAITING_FINAL_CONFIRMATION)

    def test_affirm_in_final_confirmation_preserves_purpose(self):
        # Setup: Agent has all correct details and has just asked for final confirmation.
        # The (conceptual) previous turn's LLM (from CS_CONFIRMING_BOOKING_INFO) would have said:
        # "So, I have an appointment for Lijo Jose on 2026-06-11 at 11:00 for Consultation for Fever. Is that all correct?"
        # And its thought would have included "Set next state to CS_AWAITING_FINAL_CONFIRMATION."

        user_name = "Lijo Jose"
        correct_date = "2026-06-11"
        correct_time = "11:00"
        specific_purpose = "Consultation for Fever"

        initial_booking_info_for_this_turn = {
            "name": user_name,
            "date": correct_date,
            "time": correct_time,
            "purpose": specific_purpose # Crucially, this is the correct specific purpose
        }

        # State as user input "yes" is being processed.
        # The agent's prior state (after asking for confirmation) was CS_AWAITING_FINAL_CONFIRMATION.
        state_before_processing_yes = {
            "messages": [
                {"role": "assistant", "content": "So, I have an appointment for Lijo Jose on 2026-06-11 at 11:00 for Consultation for Fever. Is that all correct?"},
                {"role": "user", "content": "yes"}
            ],
            "booking_info": initial_booking_info_for_this_turn,
            "conversation_state": CS_AWAITING_FINAL_CONFIRMATION, # This is key
            "last_action": None, "action_count": 0, "current_step": "", "next": "agent"
        }

        # LLM mock for this turn (when in CS_AWAITING_FINAL_CONFIRMATION and user says "yes")
        # It should decide to call BookAppointment with the existing, correct details.
        self.mock_book_appointment_tool.return_value = "Appointment booked successfully!"
        expected_llm_action_input = {
            "name": user_name, "date": correct_date, "time": correct_time, "purpose": specific_purpose
        }
        llm_calls_bookappointment_content = (
            f"Thought: User confirmed all details ({user_name}, {correct_date}, {correct_time}, {specific_purpose}). I will now book the appointment.\n"
            f"Effective current purpose: {specific_purpose}\n" # LLM should confirm the correct purpose
            f"Action: BookAppointment\nAction Input: {json.dumps(expected_llm_action_input)}"
        )
        self.mock_llm_instance.invoke.return_value = MagicMock(content=llm_calls_bookappointment_content)

        # Invoke the graph
        final_state = self.graph.invoke(state_before_processing_yes)

        # 1. Verify LLM was called once for this "yes" turn.
        self.mock_llm_instance.invoke.assert_called_once()

        # 2. Verify booking_info.purpose in the *final state* is still the specific_purpose.
        #    This confirms the Python logic didn't change it to "yes".
        self.assertEqual(final_state['booking_info']['purpose'], specific_purpose,
                         f"Purpose in booking_info should remain '{specific_purpose}', but was '{final_state['booking_info']['purpose']}'.")

        # 3. Verify BookAppointment tool was called with the correct specific_purpose.
        self.mock_book_appointment_tool.assert_called_once_with(expected_llm_action_input)

        # 4. Check for success message in agent's response history
        self.assertTrue(any("Appointment booked successfully!" in msg['content'] for msg in final_state['messages'] if msg['role'] == 'assistant'))

if __name__ == '__main__':
    # This allows running the tests directly if the subtask environment supports it
    # Ensure Python can find the appointment_system package.
    # Adding parent directory to sys.path might be needed if tests are in a subdir.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    unittest.main(verbosity=2)
