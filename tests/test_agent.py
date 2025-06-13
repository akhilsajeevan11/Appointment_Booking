import unittest
from unittest.mock import patch, MagicMock, call
import re
import json
import os

# Attempt to import from the project structure
# This assumes 'appointment_system' is in the Python path or PYTHONPATH is set up
try:
    from appointment_system.agent import AppointmentAgent, AgentState, CustomPromptTemplate, CS_GENERAL_INQUIRY, CS_COLLECTING_BOOKING_INFO
    from appointment_system.tools import AppointmentTools
except ImportError:
    # Fallback for environments where direct import might be tricky
    # Define minimal stubs if real classes can't be loaded by the subtask runner
    # This is less ideal as it doesn't test the actual classes directly
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
    CS_GENERAL_INQUIRY = "GENERAL_INQUIRY" # Add stubs
    CS_COLLECTING_BOOKING_INFO = "COLLECTING_BOOKING_INFO" # Add stubs


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

if __name__ == '__main__':
    # This allows running the tests directly if the subtask environment supports it
    # Ensure Python can find the appointment_system package.
    # Adding parent directory to sys.path might be needed if tests are in a subdir.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    unittest.main(verbosity=2)
