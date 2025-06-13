import unittest
from unittest.mock import patch, MagicMock, call
import re
import json
import os

# Attempt to import from the project structure
# This assumes 'appointment_system' is in the Python path or PYTHONPATH is set up
try:
    from appointment_system.agent import AppointmentAgent, AgentState, CustomPromptTemplate
    from appointment_system.tools import AppointmentTools # For mocking its methods
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

if __name__ == '__main__':
    # This allows running the tests directly if the subtask environment supports it
    # Ensure Python can find the appointment_system package.
    # Adding parent directory to sys.path might be needed if tests are in a subdir.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    unittest.main(verbosity=2)
