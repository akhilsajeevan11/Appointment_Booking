import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import re
import json

# Attempt to import from the actual application path
# If this fails in the execution environment, the test structure might need adjustment
# or specific mocks for these classes if they can't be imported directly.
try:
    from appointment_system.agent import AppointmentAgent, AgentState, CustomPromptTemplate
    from appointment_system.tools import AppointmentTools # Needed for mocking get_tools
except ImportError:
    # Define simplified stubs if direct import fails, as suggested in the prompt
    # This helps in defining the test structure even if the environment has path issues.
    print("Failed to import from appointment_system.agent, using stubs for test structure definition.")
    class AgentState(TypedDict):
        messages: list
        next: str
        current_step: str
        booking_info: dict
        last_action: str
        action_count: int

    class CustomPromptTemplate:
        def __init__(self, template, tools, input_variables):
            self.template = template
            self.tools = tools
            self.input_variables = input_variables
        def format(self, **kwargs):
            # Simplified format for testing - real formatting is complex
            # For actual test, we'd want to see the real template content
            formatted_str = self.template
            for key, value in kwargs.items():
                formatted_str = formatted_str.replace(f"{{{key}}}", str(value))
            return formatted_str

    class AppointmentAgent:
        def __init__(self):
            self.llm = MagicMock()
            self.tools_handler = MagicMock()
            self.tool_map = {} # Will be populated in create_agent
            # Mock booking_info as it's accessed in the agent
            self.booking_info = {
                "name": None, "date": None, "time": None, "purpose": None
            }


        def create_agent(self):
            # This would normally build the graph. For testing, we might need to
            # mock parts of this or test nested functions more directly if possible.
            # The actual create_agent involves LangGraph setup.
            # For many tests, we'll mock what create_agent produces or call it and mock its deps.

            # Simplified tool_map creation for standalone call_tool tests if needed
            # In full integration tests, this map comes from mocked get_tools()
            mock_tool_list = getattr(self.tools_handler, 'get_tools', MagicMock(return_value=[]))()
            self.tool_map = {tool.name: tool for tool in mock_tool_list}

            # The real create_agent returns a compiled LangGraph app
            # We will mock the app.invoke part for end-to-end style tests of graph nodes
            graph_mock = MagicMock()
            # Make call_agent and call_tool available for more direct patching/testing if needed
            # graph_mock.call_agent = self._call_agent_logic_for_test
            # graph_mock.call_tool = self._call_tool_logic_for_test
            return graph_mock

    class AppointmentTools:
        def get_tools(self):
            return []


# Global test tool instances for mocking
mock_tool_book = MagicMock()
mock_tool_book.name = "BookAppointment"
mock_tool_view = MagicMock()
mock_tool_view.name = "ViewAppointments"


class TestAgent(unittest.TestCase):

    def setUp(self):
        # Mock environment variables if agent uses os.getenv directly on import or init
        # self.env_patch = patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'})
        # self.env_patch.start()
        # logger_patch = patch('appointment_system.agent.logger', MagicMock())
        # logger_patch.start()

        # Instantiate the agent. This might require mocking ChatGoogleGenerativeAI if it's instantiated in __init__
        with patch('appointment_system.agent.ChatGoogleGenerativeAI') as MockLLM, \
             patch('appointment_system.agent.AppointmentTools') as MockApptTools:

            # Configure the mock tools_handler instance that self.agent will use
            self.mock_tools_handler_instance = MockApptTools.return_value
            self.mock_tools_handler_instance.get_tools.return_value = [
                {"name": "BookAppointment", "func": mock_tool_book.func, "description": "Books an appt"},
                {"name": "ViewAppointments", "func": mock_tool_view.func, "description": "Views appts"}
            ]

            self.agent = AppointmentAgent()
            # Ensure llm is a mock after agent instantiation
            self.agent.llm = MockLLM.return_value

            # The graph is created by calling create_agent()
            # This will use the mocked tools_handler via self.agent.tools_handler
            self.graph = self.agent.create_agent()


    def tearDown(self):
        # self.env_patch.stop()
        # patch.stopall() # Stops all patches started with start()
        pass

    def test_datetime_regex_extraction_in_call_agent(self):
        # This tests the regex directly as used in call_agent
        # The call_agent function itself is part of the graph, so we test its components
        date_regex = r'\d{4}-\d{2}-\d{2}' # As defined in agent.py
        time_regex = r'\d{2}:\d{2}'       # As defined in agent.py

        self.assertIsNotNone(re.search(date_regex, "Please book for 2024-07-25"))
        self.assertIsNone(re.search(date_regex, "Please book for 25/07/2024"))
        self.assertEqual(re.search(date_regex, "Date is 2024-03-10.").group(0), "2024-03-10")

        self.assertIsNotNone(re.search(time_regex, "at 14:30 thanks"))
        self.assertIsNone(re.search(time_regex, "at 2:30pm")) # Legacy format
        self.assertIsNone(re.search(time_regex, "at 2pm"))    # Legacy format
        self.assertEqual(re.search(time_regex, "Time: 09:00.").group(0), "09:00")
        self.assertEqual(re.search(time_regex, "call at 15:45").group(0), "15:45")

    def test_prompt_date_time_format_instructions(self):
        # Test the CustomPromptTemplate content directly
        # Fetch the template string from where it's defined in AppointmentAgent
        # This might require making the template string a class/instance variable or loading it

        # For simplicity, let's assume we can get the template string.
        # In a real scenario, you might need to instantiate AppointmentAgent and access its prompt object,
        # or if the template is a static string, import it or copy it here.
        # The prompt is defined inside create_agent, so we need to call it or extract from there.

        # Let's get an instance of the prompt by calling create_agent and then finding the prompt
        # This is a bit indirect. A better way would be if CustomPromptTemplate was easier to get.

        # Re-patch tools for this specific test if needed, or rely on setUp's tools
        tools_for_prompt = [
            MagicMock(name='BookAppointment', description='Books an appointment.'),
            MagicMock(name='ViewAppointments', description='Views appointments.')
        ]

        # The actual template string is quite large. We'll check for key phrases.
        # This is a simplified representation of the prompt template for testing.
        # Ideally, load the actual template string.
        with open("appointment_system/agent.py", "r") as f:
            agent_code = f.read()

        template_str_match = re.search(r'template = """(.*?)"""', agent_code, re.DOTALL)
        self.assertIsNotNone(template_str_match, "Could not find template string in agent.py")
        template_str = template_str_match.group(1)

        prompt_template_obj = CustomPromptTemplate(
            template=template_str,
            tools=tools_for_prompt,
            input_variables=["input", "history", "agent_scratchpad", "name", "date", "time", "purpose", "last_action", "action_count", "has_all_info"]
        )

        formatted_prompt = prompt_template_obj.format(
            input="test input", history="", agent_scratchpad="",
            name=None, date=None, time=None, purpose=None,
            last_action=None, action_count=0, has_all_info=False,
            tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools_for_prompt]), # tools and tool_names are added by CustomPromptTemplate itself
            tool_names=[tool.name for tool in tools_for_prompt]
        )

        self.assertIn("date: \"YYYY-MM-DD\"", formatted_prompt)
        self.assertIn("time: \"HH:MM\"", formatted_prompt)
        self.assertIn("YYYY-MM-DD format (e.g., 2025-09-11)", formatted_prompt)
        self.assertIn("HH:MM 24-hour format (e.g., 09:00 or 14:30)", formatted_prompt)


    @patch('appointment_system.agent.AppointmentAgent.llm', new_callable=PropertyMock) # Mock llm property
    def test_last_action_and_count_in_prompt_format_via_call_agent(self, mock_llm_prop):
        # This tests that call_agent correctly uses state's last_action and action_count

        # We need to simulate the relevant part of call_agent
        # The graph's call_agent node will internally call self.llm.invoke with a formatted prompt
        # We need to capture the arguments to prompt.format() or llm.invoke()

        # Mock the llm's invoke method on the instance
        mock_llm_instance = MagicMock()
        mock_llm_prop.return_value = mock_llm_instance # Ensure self.agent.llm is this mock

        # Setup initial state
        initial_state = AgentState(
            messages=[{"role": "user", "content": "Hello"}],
            next="agent",
            current_step="",
            booking_info={"name": None, "date": None, "time": None, "purpose": None},
            last_action="some_previous_tool",
            action_count=2
        )

        # The real `call_agent` is a node in the graph. We need to invoke the graph.
        # The graph is self.graph from setUp.
        # The prompt object is created within create_agent. We need to ensure CustomPromptTemplate.format is checkable.

        # To check what prompt.format gets, we can patch CustomPromptTemplate.format
        with patch.object(CustomPromptTemplate, 'format', wraps=CustomPromptTemplate.format) as mock_format_method:
            # Call the agent node. This is an integration test for call_agent.
            # We expect call_agent to be triggered by graph.invoke
            self.graph.invoke(initial_state)

            # Check if format was called
            self.assertTrue(mock_format_method.called)

            # Get the kwargs passed to the last call of prompt.format
            # This depends on CustomPromptTemplate being used by the agent's prompt object.
            # The prompt object is created inside create_agent.
            # We need to ensure that the prompt used by self.agent.llm.invoke is the one we are inspecting.

            # The actual CustomPromptTemplate instance is inside create_agent's scope.
            # To make this testable without major refactor, we assume llm.invoke is called with the formatted string.
            # So, we check the string passed to llm.invoke.

            self.assertTrue(self.agent.llm.invoke.called)
            call_args_to_llm_invoke = self.agent.llm.invoke.call_args[0][0] # Get the actual formatted string

            # The prompt string itself should contain these values if correctly formatted.
            # This is an indirect way to check prompt.format() kwargs.
            self.assertIn("Last action taken: some_previous_tool", call_args_to_llm_invoke)
            self.assertIn("Number of consecutive actions: 2", call_args_to_llm_invoke)


    def test_action_count_logic_in_call_tool(self):
        # This tests the logic for updating last_action and action_count in call_tool
        # call_tool is a nested function. We test it by invoking the graph to the 'tool' node.

        # Initial state
        state = AgentState(
            messages=[], current_step="Action: ViewAppointments", next="tool",
            booking_info={}, last_action=None, action_count=0
        )

        # Simulate first call
        # To call 'call_tool', we invoke the graph.
        # We need to mock the actual tool execution (mock_tool_view.func)
        mock_tool_view.func.reset_mock()
        mock_tool_view.func.return_value = "Appointments listed."

        updated_state = self.graph.invoke(state) # This should trigger call_tool

        mock_tool_view.func.assert_called_once_with(None)
        self.assertEqual(updated_state['last_action'], "ViewAppointments")
        self.assertEqual(updated_state['action_count'], 1)
        self.assertIn("Action: ViewAppointments\nObservation: Appointments listed.", updated_state['messages'][-1]['content'])

        # Simulate second call with the same action
        state_after_first_call = updated_state.copy()
        state_after_first_call['current_step'] = "Action: ViewAppointments" # Agent decided same action
        # messages should accumulate, so pass previous messages
        state_after_first_call['messages'] = list(updated_state['messages'])


        mock_tool_view.func.reset_mock()
        mock_tool_view.func.return_value = "Appointments listed again."
        updated_state_2 = self.graph.invoke(state_after_first_call)

        mock_tool_view.func.assert_called_once_with(None)
        self.assertEqual(updated_state_2['last_action'], "ViewAppointments")
        self.assertEqual(updated_state_2['action_count'], 2)

        # Simulate a different action
        state_after_second_call = updated_state_2.copy()
        state_after_second_call['current_step'] = "Action: BookAppointment\nAction Input: {}" # Agent decided new action
        state_after_second_call['messages'] = list(updated_state_2['messages'])


        mock_tool_book.func.reset_mock()
        mock_tool_book.func.return_value = "Appointment booked."
        updated_state_3 = self.graph.invoke(state_after_second_call)

        mock_tool_book.func.assert_called_once_with({}) # Assuming empty JSON for simplicity
        self.assertEqual(updated_state_3['last_action'], "BookAppointment")
        self.assertEqual(updated_state_3['action_count'], 1) # Reset for new action

        # Simulate successful BookAppointment (resets counters)
        state_for_booking_reset = updated_state_3.copy()
        state_for_booking_reset['current_step'] = "Action: BookAppointment\nAction Input: {\"name\":\"Test\",\"date\":\"2024-01-01\",\"time\":\"10:00\",\"purpose\":\"Checkup\"}"
        state_for_booking_reset['messages'] = list(updated_state_3['messages'])

        mock_tool_book.func.reset_mock()
        mock_tool_book.func.return_value = "Appointment successfully booked for Test." # Must contain "successfully booked"

        final_state = self.graph.invoke(state_for_booking_reset)

        self.assertIsNone(final_state['last_action'])
        self.assertEqual(final_state['action_count'], 0)


    def test_call_tool_dynamic_dispatch_and_input(self):
        # Test that call_tool uses the tool_map to call the correct tool
        # And that input is passed correctly to BookAppointment

        # Test ViewAppointments (no input)
        mock_tool_view.func.reset_mock()
        mock_tool_view.func.return_value = "Viewed."
        state_view = AgentState(
            messages=[], current_step="Action: ViewAppointments", next="tool",
            booking_info={}, last_action=None, action_count=0
        )
        updated_state_view = self.graph.invoke(state_view)
        mock_tool_view.func.assert_called_once_with(None)
        self.assertIn("Observation: Viewed.", updated_state_view['messages'][-1]['content'])

        # Test BookAppointment (with input)
        mock_tool_book.func.reset_mock()
        mock_tool_book.func.return_value = "Booked."
        action_input_dict = {"name": "Test", "date": "2024-01-01", "time": "10:00", "purpose": "Test"}
        action_input_json = json.dumps(action_input_dict)

        state_book = AgentState(
            messages=[], current_step=f"Action: BookAppointment\nAction Input: {action_input_json}", next="tool",
            booking_info={}, last_action=None, action_count=0
        )
        updated_state_book = self.graph.invoke(state_book)
        mock_tool_book.func.assert_called_once_with(action_input_dict)
        self.assertIn(f"Action Input: {action_input_json}", updated_state_book['messages'][-1]['content'])
        self.assertIn("Observation: Booked.", updated_state_book['messages'][-1]['content'])

    def test_call_tool_invalid_action(self):
        state = AgentState(
            messages=[], current_step="Action: NonExistentTool", next="tool",
            booking_info={}, last_action=None, action_count=0
        )
        updated_state = self.graph.invoke(state)

        self.assertIn("Error: Tool 'NonExistentTool' not found.", updated_state['messages'][-1]['content'])
        # last_action and action_count should still update for the attempt
        self.assertEqual(updated_state['last_action'], "NonExistentTool")
        self.assertEqual(updated_state['action_count'], 1)


if __name__ == '__main__':
    # This allows running the tests directly from this file
    # However, typically a test runner like `python -m unittest discover` would be used.
    # For the subtask environment, just creating the file is the goal.
    unittest.main()
