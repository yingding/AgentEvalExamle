
"""
DESCRIPTION:
    This sample demonstrates multi-agent interaction using Azure Agents service
    with Azure Monitor tracing. It shows how two agents can work together to
    provide weather information with temperature conversions.

USAGE:
    python multi_agent_tracing.py

    Before running the sample:
    pip install azure-ai-projects azure-identity opentelemetry-sdk azure-monitor-opentelemetry

    Set these environment variables:
    1) PROJECT_CONNECTION_STRING - The project connection string
    2) MODEL_DEPLOYMENT_NAME - The model deployment name
    3) AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED - Optional, set to `true` to trace chat content
"""

from typing import Any, Callable, Set
import os
import time
import json
from azure.ai.projects import AIProjectClient
from azure.ai.projects.telemetry import trace_function
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import (
    FunctionTool,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    ToolOutput,
)
from opentelemetry import trace
from azure.monitor.opentelemetry import configure_azure_monitor
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(), conn_str=os.environ["PROJECT_CONNECTION_STRING"]
)

# Enable Azure Monitor tracing
application_insights_connection_string = project_client.telemetry.get_connection_string()
if not application_insights_connection_string:
    print("Application Insights was not enabled for this project.")
    print("Enable it via the 'Tracing' tab in your AI Foundry project page.")
    exit()
configure_azure_monitor(connection_string=application_insights_connection_string)

# Enable additional instrumentations
project_client.telemetry.enable()

scenario = os.path.basename(__file__)
tracer = trace.get_tracer(__name__)


@trace_function()
def fetch_weather(location: str) -> str:
    """
    Fetches the weather information for the specified location.

    :param location (str): The location to fetch weather for.
    :return: Weather information as a JSON string.
    :rtype: str
    """
    mock_weather_data = {"New York": "Sunny, 25°C", "London": "Cloudy, 18°C", "Tokyo": "Rainy, 22°C"}

    # Adding attributes to the current span
    span = trace.get_current_span()
    span.set_attribute("requested_location", location)

    weather = mock_weather_data.get(location, "Weather data not available for this location.")
    weather_json = json.dumps({"weather": weather})
    return weather_json


@trace_function()
def convert_temperature(temperature: str) -> str:
    """
    Converts temperature between Celsius and Fahrenheit.
    
    :param temperature: Temperature string in format "25°C" or "77°F"
    :return: Converted temperature as JSON string
    :rtype: str
    """
    span = trace.get_current_span()
    span.set_attribute("input_temperature", temperature)
    
    try:
        value = float(''.join(filter(str.isdigit, temperature)))
        unit = 'C' if '°C' in temperature else 'F'
        
        if unit == 'C':
            converted = (value * 9/5) + 32
            result = f"{converted:.1f}°F"
        else:
            converted = (value - 32) * 5/9
            result = f"{converted:.1f}°C"
            
        return json.dumps({"converted_temperature": result})
    except Exception as e:
        return json.dumps({"error": f"Failed to convert temperature: {str(e)}"})


def check_for_celsius_in_messages(messages):
    """Helper function to check if any message contains a Celsius temperature"""
    for msg in messages:
        try:
            if hasattr(msg, 'content') and isinstance(msg.content, str) and '°C' in msg.content:
                return True
        except AttributeError:
            continue
    return False


def process_agent_run(thread_id: str, run_id: str, functions: FunctionTool) -> None:
    """Helper function to process an agent run and handle tool calls"""
    run = project_client.agents.get_run(thread_id=thread_id, run_id=run_id)
    
    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(1)
        run = project_client.agents.get_run(thread_id=thread_id, run_id=run_id)

        if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            if not tool_calls:
                print("No tool calls provided - cancelling run")
                project_client.agents.cancel_run(thread_id=thread_id, run_id=run_id)
                break

            tool_outputs = []
            for tool_call in tool_calls:
                if isinstance(tool_call, RequiredFunctionToolCall):
                    try:
                        output = functions.execute(tool_call)
                        tool_outputs.append(
                            ToolOutput(tool_call_id=tool_call.id, output=output)
                        )
                    except Exception as e:
                        print(f"Error executing tool_call {tool_call.id}: {e}")

            if tool_outputs:
                project_client.agents.submit_tool_outputs_to_run(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )

        print(f"Current run status: {run.status}")
    
    return run


# Initialize functions
user_functions: Set[Callable[..., Any]] = {fetch_weather, convert_temperature}
functions = FunctionTool(functions=user_functions)

with tracer.start_as_current_span(scenario):
    with project_client:
        # Create two agents with different roles
        # Create the weather assistant agent
        weather_agent = project_client.agents.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="weather-assistant",
            instructions="""You are a helpful weather assistant. Follow these steps:
1. When asked about weather, politely acknowledge the request
2. Use the fetch_weather function to get the weather information
3. Present the weather information in a friendly, conversational way
4. If the temperature is in Celsius, mention that you'll ask for a conversion to Fahrenheit
Be descriptive and natural in your responses.""",
            tools=functions.definitions,
        )
        print(f"Created weather agent, ID: {weather_agent.id}")

        # Create the temperature conversion agent
        conversion_agent = project_client.agents.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="conversion-assistant",
            instructions="""You are a helpful temperature conversion assistant. Follow these steps:
1. When asked to convert a temperature, acknowledge the request
2. Extract the temperature value from the previous messages
3. Use the convert_temperature function to perform the conversion
4. Present both temperatures in a clear, friendly way
Be detailed and explain the conversion clearly.""",
            tools=functions.definitions,
        )
        print(f"Created conversion agent, ID: {conversion_agent.id}")

        # Try block content for conversation display
        try:
            # Create a thread for the conversation
            thread = project_client.agents.create_thread()
            print(f"Created thread, ID: {thread.id}")

            # User asks about weather with a clear request
            message = project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content="Can you tell me what the weather is like in New York today? I'd like to know the temperature in both Celsius and Fahrenheit, please.",
            )
            print(f"Created initial user message, ID: {message.id}")

            # Start with the weather agent for the initial response
            weather_run = project_client.agents.create_run(
                thread_id=thread.id,
                agent_id=weather_agent.id
            )
            print(f"Started weather agent run, ID: {weather_run.id}")

            # Process the weather agent's run
            weather_run = process_agent_run(thread.id, weather_run.id, functions)
            print(f"Weather agent run completed with status: {weather_run.status}")

            # Check messages for Celsius temperatures
            messages = list(project_client.agents.list_messages(thread_id=thread.id))
            if check_for_celsius_in_messages(messages):
                # Create a message for the conversion agent with context
                conversion_msg = project_client.agents.create_message(
                    thread_id=thread.id,
                    role="user",
                    content="Could you help convert this Celsius temperature to Fahrenheit so we can see both? Please explain the conversion."
                )
                print(f"Created conversion request message, ID: {conversion_msg.id}")

                # Start the conversion agent's run
                conversion_run = project_client.agents.create_run(
                    thread_id=thread.id,
                    agent_id=conversion_agent.id
                )
                print(f"Started conversion agent run, ID: {conversion_run.id}")

                # Process the conversion agent's run
                conversion_run = process_agent_run(thread.id, conversion_run.id, functions)
                print(f"Conversion agent run completed with status: {conversion_run.status}")

            # Display the full conversation
            print("\nFull conversation:")
            final_messages = list(project_client.agents.list_messages(thread_id=thread.id))
            final_messages.reverse()  # Show messages in chronological order
            for msg in final_messages:
                try:
                    if hasattr(msg, 'content'):
                        role = msg.role if hasattr(msg, 'role') else 'system'
                        print(f"{role}: {msg.content}")
                except Exception as e:
                    print(f"Error printing message: {e}")

        finally:
            # Clean up resources
            print("\nCleaning up resources...")
            project_client.agents.delete_agent(weather_agent.id)
            project_client.agents.delete_agent(conversion_agent.id)
            print("Agents deleted successfully")