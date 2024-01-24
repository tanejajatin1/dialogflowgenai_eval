import uuid
import csv
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from google.cloud.dialogflowcx_v3beta1.services.agents import AgentsClient
from google.cloud.dialogflowcx_v3beta1.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3beta1.types import session
from google.protobuf.json_format import MessageToDict

# [START dialogflow_cx_detect_intent_text]
def run_sample():
    project_id = os.getenv("project_id")
    location_id = os.getenv("location_id")
    agent_id = os.getenv("agent_id")
    agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"
    session_id = uuid.uuid4()
    language_code = "en"

    detect_intent_texts(agent, session_id, language_code)


def detect_intent_texts(agent, session_id, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    session_path = f"{agent}/sessions/{session_id}"
    print(f"Session path: {session_path}\n")
    client_options = None
    agent_components = AgentsClient.parse_agent_path(agent)
    location_id = agent_components["location"]
    if location_id != "global":
        api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
        print(f"API Endpoint: {api_endpoint}\n")
        client_options = {"api_endpoint": api_endpoint}
    session_client = SessionsClient(client_options=client_options)

    # Set the folder name based on the current working directory
    json_folder_path = os.path.join(os.getcwd(), 'json_responses')
    os.makedirs(json_folder_path, exist_ok=True)

    # Read queries from CSV file
    with open('tests.csv', 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        fieldnames = csv_reader.fieldnames
        if 'answer' not in fieldnames:
            fieldnames += ['answer']
        
        if 'link' not in fieldnames:
            fieldnames += ['link']  # Add 'Link' as a new column

        if 'contexts' not in fieldnames:
            fieldnames += ['contexts']

        # Create or update CSV file with the 'Response' column
        with open('tests.csv', 'w', newline='', encoding='utf-8') as result_file:
            csv_writer = csv.DictWriter(result_file, fieldnames=fieldnames)

            # Write header
            csv_writer.writeheader()

            for row in csv_reader:
                query = row['question']
                text_input = session.TextInput(text=query)
                query_input = session.QueryInput(text=text_input, language_code=language_code)
                request = session.DetectIntentRequest(session=session_path, query_input=query_input)
                response = session_client.detect_intent(request=request)
                response_messages = []
                link = ''

                for msg in response.query_result.response_messages:
                    if msg.text:
                        response_messages.extend(msg.text.text)
                    elif msg.payload and 'richContent' in msg.payload:
                        link = msg.payload['richContent'][0][0]['actionLink']
                # response_messages = [" ".join(msg.text.text) for msg in response.query_result.response_messages]
                response_text = f"{' '.join(response_messages)}"

                # Extract DataStoreSequence from diagnosticInfo
                diagnostic_info = response.query_result.diagnostic_info
                data_store_sequence_dict = diagnostic_info.get('DataStore Execution Sequence', {})

                # Extract steps from DataStoreSequence
                steps = data_store_sequence_dict.get('steps', [])

                # Filter steps with name starting with "Convert UCS results"
                convert_ucs_steps = [step for step in steps if step.get('name', '').startswith('Convert UCS results')]

                # Extract text from each "Convert UCS results" step
                context_texts = [response['text'] for step in convert_ucs_steps for response in step.get('responses', [])]

                # Append the response to the CSV file
                row['answer'] = response_text

                # Append link to csv file
                row['link'] = link

                # Append the context to the CSV file
                row['contexts'] = context_texts

                csv_writer.writerow(row)

                # Create a JSON file for each row in the current working directory
                json_filename = f"{os.path.join(json_folder_path, str(csv_reader.line_num-1))}.json"
                with open(json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(MessageToDict(response._pb), json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_sample()