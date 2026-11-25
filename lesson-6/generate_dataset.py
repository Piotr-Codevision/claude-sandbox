#!/usr/bin/env python3

from dotenv import load_dotenv
from anthropic import Anthropic
import json

# Load enf variables
load_dotenv()

# Create an API client
client = Anthropic()
model = "claude-haiku-4-5"

# Helper functions
def add_user_message(messages, message_text):
    user_message = {
        "role": "user",
        "content": message_text
    }
    messages.append(user_message)

def add_assistant_message(messages, message_text):
    assistant_message = {
        "role": "assistant",
        "content": message_text
    }
    messages.append(assistant_message)

def chat(messages, system=None, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    response = client.messages.create(**params)
    return response.content[0].text

def generate_dataset():
    prompt = """
    Generate an evaluation dataset for a prompt evaluation.
    The dataset will be used to evaluate prompts that generate Python, JSON, or Regex
    specifically for AWS-related tasks. Generate an array of JSON objects, each representing
    task that requires Python, JSON, or a Regex to complete.

    Example output:
    ```json
    [
    {
        "task": "Description of task",
        "format": "json|python|regex"
    },
    ...additional
    ]
    ```

    * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
    * Focus on tasks that do not require writing much code

    Please generate 3 objects.
    """

    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)

dataset = generate_dataset()
print(dataset)

with open('dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)
