#!/usr/bin/env python3

from dotenv import load_dotenv
from anthropic import Anthropic

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

    message = client.messages.create(**params)
    return message.content[0].text

# List of messages
messages = []

# Add the initial user question
add_user_message(messages, "Generate a 'Hello World!' page as html")
add_assistant_message(messages, "```html")

answer = chat(messages, stop_sequences=["```"])
print(answer.strip())
