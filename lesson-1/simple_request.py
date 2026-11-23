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

def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=messages
    )
    return message.content[0].text

# List of messages
messages = []

# Add the initial user question
add_user_message(messages, "What is a photosynthesis? Give short, maximum 2 sentence explanation.")
answer = chat(messages)
print(answer)

# Add the assistant response
add_assistant_message(messages, answer)

# Add another user message related to previous one
add_user_message(messages, "Add one more sentence.")
answer = chat(messages)
print(answer)
