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

def chat(messages, system=None):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
    }
    
    if system:
        params["system"] = system
    
    message = client.messages.create(**params)
    return message.content[0].text

# List of messages
messages = []

# System prompt
system = '''
You are a patient teacher.
Do not directly answer a student's questions.
Guide them to a solution step by step.
'''

add_user_message(messages, "There are 4 teams in a football tournament. Each team plays against every other team. How many matches will there be?")
answer = chat(messages, system=system)

print(answer)
