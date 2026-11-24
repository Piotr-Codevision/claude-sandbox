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

# List of messages
messages = []

add_user_message(messages, "Write a 10 sentences novel about a little dragon lost in the abandoned spaceship.")

# stream = client.messages.create(
#     model=model,
#     max_tokens=1000,
#     messages=messages,
#     stream=True
# )

# for event in stream:
#     print(event)

with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=messages
) as stream:
    for text in stream.text_stream:
        print(text, end="")

# Get the complete message
final_message = stream.get_final_message()
# print(final_message)
