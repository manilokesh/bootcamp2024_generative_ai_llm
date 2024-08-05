# Extract Structured Data from a Conversation
"""
Having the text of a chat conversation in which a person talks about their favorite song.
Need to extract the names of the song and singer and archive them in JSON format.

● Use ResponseSchema to determine which data we want to extract.
● Use StructuredOutputParser to archive the extracted data in a JSON dictionary.
● Create the ChatPromptTemplate.
● Input the user message.
● Extract the data and archive it in JSON format.
"""

# Import package from parent folder
import os
import sys

sys.path.append(os.path.abspath(os.curdir)) 

from utils.MyUtils import clear_terminal, logger

clear_terminal()

# to extract from a ChatMessage the song and artist a user wants to play

# Define your extraction goal (called "the response schema")

from langchain.output_parsers import ResponseSchema

response_schemas = [
    ResponseSchema(name="singer", description="name of the singer"),
    ResponseSchema(name="song", description="name of the song"),
]

# Create the Output Parser that will extract the data

from langchain.output_parsers import StructuredOutputParser

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create the format instructions

format_instructions = output_parser.get_format_instructions()

print(format_instructions)

# Create the ChatPromptTemplate

from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

template = """
Given a command from the user,
extract the artist and song names
{format_instructions}
{user_prompt}
"""

prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(template)],
    input_variables={"user_prompt"},
    partial_variables={"format_instructions": format_instructions},
)

# Enter the chat message from the user

user_message = prompt.format_prompt(
    user_prompt="I like the song New York, New York by Frank Sinatra"
)

user_chat_message = llm.invoke(user_message.to_messages())

print(user_chat_message)

print(type(user_chat_message))

extraction = output_parser.parse(user_chat_message)

print(extraction)

print(type(extraction))
