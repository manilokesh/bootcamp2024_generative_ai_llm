# build an advanced Chatbot with session memory using LangChain
"""
* Advanced Chatbot LLM App.
    * Will be able to have a conversation.
    * Will remember previous interactions: will have memory.
    * Will be able to have different memories for different user sessions.
    * Will be able to remember a limited number of messages: limited memory.
"""


import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True


from langchain_core.messages import HumanMessage

messagesToTheChatbot = [
    HumanMessage(content="My favorite color is blue."),
]

from utils.MyModels import LlmModel, init_llm

chatbot = init_llm(LlmModel.MISTRAL, temperature=0)

chatbot.invoke(messagesToTheChatbot)

# Check if the Chatbot remembers your favorite color.

chatbot.invoke(
    [
        HumanMessage(content="What is my favorite color?"),
    ]
)

# Chatbot cannot remember our previous interaction.

"""
## Let's add memory to our Chatbot
* We will use the ChatMessageHistory package.
* We will save the Chatbot memory in a python dictionary called chatbotMemory.
* We will define the get_session_history function to create a session_id for each conversation.
"""
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

chatbotMemory = {}


# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]


from langchain_core.runnables.history import RunnableWithMessageHistory

chatbot_with_message_history = RunnableWithMessageHistory(chatbot, get_session_history)

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My name is Indian and My favorite color is red.")],
    config=session1,
)

print(responseFromChatbot.content)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session1,
)

print(responseFromChatbot.content)
