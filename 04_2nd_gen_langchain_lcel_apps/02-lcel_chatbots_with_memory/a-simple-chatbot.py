# build a simple Chatbot with stored memory using LangChain
"""
* Simple Chatbot LLM App.
    * Will be able to have a conversation.
    * Will remember previous interactions: will have memory.
    * Will be able to store memory in a json file.

## Concepts included
* Chat Model vs. LLM Model:
    *  Chat Model is based around messages.
    *  LLM Model is based around raw text.
* Chat History: allows Chat Model to remember previous interactions.

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

# Let's add memory to our Chatbot

from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

from langchain import LLMChain

chain = LLMChain(llm=chatbot, prompt=prompt, memory=memory)

chain.invoke("my name is Julio")

chain.invoke("hello! what is my name?")


"""
* Check the file messages.json in the root directory.
* This is just a simple example, in the real world you probably will not save your memory in a json file. 
* And remember: the context window is limited and it affects to the cost of using chatGPT API.
"""
