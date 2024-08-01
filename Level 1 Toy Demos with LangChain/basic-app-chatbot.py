# Chatbot with Personality and Memory
"""
to create a chatbot with personality and memory.

Use a chain with an LLM, a prompt, and the chatbot's memory.

● Define the chatbot's personality.
● Include the personality in the Prompt Template.
● Set up the chatbot's memory.
● Create the chatbot using a chain with the LLM, the prompt, and the memory.
● Ask questions to check its personality and memory.
"""

# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.MyUtils import clear_terminal, logger

clear_terminal()

# Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel, BaseChatModel

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# Define the personality of the chatbot

chatbot_role = """
You are Master Yoda, a warrior and a monk.
Your goal is to help the user to strengthen her performance and spirit.

{chat_history}
Human: {human_input}
Chatbot:
"""

# Include the personality of the chatbot in a PromptTemplate

from langchain.prompts.prompt import PromptTemplate

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=chatbot_role
)

# Configure the memory of the chatbot

from langchain.memory import ConversationBufferMemory

chatbot_memory = ConversationBufferMemory(memory_key="chat_history")

# Create the yoda_chatbot using a chain with the LLM, the prompt, and the chatbot memory
 
from langchain import LLMChain 

yoda_chatbot = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=chatbot_memory)

# Ask your questions to the chatbot

question = "Master Yoda, how should I have to face my day?"

result = yoda_chatbot.predict(human_input=question)
print(result)

question2 = """
Master Yoda,
How can I deal with an enemy that wants to kill me?
"""

result = yoda_chatbot.predict(human_input=question2)
print(result)

question3 = """
Master Yoda,
Do you remember what was my first question today?
"""

result = yoda_chatbot.predict(human_input=question3)
print(result)
