# built-in LCEL functions for runnables
"""
* .bind()
* .assign()
"""

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True


# Connect with an LLM

from utils.MyModels import LlmModel, init_llm

model = init_llm(LlmModel.MISTRAL, temperature=0)


# LCEL Chain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "tell me a curious fact about {soccer_player}"
)

output_parser = StrOutputParser()

# --------------------------------------------------

# The order (left to right) of the elements in a LCEL chain matters.
# An LCEL Chain is a Sequence of Runnables.

chain = prompt | model | output_parser

chain.invoke({"soccer_player": "Ronaldo"})

# --------------------------------------------------


# Use of .bind() to add arguments to a Runnable in a LCEL Chain
# For example, we can add an argument to stop the model response when it reaches the word "Ronaldo":

chain = prompt | model.bind(stop=["Ronaldo"]) | output_parser

chain.invoke({"soccer_player": "Ronaldo"})

# --------------------------------------------------

# Use of .bind() to call an OpenAI Function in a LCEL Chain

functions = [
    {
        "name": "soccerfacts",
        "description": "Curious facts about a soccer player",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question for the curious facts about a soccer player",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to the question",
                },
            },
            "required": ["question", "answer"],
        },
    }
]

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "soccerfacts"}, functions=functions)
    | JsonOutputFunctionsParser()
)

chain.invoke(input={"soccer_player": "Mbappe"})

# --------------------------------------------------

# The assign() function allows adding keys to a chain
# Example: we will create a key name "operation_b" assigned to a custom function with a RunnableLambda.
# We will start with a very basic chain with just RunnablePassthrough:

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

chain = RunnableParallel({"original_input": RunnablePassthrough()})

chain.invoke("whatever")

# --------------------------------------------------


# Let's now add the new key "uppercase" with the assign function.
# In the new "uppercase" key, we will use a RunnableLambda with the custom function named `make_uppercase`


def make_uppercase(arg):
    return arg["original_input"].upper()


chain = RunnableParallel({"original_input": RunnablePassthrough()}).assign(
    uppercase=RunnableLambda(make_uppercase)
)

chain.invoke("whatever")

# --------------------------------------------------
