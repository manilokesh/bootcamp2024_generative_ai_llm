# Main Operations with LCEL Chains
"""
* Chaining runnables, coercion.
* Multiple chains.
* Nested chains.
* Fallback for chains.
"""

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

# --------------------------------------------------

# Connect with an LLM

from utils.MyModels import LlmModel, init_llm

model = init_llm(LlmModel.MISTRAL, temperature=0)


# --------------------------------------------------

# Basic chain: LLM model + prompt + output parser

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Write one brief sentence about {politician}")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

# chain.invoke({"politician": "JFK"})

# --------------------------------------------------

# Mid-level chain: Retriever App Example

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

vectorstore = DocArrayInMemorySearch.from_texts(
    [
        "AI Accelera has provided Generative AI Training and Consulting Services in more than 100 countries",
        "Aceleradora AI is the branch of AI Accelera for the Spanish-Speaking market",
    ],
    embedding=SentenceEmbeddingFunction(),
)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()


from langchain_core.runnables import RunnableParallel, RunnablePassthrough

get_question_and_retrieve_relevant_docs = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

"""
The previous code creates a `RunnableParallel` object with two entries.
    * The first entry, context, will include the document results fetched by the retriever.
    * The second entry, question, will contain the user’s original question. To pass on the question, we use `RunnablePassthrough` to copy this entry.
"""
chain = get_question_and_retrieve_relevant_docs | prompt | model | output_parser

# chain.invoke("In how many countries has AI Accelera provided services?")

# --------------------------------------------------

"""
* Remember: the order of operations in a chain matters. If you try to execute the previous chain with the operations in different order, the chain will fail. We will talk more about this in a next section below.
* In the previous exercise, this is the right order:
    1. User input via RunnablePassthroug.
    2. Prompt.
    3. LLM Model.
    4. Output Parser.
"""


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from utils.MyVectorStore import chroma_from_texts

vectorstore = chroma_from_texts(
    texts=[
        "AI Accelera has trained more than 7.000 Alumni from all continents and top companies"
    ],
    embedding=SentenceEmbeddingFunction(),
    persist_directory="langclain_el_main_operations",
    collection_name="RunnableParallel",
)


retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = init_llm(LlmModel.MISTRAL, temperature=0)


retrieval_chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# retrieval_chain.invoke("who are the Alumni of AI Accelera?")

# --------------------------------------------------

"""
Rememeber: the syntax of RunnableParallel can have several variations.
* When composing a RunnableParallel with another Runnable you do not need to wrap it up in the RunnableParallel class. Inside a chain, the next three syntaxs are equivalent:
    * `RunnableParallel({"context": retriever, "question": RunnablePassthrough()})`
    * `RunnableParallel(context=retriever, question=RunnablePassthrough())`
    * `{"context": retriever, "question": RunnablePassthrough()}`
"""

# --------------------------------------------------


from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

vectorstore = chroma_from_texts(
    texts=["AI Accelera has trained more than 3,000 Enterprise Alumni."],
    embedding=SentenceEmbeddingFunction(),
    persist_directory="langclain_el_main_operations",
    collection_name="itemgetter",
)


retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

model = init_llm(LlmModel.MISTRAL, temperature=0)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# chain.invoke(
#     {
#         "question": "How many Enterprise Alumni has trained AI Accelera?",
#         "language": "Pirate English",
#     }
# )

# --------------------------------------------------

"""
RunnablePassthrough
* Allows you to pass inputs unchanged.
"""

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    user_input=RunnablePassthrough(),
    transformed_output=lambda x: x["num"] + 1,
)

# runnable.invoke({"num": 1})

# --------------------------------------------------

"""
## Chaining Runnables
* Remember: almost any component in LangChain (prompts, models, output parsers, etc) can be used as a Runnable.
* **Runnables can be chained together using the pipe operator `|`. The resulting chains of runnables are also runnables themselves**.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a sentence about {politician}")

model = init_llm(LlmModel.MISTRAL, temperature=0)

chain = prompt | model | StrOutputParser()

# --------------------------------------------------

"""
#### Coercion: combine a chain (which is a Runnable) with other Runnables to create a new chain.
* See how in the `composed_chain` we are including the previous `chain`:
"""

from langchain_core.output_parsers import StrOutputParser

historian_prompt = ChatPromptTemplate.from_template(
    "Was {politician} positive for Humanity?"
)

composed_chain = {"politician": chain} | historian_prompt | model | StrOutputParser()

# composed_chain.invoke({"politician": "Lincoln"})

# composed_chain.invoke({"politician": "Attila"})


# --------------------------------------------------

"""
* **Functions can also be included in Runnables**:
"""

composed_chain_with_lambda = (
    chain
    | (lambda input: {"politician": input})
    | historian_prompt
    | model
    | StrOutputParser()
)

# composed_chain_with_lambda.invoke({"politician": "Robespierre"})

# --------------------------------------------------

"""
Multiple chains
"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt1 = ChatPromptTemplate.from_template("what is the country {politician} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what continent is the country {country} in? respond in {language}"
)

model = init_llm(LlmModel.MISTRAL, temperature=0)

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"country": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

# chain2.invoke({"politician": "Miterrand", "language": "French"})

# --------------------------------------------------

"""
 Nested Chains
 """

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

prompt = ChatPromptTemplate.from_template(
    "tell me a curious fact about {soccer_player}"
)

output_parser = StrOutputParser()


def russian_lastname_from_dictionary(person):
    return person["name"] + "ovich"


chain = (
    RunnableParallel(
        {
            "soccer_player": RunnablePassthrough()
            | RunnableLambda(russian_lastname_from_dictionary),
            "operation_c": RunnablePassthrough(),
        }
    )
    | prompt
    | model
    | output_parser
)

# chain.invoke({"name1": "Jordam", "name": "Abram"})

# --------------------------------------------------

"""
## Fallback for Chains
* When working with language models, you may often encounter issues from the underlying APIs, whether these be rate limiting or downtime. Therefore, as you go to move your LLM applications into production it becomes more and more important to safeguard against these. That's why LangChain introduced the concept of fallbacks.
* A fallback is an alternative plan that may be used in an emergency.
* Fallbacks can be applied not only on the LLM level but on the whole runnable level. This is important because often times different models require different prompts. So if your call to OpenAI fails, you don't just want to send the same prompt to Anthropic - you probably want to use a different prompt template and send a different version there.
* We can create fallbacks for LCEL chains. Here we do that with two different models: ChatOpenAI (with a bad model name to easily create a chain that will error) and then normal OpenAI (which does not use a chat model). Because OpenAI is NOT a chat model, you likely want a different prompt.
"""

# First let's create a chain with a ChatModel
# We add in a string output parser here so the outputs between the two are the same type
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a funny assistant who always includes a joke in your response",
        ),
        ("human", "Who is the best {sport} player worldwide?"),
    ]
)
# Here we're going to use a bad model name to easily create a chain that will error
# chat_model = ChatOpenAI(model="gpt-fake")
chat_model = init_llm(LlmModel.MISTRAL, temperature=1)

bad_chain = chat_prompt | chat_model | StrOutputParser()


# Now lets create a chain with the normal OpenAI model
from langchain_core.prompts import PromptTemplate

prompt_template = """Instructions: You're a funny assistant who always includes a joke in your response.

Question: Who is the best {sport} player worldwide?"""

prompt = PromptTemplate.from_template(prompt_template)

# llm = OpenAI()
llm = init_llm(LlmModel.MISTRAL, temperature=0)

good_chain = prompt | llm

# We can now create a final chain which combines the two
chain = bad_chain.with_fallbacks([good_chain])


chain.invoke({"sport": "soccer"})

# --------------------------------------------------
