
# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.MyUtils import clear_terminal, logger

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import utils.MyUtils
from utils.MyModels import BaseChatModel, LlmModel, init_llm

# https://www.datacamp.com/tutorial/run-llama-3-locally

# loading the vectorstore
"""
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OllamaEmbeddings(model=LlmModel.MISTRAL.value),
)
"""


from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()

from utils.MyVectorStore import chroma_get

vectorstore = chroma_get(
    collection_name="qa_retrieval_chain", embedding_function=my_embeddings
)

# loading the Llama3 model
llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# using the vectorstore as the retriever
retriever = vectorstore.as_retriever()


# formating the docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# loading the QA chain from langchain hub
rag_prompt = hub.pull("rlm/rag-prompt")

print(rag_prompt)

# creating the QA chain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# running the QA chain in a loop until the user types "exit"
while True:
    question = input("Question: ")
    if question.lower() == "exit":
        break
    answer = qa_chain.invoke(question)

    print(f"\nAnswer: {answer}\n")
