# Evaluation of a QA App
"""
evaluate the quality of a question and answer application on a document (QA App)

● Load the text document with a document loader.
● Split the document into fragments with a text splitter.
● Convert the fragments into embeddings with OpenAIEmbeddings.
● Load the embeddings into a FAISS vector database.
● Create a RetrievalQA chain to retrieve the data including an input_key to identify the user's prompt (the question).
● Create a dictionary with the evaluation questions and answers.
● Use the RetrievalQA chain to manually evaluate the App.
● Use a QAEvalChain chain for the App to evaluate itself.
"""

# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.MyUtils import clear_terminal, logger

clear_terminal()

from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# Load document

from langchain_community.document_loaders.text import TextLoader

loader = TextLoader("data/be-good-and-how-not-to-die.txt")
document = loader.load()

print(f"The document has {len(document[0].page_content)} characters")

# Split the document in smaller chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)

document_chunks = text_splitter.split_documents(document)

print(f"Now you have {len(document_chunks)} chunks")
print(type(document_chunks))

# Convert text chunks in numeric embeddings and load them to the vector database

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()

from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=document_chunks, embedding=my_embeddings, collection_name="app_evaluation"
)

# Create a Retrieval Question & Answering Chain


from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA

QA_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    input_key="question",
)

# We are going to evaluate this app with 2 questions and answers we already know
# (these answers are technically known as "ground truth answers

questions_and_answers = [
    {
        "question": "Where is a whole neighborhood of YC-funded startups?",
        "answer": "In San Francisco",
    },
    {
        "question": "What may be the most valuable  thing Paul Buchheit made for Google?",
        "answer": "The motto Don't be evil",
    },
]

from langchain.callbacks import StdOutCallbackHandler

predictions = QA_chain.invoke(
    questions_and_answers, config={"callbacks": [StdOutCallbackHandler()]}
)

print(predictions)

# result from chain is expected in below format
"""
predictions = [
    {
        "question": "Where is a whole neighborhood of YC-funded startups?",
        "answer": "In San Francisco",
        "result": " A whole neighborhood of YC-funded startups is in San Francisco.",
    },
    {
        "question": "What may be the most valuable  thing Paul Buchheit made for Google?",
        "answer": "The motto Don't be evil",
        "result": ' Paul Buchheit is credited with creating the phrase "Don\'t be evil," which serves as the motto for Google. This phrase may be the most valuable thing Buchheit made for Google, as it serves as a reminder for the company to stay true to its mission.',
    },
]
"""

# The evaluation of this App has been positive, since the App has responded the 2 evaluation questions right.
# But instead of confirming that manually ourselves,
# we can ask the LLM to check if the responses are coincidental with the "ground truth answers

from langchain.evaluation.qa import QAEvalChain

evaluation_chain = QAEvalChain.from_llm(llm)

evaluate_responses = evaluation_chain.evaluate(
    questions_and_answers, predictions, question_key="question", answer_key="answer"
)

print(evaluate_responses)
