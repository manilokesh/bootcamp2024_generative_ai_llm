# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from utils.MyModels import BaseChatModel, LlmModel, init_llm
from utils.MyUtils import logger

########################################################################
# region Initialise foundation LLM

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# endregion Initialise foundation LLM

########################################################################

from langchain_community.document_loaders import DirectoryLoader, TextLoader

path_dir = "data"
path_file = "be-good-and-how-not-to-die.txt"
logger.info(f"Loaging file {path_dir} {path_file}")
loader = DirectoryLoader(
    path_dir, glob=path_file, loader_cls=TextLoader, show_progress=True
)
document = loader.load()

print(type(document))
print(len(document))
print(document[0].metadata)
print(f"You have {len(document)} document.")
print(f"Your document has {len(document[0].page_content)} characters")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)

document_chunks = text_splitter.split_documents(document)

print(f"Now you have {len(document_chunks)} chunks.")

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()

# We will use cromadb as vector database:
# > pip install langchain_chroma

from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    collection_name="lc_code_collection", embedding_function=my_embeddings
)

########################################################################


# create a chain that takes the question and the retrieved documents and generates an answer

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

my_prompt_template = ChatPromptTemplate.from_template(
    """Answer the following question based only on the 
    provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
)

my_document_answering_chain = create_stuff_documents_chain(llm, my_prompt_template)

########################################################################

# create the retrieval chain

from langchain.chains import create_retrieval_chain

my_retriever = vectorstore.as_retriever()

my_retrieval_chain = create_retrieval_chain(my_retriever, my_document_answering_chain)

########################################################################

# start using the retrieval chain

# = my_retrieval_chain.invoke(
#    {"input": "Summarize the provided context in less than 100 words"}
# )

# print(response["answer"])

########################################################################

from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/conversation")
async def conversation(query: str):
    try:
        result = my_retrieval_chain.invoke({"input": query})
        return {"response": result}
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
