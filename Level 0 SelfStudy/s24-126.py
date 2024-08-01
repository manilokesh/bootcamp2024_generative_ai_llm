
########################################################################
# region Logging

import coloredlogs
coloredlogs.install(level='DEBUG') 
# endregion Logging

########################################################################
# region LOAD ENVIRONMENT VARIABLES

#> pip install pip install python-dotenv 

from dotenv import load_dotenv, find_dotenv
import os

# Load the environment variables from the .env file
# find_dotenv() ensures the correct path to .env is used
dotenv_path = find_dotenv()
if dotenv_path == "":
    print("No .env file found.")
else:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)

# Get the value of a specific environment variable
google_modal_name = os.getenv('GOOGLE_MODEL_NAME') 
google_api_key = os.getenv('GOOGLE_API_KEY') 
embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME') 
embeddings_model_path = os.getenv('EMBEDDINGS_MODEL_PATH') 
chroma_db_path = os.getenv('CHROMA_DB_PATH') 
# endregion LOAD ENVIRONMENT VARIABLES

########################################################################
# region Initialise foundation LLM 

#> pip install langchain
#> pip install langchain_google_genai

from langchain_google_genai import ChatGoogleGenerativeAI
 
# The `GOOGLE_API_KEY`` environment variable set with your API key, or 
# Pass your API key using the google_api_key kwarg to the ChatGoogle constructor.

llm = ChatGoogleGenerativeAI(model=google_modal_name, 
                                google_api_key=google_api_key,
                                temperature=0.5)

# endregion Initialise foundation LLM 

########################################################################
# region add langchain logging

#> pip install langchain

import langchain
langchain.debug = True
langchain.verbose = True
# endregion add langchain logging

########################################################################

#result = llm.invoke("What was the name of Napoleon's wife?")
#print(result)

########################################################################
# Create a chain with LCEL

#> pip install langchain_core

from langchain_core.prompts import ChatPromptTemplate

my_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are friendly assisstant."),
    ("user", "{input}")
])

my_chain = my_prompt_template | llm 

#result = my_chain.invoke({"input": "Where was Napoleon defeated?"})
#print(result)

########################################################################
# Create an Output Parser to convert the chat message to a string

from langchain_core.output_parsers import StrOutputParser

to_string_output_parser = StrOutputParser()

#Add the Output Parser to the Chain
my_chain = my_prompt_template | llm | to_string_output_parser

#result = my_chain.invoke({"input": "Where was the main victory of Napoleon?"})
#print(result)

########################################################################
# Simple RAG: Private Document, Splitter, Vector Database and Retrieval Chain.

# We can load our private document from different sources (from a file, from the web, etc). 
# In this example we will load our private data from the web using WebBaseLoader. 
# In order to use WebBaseLoader we will need to install BeautifulSoup

#> pip install langchain_community
#> pip install beautifulsoup4

from langchain_community.document_loaders import WebBaseLoader

my_loader = WebBaseLoader("https://aiaccelera.com/ai-consulting-for-businesses/")

my_private_docs = my_loader.load()
 
########################################################################

# We will use SentenceTransformer to convert our private docs to numbers

#> pip install sentence_transformers

from sentence_transformers import SentenceTransformer

# gets the projects base path
dir_path_project = os.path.abspath(os.curdir)
# appends the project base path with configured path
transformer_model_path = os.path.abspath(os.path.join(dir_path_project, embeddings_model_path))

# download model locally to configured path
if not os.path.exists( embeddings_model_path):
    # If not, download and save the model
    model = SentenceTransformer(embeddings_model_name)
    model.save(embeddings_model_path) 
 
class SentenceEmbeddingFunction:
    def __init__(self, transformer_model_path):
        self.model = SentenceTransformer(transformer_model_path)
        self.chunk_size = 500

    def embed_text(self, text):
        return self.model.encode(text).tolist()
    
    def embed_documents(self, documentation: str) -> str:
        chunked_texts = []
        chunked_texts.extend(self.chunk_text(documentation))
        embeddings = []
        for i, text in enumerate(chunked_texts):
            embedding = self.embed_text(text)
            embeddings.extend(embedding)
        return embeddings
    
    def embed_query(self, text: str):
        return self.embed_documents([text])[0]
    
    def chunk_text(self, text):
        # Split the text into chunks of specified size
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
    
my_embeddings = SentenceEmbeddingFunction(transformer_model_path)

# We will use cromadb as vector database:
#> pip install langchain_chroma 

from langchain_chroma import Chroma
from chromadb.config import Settings

# We will use RecursiveCharacterTextSplitter to divide the private docs into smaller text chunks:

from langchain.text_splitter import RecursiveCharacterTextSplitter

my_text_splitter = RecursiveCharacterTextSplitter()

my_text_chunks = my_text_splitter.split_documents(my_private_docs)

my_vector_database = Chroma.from_documents(documents=my_text_chunks, 
                                            embedding=my_embeddings,
                                            persist_directory=chroma_db_path, 
                                            collection_name="lc_code_collection",
                                            client_settings=Settings(anonymized_telemetry=False) )

########################################################################

# create a chain that takes the question and the retrieved documents and generates an answer

from langchain.chains.combine_documents import create_stuff_documents_chain

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

my_retriever = my_vector_database.as_retriever()

my_retrieval_chain = create_retrieval_chain(my_retriever, my_document_answering_chain)

########################################################################

# start using the retrieval chain

response = my_retrieval_chain.invoke({
    "input": "Summarize the provided context in less than 100 words"
})

print(response["answer"])

######################################################################## 