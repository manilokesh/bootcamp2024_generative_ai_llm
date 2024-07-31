from utils.MyUtils import logger
from utils.MyModels import init_llm, LlmModel, BaseChatModel

########################################################################
# region Initialise foundation LLM

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# endregion Initialise foundation LLM

########################################################################

from langchain.document_loaders import TextLoader

loader = TextLoader("data/be-good-and-how-not-to-die.txt")

document = loader.load()

# The document is loaded as a Python list with metadata

print(type(document))
print(len(document))
print(document[0].metadata)
print(f"You have {len(document)} document.")
print(f"Your document has {len(document[0].page_content)} characters")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)

document_chunks = text_splitter.split_documents(document)

print(f"Now you have {len(document_chunks)} chunks.")

# Convert text chunks in numeric vectors (called "embeddings")


from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()

from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=document_chunks, embedding=my_embeddings, collection_name="qa_from_docx"
)