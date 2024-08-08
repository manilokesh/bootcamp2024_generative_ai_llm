# Import package from parent folder
import os
import sys
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
import json

sys.path.append(os.path.abspath(os.curdir))

from utils.MyUtils import logger

# Section 31 [NEW] The Second Generation LangChain Key Concepts and Areas
# 192-load-data.ipynb

_DB_PRESIST_DIR = "tirukkural"
_DB_COLLECTION_NAME = "tirukkural"


def load_document_xl() -> List[Document]:
    # The UnstructuredExcelLoader is used to load Microsoft Excel files.
    # The loader works with both .xlsx and .xls files.
    # https://python.langchain.com/v0.2/docs/integrations/document_loaders/microsoft_excel/
    # %pip install --upgrade --quiet langchain-community unstructured openpyxl

    from langchain_community.document_loaders import UnstructuredExcelLoader

    folder_path = "./tirukkural/data/table"
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            logger.info(f"loading file : {file_path}")
            loader = UnstructuredExcelLoader(file_path)
            documents.extend(loader.load())

    return documents


def load_documents_pdf() -> List[Document]:

    # load a PDF using pypdf into array of documents, where each document contains the page content and metadata with page number.
    # https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/
    # %pip install --upgrade --quiet pypdf

    from langchain_community.document_loaders import PyPDFLoader

    folder_path = "./tirukkural/data/pdfs"
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            logger.info(f"loading file : {file_path}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

            # pages = loader.load_and_split()
            # print(pages[0])

    return documents


def load_documents_txt() -> List[Document]:
    from langchain_community.document_loaders import TextLoader

    folder_path = "./tirukkural/data/txts"
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            logger.info(f"loading file : {file_path}")
            loader = TextLoader(file_path, encoding="utf8")
            documents.extend(loader.load())

            # pages = loader.load_and_split()
            # print(pages[0])

    return documents


def load_document_html() -> List[Document]:

    # use unstructured to load HTML documents.
    # https://python.langchain.com/v0.2/docs/how_to/document_loader_html/
    # %pip install unstructured

    from langchain_community.document_loaders import UnstructuredHTMLLoader

    folder_path = "./tirukkural/data/htmls"
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".html"):
            file_path = os.path.join(folder_path, file)
            logger.info(f"loading file : {file_path}")
            loader = UnstructuredHTMLLoader(file_path)
            documents.extend(loader.load())

    return documents


def load_document_json() -> List[Document]:

    folder_path = "./tirukkural/data/json"
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            logger.info(f"loading file : {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                data_list = json.load(file)
                document = json_to_documents(data_list)
                documents.extend(document)
    return documents


# Function to convert JSON data to Document objects
def json_to_documents(json_data):
    documents = []

    # Ensure json_data is a dictionary
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            # Serialize value (document) as text
            doc_text = json.dumps(value, ensure_ascii=False)

            # Create Document object with serialized value as text
            document = Document(page_content=doc_text, metadata={"id": str(key)})
            documents.append(document)

    return documents


def load_document_images() -> List[Document]:

    # use unstructured to load HTML documents.
    # https://python.langchain.com/v0.2/docs/how_to/document_loader_html/
    # %pip install unstructured

    from langchain_community.document_loaders.image import UnstructuredImageLoader

    folder_path = "./tirukkural/data/images"
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        logger.info(f"loading file : {file_path}")
        loader = UnstructuredImageLoader(file_path)
        documents.extend(loader.load())

    return documents


def setup_vectordb(documents: List[Document]) -> Chroma:

    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    chunks_of_text = text_splitter.split_documents(documents)
    logger.debug(f"chunks : {len(chunks_of_text)}")

    from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

    embeddings = SentenceEmbeddingFunction()

    from utils.MyVectorStore import chroma_get, chroma_from_documents

    # chromadb: Chroma = chroma_get(
    #     embedding_function=embeddings,
    #     persist_directory=_DB_PRESIST_DIR,
    #     collection_name=_DB_COLLECTION_NAME,
    # )

    chromadb: Chroma = chroma_from_documents(
        documents=chunks_of_text,
        embedding=embeddings,
        persist_directory=_DB_PRESIST_DIR,
        collection_name=_DB_COLLECTION_NAME,
    )

    return chromadb


def main():
    documents = []

    # documents_xl = load_document_xl()
    # documents.extend(documents_xl)
    # logger.debug(f"xl docs : {len(documents_xl)}")

    # documents_pdf = load_documents_pdf()
    # documents.extend(documents_pdf)
    # logger.debug(f"pdf docs : {len(documents_pdf)}")

    # documents_txt = load_documents_txt()
    # documents.extend(documents_txt)
    # logger.debug(f"text docs : {len(documents_txt)}")

    # documents_html = load_document_html()
    # documents.extend(documents_html)
    # logger.debug(f"html docs : {len(documents_html)}")

    documents_json = load_document_json()
    documents.extend(documents_json)
    logger.debug(f"json docs : {len(documents_json)}")

    # documents_images = load_document_images()
    # documents.extend(documents_images)
    # logger.debug(f"image docs : {len(documents_images)}")

    logger.debug(f"total docs : {len(documents)}")

    setup_vectordb(documents)


if __name__ == "__main__":
    main()
