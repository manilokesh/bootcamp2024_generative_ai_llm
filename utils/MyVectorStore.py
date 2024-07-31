import os
from typing import (
    List,
    Optional,
)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.config import Settings

def chroma_from_documents(
    documents: List[Document],
    embedding: Embeddings,
    persist_directory: Optional[str] = None,
    collection_name: str = "langchain",
) -> Chroma:

    # Check for None in persist_directory and assign chroma_db_path if not provided
    if persist_directory is None:
        persist_directory = os.getenv("CHROMA_DB_PATH")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
        client_settings=Settings(anonymized_telemetry=False),
    )
    return vectorstore
 

def chroma_get(
    embedding_function: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = "langchain",
) -> Chroma:

    # Check for None in persist_directory and assign chroma_db_path if not provided
    if persist_directory is None:
        persist_directory = os.getenv("CHROMA_DB_PATH")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function,
        client_settings=Settings(anonymized_telemetry=False),
    )
    return vectorstore
