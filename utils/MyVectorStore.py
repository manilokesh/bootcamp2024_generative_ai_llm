import os
from pathlib import Path
from typing import (
    List,
    Optional,
)

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def normalize_vectordb_path(optional_folder: Optional[str] = None) -> str:

    configured_directory = os.getenv("CHROMA_DB_PATH")
    final_directory = configured_directory

    # Check for None in persist_directory and assign chroma_db_path if not provided
    if optional_folder is not None:
        # final_directory = os.path.join(os.path.abspath(os.curdir), configured_directory, optional_folder)
        final_directory = os.path.join(configured_directory, optional_folder)

    # Create the directory if it does not exist
    final_path = Path(final_directory)
    # final_path.mkdir(parents=True, exist_ok=True)

    print(f"Vector DB persist folder: '{final_path}'")

    return str(final_path)


def chroma_from_documents(
    documents: List[Document],
    embedding: Embeddings,
    persist_directory: Optional[str] = None,
    collection_name: str = "langchain",
) -> Chroma:

    persist_directory = normalize_vectordb_path(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
        ),
    )
    return vectorstore


def chroma_get(
    embedding_function: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = "langchain",
) -> Chroma:

    persist_directory = normalize_vectordb_path(persist_directory)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
        ),
    )
    return vectorstore
