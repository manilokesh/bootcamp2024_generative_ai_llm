import json
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction


class CharacterTextSplitter:
    """Splitting text that looks at characters and supports chunking with overlaps."""

    def __init__(
        self,
        separator: str = "\n\n",
        is_separator_regex: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        keep_separator: Union[bool, Literal["start", "end"]] = False,
    ) -> None:
        """
        Initialize the text splitter with options for chunking and overlap.

        :param separator: The string or regex used to split text.
        :param is_separator_regex: If True, treats separator as a regex.
        :param chunk_size: The maximum size of each chunk.
        :param chunk_overlap: The number of characters overlapping between chunks.
        :param keep_separator: Whether to keep the separator in the result;
                               Can be 'start', 'end', or boolean.
        """
        self._separator = separator
        self._is_separator_regex = is_separator_regex
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._keep_separator = keep_separator

        # Ensure valid input for chunk size and overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(self, text: str) -> List[str]:
        """Split incoming text into chunks with overlap and return the result."""
        # Escape separator if it is not a regex
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )

        # Split the text using regex or the separator
        splits = self._split_text_with_regex(text, separator, self._keep_separator)

        # Merge the splits back into chunks with overlap
        merged_chunks = self._merge_splits(splits)
        return merged_chunks

    def _split_text_with_regex(
        self,
        text: str,
        separator: str,
        keep_separator: Union[bool, Literal["start", "end"]],
    ) -> List[str]:
        """Splits text by a regex or string and keeps separators optionally."""
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = (
                    (
                        [
                            _splits[i] + _splits[i + 1]
                            for i in range(0, len(_splits) - 1, 2)
                        ]
                    )
                    if keep_separator == "end"
                    else (
                        [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                    )
                )
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = (
                    (splits + [_splits[-1]])
                    if keep_separator == "end"
                    else ([_splits[0]] + splits)
                )
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge the splits into chunks with overlap."""
        chunks = []
        start = 0
        text_length = len(splits)

        # Create chunks with overlap
        while start < text_length:
            # End point is constrained by chunk_size and text length
            end = min(start + self.chunk_size, text_length)
            chunk = "".join(splits[start:end])  # Merge splits into a chunk

            # Add the chunk to the list
            chunks.append(chunk)

            # Move the starting point by chunk_size minus the overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks


class ChromaDBHandler:

    _DEFAULT_PRESIST_PATH = "./chroma_db"
    _DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        persist_directory: Optional[str] = _DEFAULT_PRESIST_PATH,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[Dict] = None,
        embedding=None,
    ):
        self.chunk_size = 500
        self.n_results_documentation = 10

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata

        # self.embedder = embedding
        self.embedder = SentenceEmbeddingFunction()

        print(self.persist_directory)
        self.db_path = self.normalize_vectordb_path(
            optional_folder=self.persist_directory
        )

        # Initialize the ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
            ),
        )

        self.documentation_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata=self.collection_metadata,
        )

    def normalize_vectordb_path(self, optional_folder: Optional[str] = None) -> str:

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

    def add_documentation(self, page_content=str, metadata=Dict) -> str:
        chunked_texts = []
        splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=50, chunk_overlap=10
        )
        chunked_texts = splitter.split_text(page_content)
        ids = [str(uuid.uuid4()) for _ in chunked_texts]

        # Use tqdm for the progress bar
        for i, text in enumerate(tqdm(chunked_texts, desc="Inserting text chunks")):
            id = str(uuid.uuid4()) + f"-{i}-doc"

            embedding = self.embedder.embed_text(text)
            self.documentation_collection.add(
                documents=text, embeddings=embedding, metadatas=metadata, ids=id
            )
            # Print the first 50 characters of the inserted text chunk
            # print(f"Inserted text chunk {i+1}: {text[:50]}...")

    def get_collection(self):
        return self.documentation_collection

    def get_related_documentation(self, question: str) -> list:
        query_vector = self.embedder.embed_text(question)
        return self._extract_documents(
            self.documentation_collection.query(
                query_embeddings=[query_vector],
                # query_texts=[question],
                n_results=self.n_results_documentation,
                include=["documents", "distances", "metadatas"],
            )
        )

    @staticmethod
    def _extract_documents(query_results) -> list:
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents
