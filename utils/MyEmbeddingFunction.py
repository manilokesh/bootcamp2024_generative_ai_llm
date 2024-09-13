import os

from sentence_transformers import SentenceTransformer


class SentenceEmbeddingFunction:
    def __init__(self):

        embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
        embeddings_model_path = os.getenv("EMBEDDINGS_MODEL_PATH")
        # append model name with path name to create folder in model name
        embeddings_model_path = os.path.join(
            embeddings_model_path, embeddings_model_name
        )

        # gets the projects base path
        dir_path_project = os.path.abspath(os.curdir)
        # appends the project base path with configured path
        self.transformer_model_path = os.path.abspath(
            os.path.join(dir_path_project, embeddings_model_path)
        )

        # download model locally to configured path
        if not os.path.exists(embeddings_model_path):
            # If not, download and save the model
            model = SentenceTransformer(embeddings_model_name, truncate_dim=384)
            model.save(embeddings_model_path)

        self.model = SentenceTransformer(self.transformer_model_path, truncate_dim=384)
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
        return [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]
