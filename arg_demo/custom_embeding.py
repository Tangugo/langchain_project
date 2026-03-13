from typing import Any

from langchain_core.embeddings import Embeddings
from numpy import ndarray, dtype
from sentence_transformers import SentenceTransformer


class CustomQwen3Embedding(Embeddings):
    def __init__(self, model_path):
        self.qwen_embedding = SentenceTransformer(model_path)

    def embed_query(self, text: str) -> list[float]:
        return self.qwen_embedding.encode([text])[0]

    def embed_documents(self, texts: list[str]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        return self.qwen_embedding.encode(texts)
