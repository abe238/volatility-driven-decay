"""
Ollama Embeddings Utility for VDD Real-World RAG Experiments.

Uses local Ollama models for embedding generation.
"""

import numpy as np
import requests
from typing import List, Optional
import json


class OllamaEmbeddings:
    """Generate embeddings using local Ollama models."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._dim = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension (lazy loaded)."""
        if self._dim is None:
            test_emb = self.embed("test")
            self._dim = len(test_emb)
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name", "").startswith(self.model) for m in models)
            return False
        except Exception:
            return False


class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file
        self._cache = {}
        if cache_file:
            self._load()

    def _load(self):
        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                self._cache = {k: np.array(v) for k, v in data.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            self._cache = {}

    def _save(self):
        if self.cache_file:
            data = {k: v.tolist() for k, v in self._cache.items()}
            with open(self.cache_file, "w") as f:
                json.dump(data, f)

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._cache.get(key)

    def set(self, key: str, embedding: np.ndarray):
        self._cache[key] = embedding
        self._save()

    def get_or_compute(
        self, key: str, text: str, embedder: OllamaEmbeddings
    ) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        emb = embedder.embed(text)
        self.set(key, emb)
        return emb


if __name__ == "__main__":
    embedder = OllamaEmbeddings()
    print(f"Ollama available: {embedder.is_available()}")
    if embedder.is_available():
        print(f"Model: {embedder.model}")
        print(f"Dimension: {embedder.dimension}")
        test = embedder.embed("Hello world")
        print(f"Test embedding shape: {test.shape}")
