"""Embedding utilities for VDD."""

from typing import Callable, Optional

import numpy as np


class DummyEmbedder:
    """
    Dummy embedder for testing without ML dependencies.

    Generates deterministic random embeddings based on text hash.
    """

    def __init__(self, dim: int = 384, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def __call__(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text."""
        # Use hash for determinism
        text_hash = hash(text) % (2**32)
        rng = np.random.RandomState(text_hash + self.seed)
        embedding = rng.randn(self.dim)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


def get_embedding_fn(
    model_name: str = "all-MiniLM-L6-v2",
    use_dummy: bool = False,
    dim: int = 384,
) -> Callable[[str], np.ndarray]:
    """
    Get an embedding function.

    Args:
        model_name: Sentence transformer model name
        use_dummy: Use dummy embedder (no ML deps required)
        dim: Embedding dimension for dummy embedder

    Returns:
        Callable that takes text and returns embedding
    """
    if use_dummy:
        return DummyEmbedder(dim=dim)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

        def embed(text: str) -> np.ndarray:
            return model.encode(text, convert_to_numpy=True)

        return embed

    except ImportError:
        print("Warning: sentence-transformers not installed. Using dummy embedder.")
        return DummyEmbedder(dim=dim)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(
    query: np.ndarray, embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple embeddings.

    Args:
        query: Query embedding (1D)
        embeddings: Matrix of embeddings (2D, each row is an embedding)

    Returns:
        Array of similarity scores
    """
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(len(embeddings))

    query_normalized = query / query_norm
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)  # Avoid division by zero
    embeddings_normalized = embeddings / norms

    return embeddings_normalized @ query_normalized
