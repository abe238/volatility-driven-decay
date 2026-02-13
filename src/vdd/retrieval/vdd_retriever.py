"""
VDD Retriever for integration with RAG systems.

Provides a high-level interface for adding documents and
retrieving with volatility-driven decay.
"""

from typing import Any, Callable, Optional

import numpy as np

from vdd.drift_detection.base import DriftDetector
from vdd.drift_detection.embedding_distance import EmbeddingDistance
from vdd.memory.vdd_memory import VDDMemoryBank
from vdd.memory.base import RetrievalResult


class VDDRetriever:
    """
    High-level retriever with VDD memory management.

    Combines embedding function, drift detection, and VDD memory
    into a simple interface for RAG applications.

    Args:
        embed_fn: Function to embed text -> numpy array
        drift_detector: DriftDetector instance (default: EmbeddingDistance)
        lambda_base: Resting decay rate
        lambda_max: Maximum decay rate
        k: Sigmoid steepness
        v_threshold: Volatility threshold
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        drift_detector: Optional[DriftDetector] = None,
        lambda_base: float = 0.05,
        lambda_max: float = 0.9,
        k: float = 10.0,
        v_threshold: float = 0.1,
    ):
        self.embed_fn = embed_fn

        # Default to embedding distance detector
        if drift_detector is None:
            drift_detector = EmbeddingDistance()

        self.memory_bank = VDDMemoryBank(
            drift_detector=drift_detector,
            lambda_base=lambda_base,
            lambda_max=lambda_max,
            k=k,
            v_threshold=v_threshold,
        )

    def add_document(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a document to the memory bank.

        Args:
            text: Document text
            metadata: Optional metadata
        """
        embedding = self.embed_fn(text)
        self.memory_bank.add(
            embedding=embedding,
            content=text,
            metadata=metadata or {},
        )

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Add multiple documents."""
        metadatas = metadatas or [{}] * len(texts)
        for text, meta in zip(texts, metadatas):
            self.add_document(text, meta)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            k: Number of results
            threshold: Minimum score threshold

        Returns:
            List of dicts with 'content', 'score', 'metadata'
        """
        query_embedding = self.embed_fn(query)
        results = self.memory_bank.retrieve(
            query_embedding=query_embedding,
            k=k,
            threshold=threshold,
        )

        return [
            {
                "content": r.memory.content,
                "score": r.score,
                "similarity": r.similarity,
                "weight": r.memory.weight,
                "metadata": r.memory.metadata,
            }
            for r in results
        ]

    def step(self) -> None:
        """Advance time and apply decay."""
        self.memory_bank.step()

    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return self.memory_bank.get_stats()

    def get_volatility(self) -> float:
        """Get current volatility."""
        return self.memory_bank.drift_detector.get_volatility()

    def get_lambda(self) -> float:
        """Get current decay rate."""
        return self.memory_bank.get_current_lambda()
