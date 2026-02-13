"""Base classes for memory banks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Memory:
    """
    A single memory unit in the memory bank.

    Attributes:
        embedding: Vector representation of the content
        content: The actual content (text, metadata, etc.)
        timestamp: When the memory was created
        weight: Current decay weight (1.0 = fresh, 0.0 = forgotten)
        metadata: Additional metadata
    """

    embedding: np.ndarray
    content: Any
    timestamp: int
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.embedding = np.asarray(self.embedding)


@dataclass
class RetrievalResult:
    """Result of a memory retrieval."""

    memory: Memory
    score: float  # Combined similarity + weight score
    similarity: float  # Raw similarity score
    weighted_score: float  # Score after weight application


class MemoryBank(ABC):
    """
    Abstract base class for memory banks.

    Memory banks store and retrieve memories with decay mechanisms.
    Different implementations use different decay strategies.
    """

    def __init__(self, name: str = "MemoryBank"):
        self.name = name
        self._memories: list[Memory] = []
        self._current_time: int = 0

    @abstractmethod
    def add(
        self,
        embedding: np.ndarray,
        content: Any,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """
        Add a new memory to the bank.

        Args:
            embedding: Vector embedding of the content
            content: The actual content
            metadata: Optional metadata

        Returns:
            The created Memory object
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k memories most relevant to query.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Minimum score threshold

        Returns:
            List of RetrievalResult sorted by score (descending)
        """
        pass

    @abstractmethod
    def decay(self, lambda_t: float) -> None:
        """
        Apply decay to all memories.

        Args:
            lambda_t: Decay rate for this timestep
        """
        pass

    def step(self) -> None:
        """Advance time by one step."""
        self._current_time += 1

    def prune(self, threshold: float = 0.01) -> int:
        """
        Remove memories with weight below threshold.

        Args:
            threshold: Weight threshold for pruning

        Returns:
            Number of memories pruned
        """
        original_count = len(self._memories)
        self._memories = [m for m in self._memories if m.weight >= threshold]
        return original_count - len(self._memories)

    @property
    def size(self) -> int:
        """Number of memories in the bank."""
        return len(self._memories)

    @property
    def current_time(self) -> int:
        """Current logical timestamp."""
        return self._current_time

    def get_stats(self) -> dict:
        """Get statistics about the memory bank."""
        if not self._memories:
            return {
                "size": 0,
                "mean_weight": 0.0,
                "min_weight": 0.0,
                "max_weight": 0.0,
            }

        weights = [m.weight for m in self._memories]
        return {
            "size": len(self._memories),
            "mean_weight": np.mean(weights),
            "min_weight": np.min(weights),
            "max_weight": np.max(weights),
            "current_time": self._current_time,
        }

    def __len__(self) -> int:
        return len(self._memories)

    def __repr__(self) -> str:
        return f"{self.name}(size={self.size}, time={self._current_time})"
