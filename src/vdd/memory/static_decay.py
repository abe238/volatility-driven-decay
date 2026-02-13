"""
Static Decay Memory Bank (Baseline).

Uses a fixed decay rate λ regardless of environmental volatility.
This is the baseline that VDD aims to improve upon.
"""

from typing import Any, Optional

import numpy as np

from vdd.memory.base import Memory, MemoryBank, RetrievalResult


class StaticDecayMemory(MemoryBank):
    """
    Memory bank with static (fixed) decay rate.

    All memories decay at a constant rate λ regardless of
    the environment. This serves as the primary baseline
    for comparison with VDD.

    Args:
        lambda_static: Fixed decay rate (default 0.1)
        prune_threshold: Weight threshold for pruning (default 0.01)
    """

    def __init__(
        self,
        lambda_static: float = 0.1,
        prune_threshold: float = 0.01,
    ):
        super().__init__(name="StaticDecayMemory")
        self.lambda_static = lambda_static
        self.prune_threshold = prune_threshold

        # Track for analysis
        self._lambda_history: list[float] = []

    def add(
        self,
        embedding: np.ndarray,
        content: Any,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """Add a new memory."""
        embedding = np.asarray(embedding)

        memory = Memory(
            embedding=embedding,
            content=content,
            timestamp=self._current_time,
            weight=1.0,
            metadata=metadata or {},
        )
        self._memories.append(memory)

        return memory

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve top-k memories using weighted scoring."""
        query_embedding = np.asarray(query_embedding)

        results = []
        for memory in self._memories:
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            weighted_score = similarity * memory.weight

            if weighted_score >= threshold:
                results.append(
                    RetrievalResult(
                        memory=memory,
                        score=weighted_score,
                        similarity=similarity,
                        weighted_score=weighted_score,
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def decay(self, lambda_t: Optional[float] = None) -> None:
        """Apply static decay to all memories."""
        if lambda_t is None:
            lambda_t = self.lambda_static

        self._lambda_history.append(lambda_t)

        for memory in self._memories:
            age = self._current_time - memory.timestamp
            if age > 0:
                memory.weight *= (1 - lambda_t)

        if self.prune_threshold > 0:
            self.prune(self.prune_threshold)

    def step(self) -> None:
        """Advance time and apply decay."""
        self._current_time += 1
        self.decay()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_history(self) -> dict:
        """Get lambda history for analysis."""
        return {"lambda": self._lambda_history.copy()}
