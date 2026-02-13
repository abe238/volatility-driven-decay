"""
VDD Memory Bank with volatility-driven decay.

The core implementation of Volatility-Driven Decay for RAG systems.
Decay rate λ(t) is dynamically adjusted based on detected volatility.
"""

from typing import Any, Optional

import numpy as np

from vdd.drift_detection.base import DriftDetector
from vdd.memory.base import Memory, MemoryBank, RetrievalResult


class VDDMemoryBank(MemoryBank):
    """
    Memory bank with Volatility-Driven Decay.

    The decay rate λ(t) is computed as:
        λ(t) = λ_base + (λ_max - λ_base) * σ(k * (V_t - V_0))

    Where:
        - λ_base: Resting decay rate (slow decay in stable periods)
        - λ_max: Maximum decay rate (fast decay during regime changes)
        - V_t: Current volatility from drift detector
        - k: Sigmoid steepness (hysteresis factor)
        - V_0: Volatility threshold

    Args:
        drift_detector: DriftDetector instance for volatility estimation
        lambda_base: Resting decay rate (default 0.05)
        lambda_max: Maximum decay rate (default 0.9)
        k: Sigmoid steepness (default 10.0)
        v_threshold: Volatility threshold V_0 (default 0.1)
        prune_threshold: Weight threshold for automatic pruning (default 0.01)
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        lambda_base: float = 0.05,
        lambda_max: float = 0.9,
        k: float = 10.0,
        v_threshold: float = 0.1,
        prune_threshold: float = 0.01,
    ):
        super().__init__(name="VDDMemoryBank")
        self.drift_detector = drift_detector
        self.lambda_base = lambda_base
        self.lambda_max = lambda_max
        self.k = k
        self.v_threshold = v_threshold
        self.prune_threshold = prune_threshold

        # Track λ history for analysis
        self._lambda_history: list[float] = []
        self._volatility_history: list[float] = []

    def add(
        self,
        embedding: np.ndarray,
        content: Any,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """Add a new memory and update drift detector."""
        embedding = np.asarray(embedding)

        # Update drift detector with new embedding
        self.drift_detector.update(embedding)

        # Create memory
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
        """
        Retrieve top-k memories using VDD-weighted scoring.

        Score = similarity * weight

        Where weight has been decayed based on volatility history.
        """
        query_embedding = np.asarray(query_embedding)

        # Also update drift detector with query
        self.drift_detector.update(query_embedding)

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

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def decay(self, lambda_t: Optional[float] = None) -> None:
        """
        Apply decay to all memories.

        If lambda_t is not provided, compute it from current volatility.
        """
        if lambda_t is None:
            lambda_t = self._compute_lambda()

        # Record for analysis
        self._lambda_history.append(lambda_t)
        self._volatility_history.append(self.drift_detector.get_volatility())

        # Apply exponential decay
        for memory in self._memories:
            # Decay based on age
            age = self._current_time - memory.timestamp
            if age > 0:
                memory.weight *= (1 - lambda_t)

        # Auto-prune if enabled
        if self.prune_threshold > 0:
            self.prune(self.prune_threshold)

    def step(self) -> None:
        """Advance time and apply decay."""
        self._current_time += 1
        self.decay()

    def _compute_lambda(self) -> float:
        """
        Compute current decay rate from volatility.

        Uses sigmoid activation:
            λ(t) = λ_base + (λ_max - λ_base) * sigmoid(k * (V_t - V_0))
        """
        v_t = self.drift_detector.get_volatility()

        # Sigmoid activation
        x = self.k * (v_t - self.v_threshold)
        sigmoid = 1.0 / (1.0 + np.exp(-x))

        # Interpolate between base and max
        lambda_t = self.lambda_base + (self.lambda_max - self.lambda_base) * sigmoid

        return lambda_t

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_current_lambda(self) -> float:
        """Get current computed decay rate."""
        return self._compute_lambda()

    def get_history(self) -> dict:
        """Get lambda and volatility history for analysis."""
        return {
            "lambda": self._lambda_history.copy(),
            "volatility": self._volatility_history.copy(),
        }

    def get_stats(self) -> dict:
        """Get extended statistics."""
        stats = super().get_stats()
        stats.update(
            {
                "current_lambda": self._compute_lambda(),
                "current_volatility": self.drift_detector.get_volatility(),
                "drift_count": self.drift_detector.drift_count,
                "lambda_base": self.lambda_base,
                "lambda_max": self.lambda_max,
            }
        )
        return stats
