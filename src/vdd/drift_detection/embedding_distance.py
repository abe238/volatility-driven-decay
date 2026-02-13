"""
Embedding Distance drift detector for semantic drift in NLP.

Computes distributional shift by comparing centroids of recent
embeddings vs historical archive embeddings.

This is the recommended drift detector for RAG systems as it
captures semantic drift directly in embedding space.
"""

from collections import deque
from typing import Optional

import numpy as np

from vdd.drift_detection.base import DriftDetector, DriftResult


class EmbeddingDistance(DriftDetector):
    """
    Drift detector using embedding centroid distance.

    Computes volatility as:
        V_t = 1 - cos(μ_curr, μ_arch)

    Where μ_curr is the centroid of recent embeddings and μ_arch
    is the centroid of the archive.

    Args:
        curr_window: Size of recent window (default 10)
        arch_window: Size of archive window (default 1000)
        drift_threshold: Cosine distance threshold for drift (default 0.3)
        smoothing: Exponential smoothing factor for volatility (default 0.1)
    """

    def __init__(
        self,
        curr_window: int = 10,
        arch_window: int = 1000,
        drift_threshold: float = 0.3,
        smoothing: float = 0.1,
    ):
        super().__init__(name="EmbeddingDistance")
        self.curr_window = curr_window
        self.arch_window = arch_window
        self.drift_threshold = drift_threshold
        self.smoothing = smoothing

        # Sliding windows for embeddings
        self._curr_buffer: deque = deque(maxlen=curr_window)
        self._arch_buffer: deque = deque(maxlen=arch_window)

        # State
        self._current_volatility: float = 0.0
        self._embedding_dim: Optional[int] = None
        self._last_distance: float = 0.0

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._curr_buffer.clear()
        self._arch_buffer.clear()
        self._current_volatility = 0.0
        self._drift_count = 0
        self._observation_count = 0
        self._embedding_dim = None
        self._last_distance = 0.0

    def update(self, value: np.ndarray | float) -> DriftResult:
        """
        Add embedding and check for semantic drift.

        Args:
            value: Embedding vector (1D numpy array)

        Returns:
            DriftResult with detection status
        """
        # Convert to numpy array if needed
        if isinstance(value, (int, float)):
            value = np.array([value])
        value = np.asarray(value).flatten()

        self._observation_count += 1

        # Initialize embedding dimension
        if self._embedding_dim is None:
            self._embedding_dim = len(value)

        # Add to buffers
        self._curr_buffer.append(value)
        self._arch_buffer.append(value)

        # Need enough data to compare
        if len(self._curr_buffer) < self.curr_window // 2:
            return DriftResult(
                detected=False,
                volatility=0.0,
                confidence=0.0,
                details={"status": "warming_up"},
            )

        if len(self._arch_buffer) < self.curr_window * 2:
            return DriftResult(
                detected=False,
                volatility=0.0,
                confidence=0.0,
                details={"status": "building_archive"},
            )

        # Compute centroids
        curr_centroid = self._compute_centroid(list(self._curr_buffer))

        # Archive centroid excludes current window for fair comparison
        arch_list = list(self._arch_buffer)
        arch_centroid = self._compute_centroid(arch_list[: -self.curr_window])

        # Compute cosine distance
        distance = self._cosine_distance(curr_centroid, arch_centroid)
        self._last_distance = distance

        # Check for drift
        drift_detected = distance > self.drift_threshold

        if drift_detected:
            self._drift_count += 1

        # Update volatility with exponential smoothing
        # Higher distance = higher volatility
        raw_volatility = min(1.0, distance / self.drift_threshold)
        self._current_volatility = (
            self.smoothing * raw_volatility
            + (1 - self.smoothing) * self._current_volatility
        )

        # Spike on drift detection
        if drift_detected:
            self._current_volatility = max(self._current_volatility, 0.8)

        return DriftResult(
            detected=drift_detected,
            volatility=self._current_volatility,
            confidence=min(1.0, distance / self.drift_threshold),
            details={
                "cosine_distance": distance,
                "curr_window_size": len(self._curr_buffer),
                "arch_window_size": len(self._arch_buffer),
            },
        )

    def _compute_centroid(self, embeddings: list) -> np.ndarray:
        """Compute centroid (mean) of embeddings."""
        if not embeddings:
            return np.zeros(self._embedding_dim or 1)
        return np.mean(embeddings, axis=0)

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine distance between two vectors.

        Returns:
            Distance in [0, 2], where 0 = identical, 1 = orthogonal, 2 = opposite
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        cos_sim = np.dot(a, b) / (norm_a * norm_b)
        # Clamp to [-1, 1] for numerical stability
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        return 1.0 - cos_sim

    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        return self._current_volatility

    @property
    def last_distance(self) -> float:
        """Last computed cosine distance."""
        return self._last_distance
