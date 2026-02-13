"""Base class for drift detection algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    detected: bool
    volatility: float  # Normalized to [0, 1]
    confidence: float  # Confidence in the detection
    details: Optional[dict] = None


class DriftDetector(ABC):
    """
    Abstract base class for drift detection algorithms.

    Drift detectors monitor a stream of observations and detect when
    the underlying distribution has changed (regime shift / concept drift).

    The volatility signal is used by VDD to modulate the decay rate:
    - Low volatility → slow decay (λ_base)
    - High volatility → fast decay (λ_max)
    """

    def __init__(self, name: str = "DriftDetector"):
        self.name = name
        self._drift_count = 0
        self._observation_count = 0

    @abstractmethod
    def update(self, value: np.ndarray | float) -> DriftResult:
        """
        Add a new observation and check for drift.

        Args:
            value: New observation (scalar or embedding vector)

        Returns:
            DriftResult with detection status and volatility score
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the detector to initial state."""
        pass

    @property
    def drift_count(self) -> int:
        """Number of drifts detected since initialization."""
        return self._drift_count

    @property
    def observation_count(self) -> int:
        """Number of observations processed."""
        return self._observation_count

    def get_volatility(self) -> float:
        """
        Get current volatility estimate.

        Returns:
            Volatility score in [0, 1]
        """
        # Subclasses should override with more sophisticated estimates
        return 0.0

    def __repr__(self) -> str:
        return f"{self.name}(observations={self._observation_count}, drifts={self._drift_count})"
