"""
Simplified ADWIN (ADaptive WINdowing) implementation.

ADWIN detects change by comparing statistics of two sub-windows.
When the means differ significantly, it signals drift.
"""

import numpy as np
from collections import deque
from vdd.drift_detection.base import DriftDetector, DriftResult


class SimpleADWIN(DriftDetector):
    """
    Simplified ADWIN change detector.

    Maintains a sliding window and detects change when
    the means of two sub-windows differ significantly.

    Args:
        delta: Confidence parameter (lower = more sensitive)
        min_window: Minimum window size before checking
        max_window: Maximum window size
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window: int = 10,
        max_window: int = 200,
    ):
        super().__init__(name="SimpleADWIN")
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window

        self._window: deque = deque(maxlen=max_window)
        self._drift_detected = False
        self._volatility = 0.0
        self._prev_embedding = None

    def update(self, data) -> DriftResult:
        """
        Update with new observation.

        For embeddings, we track cosine distance from previous embedding.
        """
        # Convert embedding to scalar statistic
        if hasattr(data, '__len__') and len(data) > 1:
            # It's an embedding - use cosine distance from previous
            data = np.asarray(data)
            if self._prev_embedding is not None:
                # Cosine distance
                norm_a = np.linalg.norm(data)
                norm_b = np.linalg.norm(self._prev_embedding)
                if norm_a > 1e-10 and norm_b > 1e-10:
                    cos_sim = np.dot(data, self._prev_embedding) / (norm_a * norm_b)
                    stat = 1.0 - cos_sim  # Distance = 1 - similarity
                else:
                    stat = 0.0
            else:
                stat = 0.0
            self._prev_embedding = data.copy()
        else:
            stat = float(data)

        self._window.append(stat)
        self._drift_detected = False

        if len(self._window) >= self.min_window:
            self._drift_detected = self._check_drift()

        if self._drift_detected:
            self._drift_count += 1
            # Reset window on drift
            self._window.clear()
            self._volatility = 1.0
        else:
            self._volatility = max(0, self._volatility - 0.05)

        return DriftResult(
            detected=self._drift_detected,
            volatility=self._volatility,
            confidence=1.0 if self._drift_detected else 0.0,
            details={"statistic": stat, "detector_name": self.name},
        )

    def _check_drift(self) -> bool:
        """Check if drift occurred using ADWIN criterion."""
        n = len(self._window)
        if n < 2 * self.min_window:
            return False

        window_array = np.array(self._window)

        # Try different split points
        for split in range(self.min_window, n - self.min_window + 1):
            w1 = window_array[:split]
            w2 = window_array[split:]

            # Compare means
            mu1, mu2 = np.mean(w1), np.mean(w2)
            n1, n2 = len(w1), len(w2)

            # Simplified threshold based on variance
            combined_var = np.var(window_array)
            if combined_var < 1e-10:
                continue

            # Normalized difference
            diff = abs(mu1 - mu2) / np.sqrt(combined_var)

            # Use delta as threshold multiplier (higher delta = more sensitive)
            if diff > (1.0 / self.delta):
                return True

        return False

    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        return self._volatility

    def reset(self) -> None:
        """Reset detector state."""
        self._window.clear()
        self._drift_detected = False
        self._volatility = 0.0
        self._drift_count = 0
        self._prev_embedding = None
