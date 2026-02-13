"""
ADWIN (Adaptive Windowing) drift detector.

Based on: Bifet & Gavaldà, "Learning from Time-Changing Data with
Adaptive Windowing" (2007)

ADWIN maintains a variable-length window of recent observations.
When statistical tests suggest the mean has changed, the window
shrinks by dropping old observations.
"""

import math
from collections import deque
from typing import Optional

import numpy as np

from vdd.drift_detection.base import DriftDetector, DriftResult


class ADWIN(DriftDetector):
    """
    ADWIN drift detector using adaptive windowing.

    The algorithm maintains a sliding window W and checks if any two
    sub-windows have "sufficiently different" means. If so, old data
    is dropped and drift is signaled.

    Args:
        delta: Confidence parameter (default 0.002). Lower = more sensitive.
        max_buckets: Maximum number of buckets per row (memory limit).
        min_window: Minimum observations before drift detection starts.
        min_sub_window: Minimum sub-window size for comparison.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_buckets: int = 5,
        min_window: int = 10,
        min_sub_window: int = 5,
    ):
        super().__init__(name="ADWIN")
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window = min_window
        self.min_sub_window = min_sub_window

        # Internal state
        self._window: deque = deque()
        self._total: float = 0.0
        self._variance: float = 0.0
        self._width: int = 0
        self._last_drift_width: int = 0
        self._current_volatility: float = 0.0

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._window.clear()
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._drift_count = 0
        self._observation_count = 0
        self._last_drift_width = 0
        self._current_volatility = 0.0

    def update(self, value: np.ndarray | float) -> DriftResult:
        """
        Add observation and check for drift.

        Args:
            value: Scalar value or 1D array (will use mean if array)

        Returns:
            DriftResult with detection status
        """
        # Handle vector input
        if isinstance(value, np.ndarray):
            value = float(np.mean(value))

        self._observation_count += 1
        self._add_element(value)

        # Check for drift
        drift_detected = False
        if self._width >= self.min_window:
            drift_detected = self._detect_drift()

        # Update volatility estimate
        self._update_volatility(drift_detected)

        return DriftResult(
            detected=drift_detected,
            volatility=self._current_volatility,
            confidence=1.0 - self.delta if drift_detected else self.delta,
            details={
                "window_size": self._width,
                "mean": self._total / self._width if self._width > 0 else 0,
            },
        )

    def _add_element(self, value: float) -> None:
        """Add element to the window."""
        self._window.append(value)
        self._total += value
        self._width += 1

        # Update variance using Welford's online algorithm
        if self._width > 1:
            mean = self._total / self._width
            self._variance += (value - mean) ** 2

    def _detect_drift(self) -> bool:
        """
        Check if drift occurred by comparing sub-windows.

        Returns:
            True if drift detected
        """
        if self._width < 2 * self.min_sub_window:
            return False

        found_drift = False
        window_list = list(self._window)

        # Try different split points
        for split in range(self.min_sub_window, self._width - self.min_sub_window):
            w0 = window_list[:split]
            w1 = window_list[split:]

            n0, n1 = len(w0), len(w1)
            if n0 < self.min_sub_window or n1 < self.min_sub_window:
                continue

            mean0 = sum(w0) / n0
            mean1 = sum(w1) / n1

            # Compute epsilon bound (Hoeffding bound)
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            delta_prime = self.delta / math.log(self._width)
            epsilon = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / delta_prime))

            if abs(mean0 - mean1) > epsilon:
                # Drift detected - shrink window
                self._window = deque(w1)
                self._total = sum(w1)
                self._width = n1
                self._last_drift_width = split
                self._drift_count += 1
                found_drift = True
                break

        return found_drift

    def _update_volatility(self, drift_detected: bool) -> None:
        """Update volatility estimate based on detection."""
        # Exponential decay of volatility
        decay = 0.95

        if drift_detected:
            # Spike volatility on drift
            self._current_volatility = 1.0
        else:
            # Decay towards baseline
            self._current_volatility *= decay

        # Clamp to [0, 1]
        self._current_volatility = max(0.0, min(1.0, self._current_volatility))

    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        return self._current_volatility

    @property
    def window_size(self) -> int:
        """Current window size."""
        return self._width

    @property
    def mean(self) -> float:
        """Current window mean."""
        return self._total / self._width if self._width > 0 else 0.0
