"""
Page-Hinkley drift detector.

A sequential analysis technique for detecting changes in the
mean of a sequence. Originally designed for quality control.

Simple and fast, but requires tuning of threshold and delta parameters.
"""

import numpy as np

from vdd.drift_detection.base import DriftDetector, DriftResult


class PageHinkley(DriftDetector):
    """
    Page-Hinkley drift detector for mean shift detection.

    Monitors cumulative sum of deviations from running mean.
    Drift is detected when the cumulative sum exceeds a threshold.

    Args:
        delta: Magnitude of allowed change (default 0.005)
        threshold: Detection threshold (default 50)
        alpha: Forgetting factor for mean update (default 0.9999)
        burn_in: Observations before detection starts (default 30)
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        burn_in: int = 30,
    ):
        super().__init__(name="PageHinkley")
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.burn_in = burn_in

        # Internal state
        self._sum: float = 0.0
        self._running_mean: float = 0.0
        self._sample_count: int = 0
        self._min_sum: float = float("inf")
        self._max_sum: float = float("-inf")
        self._current_volatility: float = 0.0

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._sum = 0.0
        self._running_mean = 0.0
        self._sample_count = 0
        self._min_sum = float("inf")
        self._max_sum = float("-inf")
        self._drift_count = 0
        self._observation_count = 0
        self._current_volatility = 0.0

    def update(self, value: np.ndarray | float) -> DriftResult:
        """
        Add observation and check for drift.

        Args:
            value: Scalar value or array (will use mean if array)

        Returns:
            DriftResult with detection status
        """
        # Handle vector input
        if isinstance(value, np.ndarray):
            value = float(np.mean(value))

        self._observation_count += 1
        self._sample_count += 1

        # Update running mean with exponential smoothing
        if self._sample_count == 1:
            self._running_mean = value
        else:
            self._running_mean = self.alpha * self._running_mean + (1 - self.alpha) * value

        # Update cumulative sum
        self._sum += value - self._running_mean - self.delta

        # Track min/max
        self._min_sum = min(self._min_sum, self._sum)
        self._max_sum = max(self._max_sum, self._sum)

        # Check for drift (after burn-in)
        drift_detected = False
        ph_value = self._sum - self._min_sum

        if self._sample_count >= self.burn_in:
            if ph_value > self.threshold:
                drift_detected = True
                self._drift_count += 1
                # Reset after drift
                self._sum = 0.0
                self._min_sum = float("inf")
                self._max_sum = float("-inf")

        # Update volatility
        self._update_volatility(ph_value, drift_detected)

        return DriftResult(
            detected=drift_detected,
            volatility=self._current_volatility,
            confidence=min(1.0, ph_value / self.threshold) if self.threshold > 0 else 0.0,
            details={
                "ph_value": ph_value,
                "running_mean": self._running_mean,
                "threshold": self.threshold,
            },
        )

    def _update_volatility(self, ph_value: float, drift_detected: bool) -> None:
        """Update volatility estimate."""
        # Normalize PH value to [0, 1]
        normalized = min(1.0, ph_value / self.threshold) if self.threshold > 0 else 0.0

        # Exponential smoothing
        decay = 0.9
        self._current_volatility = decay * self._current_volatility + (1 - decay) * normalized

        # Spike on drift
        if drift_detected:
            self._current_volatility = 1.0

    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        return self._current_volatility
