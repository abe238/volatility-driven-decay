"""
Prediction-Error Drift Detector (Option A).

Detects drift by measuring prediction error: when the agent's predictions
about incoming data fail, this indicates knowledge staleness.

This differs from semantic distance (Option B) which detects topic changes
but may trigger on natural topic switches even when knowledge remains valid.
"""

import numpy as np
from collections import deque
from vdd.drift_detection.base import DriftDetector, DriftResult


class PredictionErrorDetector(DriftDetector):
    """
    Drift detection based on query prediction error.

    Instead of detecting topic shifts, this detector tracks how well
    we can predict the next query from recent history. High prediction
    error suggests the environment is changing unpredictably.

    Args:
        window: Size of recent query window
        threshold: Volatility threshold for drift detection
        alpha: EMA smoothing factor for prediction error
    """

    def __init__(
        self,
        window: int = 10,
        threshold: float = 0.5,
        alpha: float = 0.2,
    ):
        super().__init__(name="PredictionError")
        self.window = window
        self.threshold = threshold
        self.alpha = alpha

        self._query_history: deque = deque(maxlen=window)
        self._prediction_errors: deque = deque(maxlen=window)
        self._smoothed_error = 0.0
        self._predictor = SimpleLinearPredictor()

    def update(self, embedding: np.ndarray) -> DriftResult:
        """Update with new query embedding and check for drift."""
        embedding = np.asarray(embedding).flatten()

        # Compute prediction error if we have history
        error = 0.0
        if len(self._query_history) >= 2:
            # Predict current embedding from recent history
            predicted = self._predictor.predict(list(self._query_history))

            if predicted is not None:
                # Compute cosine distance as error
                error = self._cosine_distance(predicted, embedding)

            # Update predictor with new observation
            self._predictor.update(list(self._query_history), embedding)

        self._prediction_errors.append(error)
        self._query_history.append(embedding.copy())

        # Smooth the error with EMA
        self._smoothed_error = (
            self.alpha * error + (1 - self.alpha) * self._smoothed_error
        )

        # Compute volatility as normalized smoothed error
        volatility = min(1.0, self._smoothed_error / 0.5)  # Normalize to [0,1]

        # Detect drift
        detected = volatility > self.threshold

        if detected:
            self._drift_count += 1

        return DriftResult(
            detected=detected,
            volatility=volatility,
            confidence=volatility,
            details={
                "raw_error": error,
                "smoothed_error": self._smoothed_error,
            },
        )

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance (1 - similarity)."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0  # Max distance if one is zero

        similarity = np.dot(a, b) / (norm_a * norm_b)
        return 1.0 - similarity

    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        return min(1.0, self._smoothed_error / 0.5)

    def reset(self) -> None:
        """Reset detector state."""
        self._query_history.clear()
        self._prediction_errors.clear()
        self._smoothed_error = 0.0
        self._drift_count = 0
        self._predictor = SimpleLinearPredictor()


class SimpleLinearPredictor:
    """
    Simple linear predictor for embeddings.

    Predicts next embedding as weighted average of recent embeddings,
    with exponential weighting favoring more recent observations.
    """

    def __init__(self, decay: float = 0.8):
        self.decay = decay
        self._weights = None

    def predict(self, history: list) -> np.ndarray:
        """
        Predict next embedding from history.

        Args:
            history: List of recent embeddings

        Returns:
            Predicted embedding or None if insufficient history
        """
        if len(history) < 2:
            return None

        history = [np.asarray(h) for h in history]

        # Exponential weights favoring recent
        n = len(history)
        weights = np.array([self.decay ** (n - i - 1) for i in range(n)])
        weights = weights / weights.sum()

        # Weighted average
        predicted = np.zeros_like(history[0])
        for w, h in zip(weights, history):
            predicted += w * h

        return predicted

    def update(self, history: list, actual: np.ndarray) -> None:
        """
        Update predictor based on actual observation.

        In this simple version, we don't do explicit updates -
        the prediction is always based on recent history.
        A more sophisticated version could learn the dynamics.
        """
        pass  # Simple predictor doesn't need online updates


class AdaptivePredictor:
    """
    More sophisticated predictor that learns embedding dynamics.

    Uses simple momentum-based prediction:
    predicted = last + velocity, where velocity = last - prev
    """

    def __init__(self, momentum: float = 0.5):
        self.momentum = momentum
        self._velocity = None

    def predict(self, history: list) -> np.ndarray:
        """Predict using momentum."""
        if len(history) < 2:
            return None

        history = [np.asarray(h) for h in history]
        last = history[-1]
        prev = history[-2]

        # Current velocity
        current_velocity = last - prev

        # Blend with historical velocity if available
        if self._velocity is not None:
            blended_velocity = (
                self.momentum * self._velocity +
                (1 - self.momentum) * current_velocity
            )
        else:
            blended_velocity = current_velocity

        # Predict: last + velocity
        return last + blended_velocity

    def update(self, history: list, actual: np.ndarray) -> None:
        """Update velocity estimate."""
        if len(history) >= 2:
            history = [np.asarray(h) for h in history]
            self._velocity = actual - history[-1]
