"""Evaluation metrics for VDD experiments."""

from typing import Optional

import numpy as np


def integrated_absolute_error(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute Integrated Absolute Error (IAE).

    IAE = Σ |prediction - truth|

    Lower is better.

    Args:
        predictions: Predicted values
        ground_truth: True values

    Returns:
        IAE score
    """
    return float(np.sum(np.abs(predictions - ground_truth)))


def mean_absolute_error(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Args:
        predictions: Predicted values
        ground_truth: True values

    Returns:
        MAE score
    """
    return float(np.mean(np.abs(predictions - ground_truth)))


def compute_retrieval_precision(
    retrieved_ids: list,
    relevant_ids: set,
    k: Optional[int] = None,
) -> float:
    """
    Compute precision@k for retrieval.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider (default: all)

    Returns:
        Precision score in [0, 1]
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:
        return 0.0

    relevant_count = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
    return relevant_count / len(retrieved_ids)


def compute_retrieval_recall(
    retrieved_ids: list,
    relevant_ids: set,
    k: Optional[int] = None,
) -> float:
    """
    Compute recall@k for retrieval.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider (default: all)

    Returns:
        Recall score in [0, 1]
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not relevant_ids:
        return 0.0

    relevant_count = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
    return relevant_count / len(relevant_ids)


def compute_temporal_accuracy(
    retrieved_timestamps: list[int],
    query_timestamp: int,
    max_age: int,
) -> float:
    """
    Compute temporal accuracy of retrieval.

    Measures how well the retrieved documents match the current
    time context. Documents older than max_age are considered stale.

    Args:
        retrieved_timestamps: Timestamps of retrieved documents
        query_timestamp: Current query timestamp
        max_age: Maximum acceptable age

    Returns:
        Temporal accuracy in [0, 1]
    """
    if not retrieved_timestamps:
        return 0.0

    fresh_count = 0
    for ts in retrieved_timestamps:
        age = query_timestamp - ts
        if 0 <= age <= max_age:
            fresh_count += 1

    return fresh_count / len(retrieved_timestamps)


def compute_staleness_ratio(
    retrieved_timestamps: list[int],
    query_timestamp: int,
    regime_change_time: int,
) -> float:
    """
    Compute ratio of stale (pre-regime-change) documents retrieved.

    Used to measure how well the system forgets outdated information
    after a regime change.

    Args:
        retrieved_timestamps: Timestamps of retrieved documents
        query_timestamp: Current query timestamp
        regime_change_time: When the regime changed

    Returns:
        Staleness ratio in [0, 1] (lower is better)
    """
    if not retrieved_timestamps:
        return 0.0

    stale_count = sum(1 for ts in retrieved_timestamps if ts < regime_change_time)
    return stale_count / len(retrieved_timestamps)


def compute_adaptation_time(
    errors: np.ndarray,
    regime_change_idx: int,
    threshold: float = 0.1,
) -> int:
    """
    Compute how many steps until error drops below threshold after regime change.

    Args:
        errors: Array of error values
        regime_change_idx: Index where regime changed
        threshold: Error threshold for "adapted"

    Returns:
        Number of steps to adapt (-1 if never adapts)
    """
    post_change_errors = errors[regime_change_idx:]

    for i, error in enumerate(post_change_errors):
        if error < threshold:
            return i

    return -1  # Never adapted
