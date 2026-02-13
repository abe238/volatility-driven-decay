"""Tests for drift detection module."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import ADWIN, PageHinkley, EmbeddingDistance


class TestADWIN:
    def test_initialization(self):
        detector = ADWIN()
        assert detector.name == "ADWIN"
        assert detector.drift_count == 0

    def test_stable_stream(self):
        detector = ADWIN(min_window=5)
        np.random.seed(42)

        for _ in range(100):
            result = detector.update(np.random.normal(0, 0.1))

        assert detector.drift_count == 0

    def test_drift_detection(self):
        detector = ADWIN(delta=0.01, min_window=10)

        # Stable period
        for _ in range(50):
            detector.update(0.0)

        # Regime shift
        for _ in range(50):
            detector.update(5.0)

        assert detector.drift_count >= 1


class TestPageHinkley:
    def test_initialization(self):
        detector = PageHinkley()
        assert detector.name == "PageHinkley"

    def test_volatility_range(self):
        detector = PageHinkley()

        for i in range(100):
            result = detector.update(float(i))
            assert 0.0 <= result.volatility <= 1.0


class TestEmbeddingDistance:
    def test_initialization(self):
        detector = EmbeddingDistance()
        assert detector.name == "EmbeddingDistance"

    def test_embedding_input(self):
        detector = EmbeddingDistance(curr_window=5, arch_window=20)

        for _ in range(30):
            emb = np.random.randn(64)
            result = detector.update(emb)

        assert 0.0 <= result.volatility <= 1.0

    def test_drift_with_orthogonal_shift(self):
        detector = EmbeddingDistance(curr_window=5, arch_window=50, drift_threshold=0.3)

        # Stable embeddings around centroid A
        centroid_a = np.random.randn(64)
        centroid_a /= np.linalg.norm(centroid_a)

        for _ in range(60):
            noise = np.random.randn(64) * 0.1
            emb = centroid_a + noise
            detector.update(emb)

        # Shift to orthogonal centroid B
        centroid_b = np.random.randn(64)
        centroid_b /= np.linalg.norm(centroid_b)

        drift_detected = False
        for _ in range(20):
            noise = np.random.randn(64) * 0.1
            emb = centroid_b + noise
            result = detector.update(emb)
            if result.detected:
                drift_detected = True

        # Should detect the shift
        assert drift_detected or detector.get_volatility() > 0.3
