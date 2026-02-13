"""
VDD: Volatility-Driven Decay for RAG Memory Systems

A control-theory inspired approach where memory decay rate dynamically
adjusts based on environmental volatility.
"""

__version__ = "0.1.0"

from vdd.drift_detection import DriftDetector, ADWIN, EmbeddingDistance
from vdd.memory import MemoryBank, VDDMemoryBank, StaticDecayMemory
from vdd.retrieval import VDDRetriever

__all__ = [
    "DriftDetector",
    "ADWIN",
    "EmbeddingDistance",
    "MemoryBank",
    "VDDMemoryBank",
    "StaticDecayMemory",
    "VDDRetriever",
]
