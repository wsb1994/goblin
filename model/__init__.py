"""
Hate Speech Detection Model Library

This package provides a unified interface for training and evaluating
various hate speech detection models, from simple traditional ML approaches
to sophisticated deep learning and API-based solutions.
"""

from .base import BaseModel, APIBaseModel, ModelConfig, DataProcessor

__version__ = "0.1.0"
__all__ = ['BaseModel', 'APIBaseModel', 'ModelConfig', 'DataProcessor']