"""
Hate Speech Detection Model Library

This package provides a unified interface for training and evaluating
various hate speech detection models, from simple traditional ML approaches
to sophisticated deep learning and API-based solutions.
"""

from .base import BaseModel, APIBaseModel, ModelConfig, DataProcessor

# Import models (with optional dependencies)
try:
    from .basic import BagOfWordsModel
    BASIC_MODELS_AVAILABLE = True
except ImportError:
    BASIC_MODELS_AVAILABLE = False

try:
    from .sophisticated import HuggingFaceLightGBMModel
    SOPHISTICATED_MODELS_AVAILABLE = True
except ImportError:
    SOPHISTICATED_MODELS_AVAILABLE = False

try:
    from .api import ClaudeHateSpeechModel
    API_MODELS_AVAILABLE = True
except ImportError:
    API_MODELS_AVAILABLE = False

try:
    from .training import ModelTrainer, ModelEvaluator
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

__version__ = "0.1.0"

__all__ = ['BaseModel', 'APIBaseModel', 'ModelConfig', 'DataProcessor']

# Add available models to __all__
if BASIC_MODELS_AVAILABLE:
    __all__.append('BagOfWordsModel')
if SOPHISTICATED_MODELS_AVAILABLE:
    __all__.append('HuggingFaceLightGBMModel')
if API_MODELS_AVAILABLE:
    __all__.append('ClaudeHateSpeechModel')
if TRAINING_AVAILABLE:
    __all__.extend(['ModelTrainer', 'ModelEvaluator'])