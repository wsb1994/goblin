from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class ModelConfig:
    """Base configuration class for all models"""
    model_name: str
    model_type: str  # 'basic', 'sophisticated', 'api'
    config: Dict[str, Any]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback"""
        return self.config.get(key, default)


class BaseModel(ABC):
    """Abstract base class for all hate speech detection models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_trained = False
        
    @abstractmethod
    def train(self, texts: List[str], labels: List[int]) -> None:
        """
        Train the model on text data
        
        Args:
            texts: List of text inputs
            labels: Binary labels (0 = not hate speech, 1 = hate speech)
        """
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Generate probability predictions for text inputs
        
        Args:
            texts: List of text inputs
            
        Returns:
            Array of probabilities (0-1) for hate speech detection
        """
        pass
    
    def predict_single(self, text: str) -> float:
        """Convenience method for single text prediction"""
        return float(self.predict([text])[0])
    
    @abstractmethod
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            texts: List of text inputs
            labels: Binary ground truth labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save trained model to file"""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load trained model from file"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        return {
            'name': self.config.model_name,
            'type': self.config.model_type,
            'is_trained': self.is_trained,
            'config': self.config.config
        }


class APIBaseModel(BaseModel):
    """Base class for API-based models that don't require traditional training"""
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        """API models don't require training, but may store examples for few-shot"""
        self.is_trained = True
        # Store examples for potential few-shot learning
        self._few_shot_examples = list(zip(texts[:10], labels[:10]))  # Keep 10 examples
    
    def save(self, filepath: str) -> None:
        """Save few-shot examples if any"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'few_shot_examples': getattr(self, '_few_shot_examples', [])
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load few-shot examples"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self._few_shot_examples = data.get('few_shot_examples', [])
            self.is_trained = True