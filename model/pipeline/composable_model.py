from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import json
import os
from pathlib import Path

from ..base import BaseModel, ModelConfig
from ..components import get_dataset_provider, get_feature_extractor, get_model_architecture


class ComposableModel(BaseModel):
    """
    Composable model that combines dataset, feature extractor, and model architecture
    
    This allows mixing and matching components:
    - Berkeley Dataset + HuggingFace Embeddings + Logistic Regression
    - CSV Dataset + TF-IDF + LightGBM
    - Synthetic Dataset + Statistical Features + Random Forest
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Extract component configurations
        self.dataset_config = config.get('dataset', {})
        self.feature_config = config.get('feature_extractor', {})
        self.architecture_config = config.get('model_architecture', {})
        
        # Component types
        self.dataset_type = self.dataset_config.get('type', 'berkeley')
        self.feature_type = self.feature_config.get('type', 'tfidf')
        self.architecture_type = self.architecture_config.get('type', 'logistic')
        
        # Initialize components
        self.dataset_provider = None
        self.feature_extractor = None 
        self.model_architecture = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            # Initialize dataset provider
            self.dataset_provider = get_dataset_provider(
                self.dataset_type, 
                self.dataset_config
            )
            
            # Initialize feature extractor
            self.feature_extractor = get_feature_extractor(
                self.feature_type,
                self.feature_config
            )
            
            # Initialize model architecture
            self.model_architecture = get_model_architecture(
                self.architecture_type,
                self.architecture_config
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {e}")
    
    def train(self, texts: List[str] = None, labels: List[int] = None) -> None:
        """
        Train the composable model
        
        Args:
            texts: Optional texts (if None, will load from dataset provider)
            labels: Optional labels (if None, will load from dataset provider)
        """
        # Load data if not provided
        if texts is None or labels is None:
            print(f"Loading data from {self.dataset_type} dataset...")
            texts, labels = self.dataset_provider.load()
        
        print(f"Training on {len(texts)} samples...")
        
        # Extract features
        print(f"Extracting features using {self.feature_type}...")
        X = self.feature_extractor.fit_transform(texts)
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Train model architecture
        print(f"Training {self.architecture_type} architecture...")
        self.model_architecture.fit(X, np.array(labels))
        
        self.is_trained = True
        print("Training completed!")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Generate probability predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Extract features
        X = self.feature_extractor.transform(texts)
        
        # Get predictions
        probabilities = self.model_architecture.predict_proba(X)
        
        return probabilities
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Extract features
        X = self.feature_extractor.transform(texts)
        
        # Get evaluation metrics
        return self.model_architecture.evaluate(X, np.array(labels))
    
    def save(self, filepath: str) -> None:
        """Save the entire composable model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        save_dir = Path(filepath).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save components separately
        base_path = str(Path(filepath).with_suffix(''))
        
        feature_path = f"{base_path}_feature_extractor.pkl"
        architecture_path = f"{base_path}_architecture.pkl"
        
        self.feature_extractor.save(feature_path)
        self.model_architecture.save(architecture_path)
        
        # Save metadata
        metadata = {
            'config': self.config.__dict__,
            'dataset_type': self.dataset_type,
            'feature_type': self.feature_type,
            'architecture_type': self.architecture_type,
            'dataset_config': self.dataset_config,
            'feature_config': self.feature_config,
            'architecture_config': self.architecture_config,
            'is_trained': self.is_trained,
            'feature_extractor_path': feature_path,
            'model_architecture_path': architecture_path
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load the entire composable model"""
        # Load metadata
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        # Restore configuration
        self.config = ModelConfig(**metadata['config'])
        self.dataset_type = metadata['dataset_type']
        self.feature_type = metadata['feature_type']
        self.architecture_type = metadata['architecture_type']
        self.dataset_config = metadata['dataset_config']
        self.feature_config = metadata['feature_config']
        self.architecture_config = metadata['architecture_config']
        self.is_trained = metadata['is_trained']
        
        # Reinitialize components
        self._initialize_components()
        
        # Load trained components
        feature_path = metadata['feature_extractor_path']
        architecture_path = metadata['model_architecture_path']
        
        if os.path.exists(feature_path):
            self.feature_extractor.load(feature_path)
        
        if os.path.exists(architecture_path):
            self.model_architecture.load(architecture_path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()
        
        info.update({
            'dataset_type': self.dataset_type,
            'feature_type': self.feature_type,
            'architecture_type': self.architecture_type,
            'dataset_config': self.dataset_config,
            'feature_config': self.feature_config,
            'architecture_config': self.architecture_config
        })
        
        # Add dataset stats if available
        if self.dataset_provider and hasattr(self.dataset_provider, 'texts'):
            info['dataset_stats'] = self.dataset_provider.get_stats()
        
        # Add feature info
        if self.feature_extractor and self.feature_extractor.is_fitted:
            feature_names = self.feature_extractor.get_feature_names()
            info['num_features'] = len(feature_names) if feature_names else "Unknown"
        
        return info
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, float]:
        """Get feature importance with feature names"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")
        
        # Get importance scores
        importance = self.model_architecture.get_feature_importance()
        if importance is None:
            return {}
        
        # Get feature names
        feature_names = self.feature_extractor.get_feature_names()
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Create importance dictionary
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance and return top k
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return dict(sorted_features[:top_k])


def create_composable_model(dataset_type: str,
                          feature_type: str, 
                          architecture_type: str,
                          model_name: str,
                          dataset_config: Dict[str, Any] = None,
                          feature_config: Dict[str, Any] = None,
                          architecture_config: Dict[str, Any] = None) -> ComposableModel:
    """
    Convenience function to create composable models
    
    Args:
        dataset_type: 'berkeley', 'csv', 'synthetic', 'memory'
        feature_type: 'tfidf', 'bow', 'huggingface', 'statistical'  
        architecture_type: 'logistic', 'svm', 'random_forest', 'lightgbm', 'xgboost'
        model_name: Name for the model
        dataset_config: Dataset configuration
        feature_config: Feature extractor configuration
        architecture_config: Model architecture configuration
        
    Returns:
        Configured ComposableModel instance
    """
    config = ModelConfig(
        model_name=model_name,
        model_type="composable",
        config={
            'dataset': {
                'type': dataset_type,
                **(dataset_config or {})
            },
            'feature_extractor': {
                'type': feature_type,
                **(feature_config or {})
            },
            'model_architecture': {
                'type': architecture_type,
                **(architecture_config or {})
            }
        }
    )
    
    return ComposableModel(config)