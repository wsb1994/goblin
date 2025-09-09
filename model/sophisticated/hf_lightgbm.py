import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb

from ..base import BaseModel, ModelConfig, DataProcessor

# Try to import transformers, but make it optional
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceLightGBMModel(BaseModel):
    """HuggingFace embeddings + LightGBM for hate speech detection"""
    
    def __init__(self, config: ModelConfig):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for this model. "
                "Install with: pip install transformers torch"
            )
            
        super().__init__(config)
        
        # Configuration
        self.model_name = config.get('hf_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.max_length = config.get('max_length', 128)
        self.batch_size = config.get('batch_size', 32)
        
        # LightGBM parameters
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.1),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'verbose': -1,
            'random_state': config.get('random_state', 42),
            'is_unbalance': True  # Handle class imbalance
        }
        
        # Initialize HuggingFace components
        self.tokenizer = None
        self.embedding_model = None
        self.lgb_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize HuggingFace tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.embedding_model = AutoModel.from_pretrained(self.model_name)
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
            
            print(f"Loaded HuggingFace model: {self.model_name}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model {self.model_name}: {e}")
    
    def _extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings from texts using HuggingFace model"""
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**encoded)
                
                # Use mean pooling over sequence dimension
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy()
                
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        """Train the HuggingFace + LightGBM model"""
        print("Extracting embeddings from training texts...")
        
        # Preprocess texts
        processed_texts = DataProcessor.preprocess_text(texts)
        
        # Extract embeddings
        embeddings = self._extract_embeddings(processed_texts)
        
        print(f"Extracted embeddings shape: {embeddings.shape}")
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(embeddings, label=labels)
        
        # Train LightGBM model
        print("Training LightGBM classifier...")
        self.lgb_model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.config.get('num_boost_round', 100),
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        print(f"Model trained on {len(texts)} samples")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Generate probability predictions"""
        if not self.is_trained or self.lgb_model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Preprocess texts
        processed_texts = DataProcessor.preprocess_text(texts)
        
        # Extract embeddings
        embeddings = self._extract_embeddings(processed_texts)
        
        # Get predictions from LightGBM
        probabilities = self.lgb_model.predict(embeddings)
        
        return probabilities
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Get predictions
        probabilities = self.predict(texts)
        binary_predictions = (probabilities > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, binary_predictions),
            'precision': precision_score(labels, binary_predictions, zero_division=0),
            'recall': recall_score(labels, binary_predictions, zero_division=0),
            'f1': f1_score(labels, binary_predictions, zero_division=0),
            'auc_roc': roc_auc_score(labels, probabilities)
        }
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """Save trained model"""
        if not self.is_trained or self.lgb_model is None:
            raise RuntimeError("Model must be trained before saving")
        
        model_data = {
            'config': self.config,
            'lgb_params': self.lgb_params,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'is_trained': self.is_trained
        }
        
        # Save LightGBM model separately (it has its own format)
        lgb_filepath = filepath.replace('.pkl', '_lgb.txt')
        self.lgb_model.save_model(lgb_filepath)
        
        # Save metadata
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath} and {lgb_filepath}")
    
    def load(self, filepath: str) -> None:
        """Load trained model"""
        # Load metadata
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.lgb_params = model_data['lgb_params']
        self.model_name = model_data['model_name']
        self.max_length = model_data['max_length']
        self.batch_size = model_data['batch_size']
        self.is_trained = model_data['is_trained']
        
        # Load LightGBM model
        lgb_filepath = filepath.replace('.pkl', '_lgb.txt')
        self.lgb_model = lgb.Booster(model_file=lgb_filepath)
        
        # Reinitialize HuggingFace components
        self._initialize_embedding_model()
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """Get LightGBM feature importance"""
        if not self.is_trained or self.lgb_model is None:
            raise RuntimeError("Model must be trained to get feature importance")
        
        importance = self.lgb_model.feature_importance(importance_type=importance_type)
        
        # Create feature names (embeddings don't have meaningful names)
        feature_names = [f'embedding_dim_{i}' for i in range(len(importance))]
        
        return dict(zip(feature_names, importance))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata with additional HF/LightGBM info"""
        info = super().get_model_info()
        info.update({
            'embedding_model': self.model_name,
            'embedding_dim': None if not self.is_trained else len(self.get_feature_importance()),
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'lgb_params': self.lgb_params
        })
        return info