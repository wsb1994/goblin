import numpy as np
import pickle
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from ..base import BaseModel, ModelConfig, DataProcessor


class BagOfWordsModel(BaseModel):
    """Bag of Words with Logistic Regression for hate speech detection"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Get configuration parameters
        self.max_features = config.get('max_features', 10000)
        self.ngram_range = tuple(config.get('ngram_range', [1, 2]))
        self.min_df = config.get('min_df', 2)
        self.max_df = config.get('max_df', 0.95)
        self.C = config.get('C', 1.0)
        self.random_state = config.get('random_state', 42)
        
        # Initialize pipeline
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english',
                lowercase=True,
                strip_accents='ascii'
            )),
            ('classifier', LogisticRegression(
                C=self.C,
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'  # Handle class imbalance
            ))
        ])
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        """Train the bag of words model"""
        # Preprocess texts
        processed_texts = DataProcessor.preprocess_text(texts)
        
        # Train pipeline
        self.pipeline.fit(processed_texts, labels)
        self.is_trained = True
        
        print(f"Model trained on {len(texts)} samples")
        print(f"Vocabulary size: {len(self.pipeline['vectorizer'].vocabulary_)}")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Generate probability predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Preprocess texts
        processed_texts = DataProcessor.preprocess_text(texts)
        
        # Return probabilities for positive class
        probabilities = self.pipeline.predict_proba(processed_texts)[:, 1]
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
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        model_data = {
            'pipeline': self.pipeline,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str) -> None:
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, float]:
        """Get most important features for hate speech detection"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")
        
        # Get feature names and coefficients
        feature_names = self.pipeline['vectorizer'].get_feature_names_out()
        coefficients = self.pipeline['classifier'].coef_[0]
        
        # Get top positive and negative features
        top_positive_idx = np.argsort(coefficients)[-top_k:][::-1]
        top_negative_idx = np.argsort(coefficients)[:top_k]
        
        importance = {}
        
        # Positive features (indicative of hate speech)
        for idx in top_positive_idx:
            importance[f"{feature_names[idx]} (+)"] = float(coefficients[idx])
        
        # Negative features (indicative of non-hate speech)
        for idx in top_negative_idx:
            importance[f"{feature_names[idx]} (-)"] = float(coefficients[idx])
        
        return importance