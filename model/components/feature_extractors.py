from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Optional imports
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """
        Fit the feature extractor on training texts
        
        Args:
            texts: List of training texts
        """
        pass
    
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors
        
        Args:
            texts: List of texts to transform
            
        Returns:
            Feature matrix (n_samples x n_features)
        """
        pass
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save fitted extractor"""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load fitted extractor"""
        pass
    
    def get_feature_names(self) -> List[str]:
        """Get feature names if available"""
        return []


class TFIDFExtractor(BaseFeatureExtractor):
    """TF-IDF feature extractor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.vectorizer = TfidfVectorizer(
            max_features=config.get('max_features', 10000),
            ngram_range=tuple(config.get('ngram_range', [1, 2])),
            min_df=config.get('min_df', 2),
            max_df=config.get('max_df', 0.95),
            stop_words=config.get('stop_words', 'english'),
            lowercase=config.get('lowercase', True),
            strip_accents=config.get('strip_accents', 'ascii')
        )
    
    def fit(self, texts: List[str]) -> None:
        """Fit TF-IDF vectorizer"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF features"""
        if not self.is_fitted:
            raise RuntimeError("Extractor must be fitted before transform")
        return self.vectorizer.transform(texts).toarray()
    
    def save(self, filepath: str) -> None:
        """Save fitted vectorizer"""
        if not self.is_fitted:
            raise RuntimeError("Extractor must be fitted before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'config': self.config,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load fitted vectorizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.config = data['config']
            self.is_fitted = data['is_fitted']
    
    def get_feature_names(self) -> List[str]:
        """Get TF-IDF feature names"""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


class BagOfWordsExtractor(BaseFeatureExtractor):
    """Bag of Words (Count) feature extractor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.vectorizer = CountVectorizer(
            max_features=config.get('max_features', 10000),
            ngram_range=tuple(config.get('ngram_range', [1, 1])),
            min_df=config.get('min_df', 2),
            max_df=config.get('max_df', 0.95),
            stop_words=config.get('stop_words', 'english'),
            lowercase=config.get('lowercase', True),
            strip_accents=config.get('strip_accents', 'ascii'),
            binary=config.get('binary', False)
        )
    
    def fit(self, texts: List[str]) -> None:
        """Fit Count vectorizer"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to count features"""
        if not self.is_fitted:
            raise RuntimeError("Extractor must be fitted before transform")
        return self.vectorizer.transform(texts).toarray()
    
    def save(self, filepath: str) -> None:
        """Save fitted vectorizer"""
        if not self.is_fitted:
            raise RuntimeError("Extractor must be fitted before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'config': self.config,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load fitted vectorizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.config = data['config']
            self.is_fitted = data['is_fitted']
    
    def get_feature_names(self) -> List[str]:
        """Get bag of words feature names"""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


class HuggingFaceExtractor(BaseFeatureExtractor):
    """HuggingFace transformer embeddings extractor"""
    
    def __init__(self, config: Dict[str, Any]):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch required for HuggingFace extractor")
            
        super().__init__(config)
        
        self.model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.max_length = config.get('max_length', 128)
        self.batch_size = config.get('batch_size', 32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize HuggingFace model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model {self.model_name}: {e}")
    
    def fit(self, texts: List[str]) -> None:
        """HuggingFace models don't need fitting"""
        self.is_fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to embeddings"""
        if not self.is_fitted:
            raise RuntimeError("Extractor must be fitted before transform")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
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
                outputs = self.model(**encoded)
                # Use mean pooling over sequence dimension
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def save(self, filepath: str) -> None:
        """Save extractor config (model is reloaded from HuggingFace)"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'model_name': self.model_name,
                'max_length': self.max_length,
                'batch_size': self.batch_size,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load extractor config and reinitialize model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self.model_name = data['model_name']
            self.max_length = data['max_length']
            self.batch_size = data['batch_size']
            self.is_fitted = data['is_fitted']
        
        self._initialize_model()


class StatisticalExtractor(BaseFeatureExtractor):
    """Statistical text features (length, word count, etc.)"""
    
    def fit(self, texts: List[str]) -> None:
        """Statistical features don't need fitting"""
        self.is_fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Extract statistical features"""
        if not self.is_fitted:
            raise RuntimeError("Extractor must be fitted before transform")
        
        features = []
        
        for text in texts:
            text_features = [
                len(text),  # Character count
                len(text.split()),  # Word count
                len([w for w in text.split() if len(w) > 3]),  # Long word count
                text.count('!'),  # Exclamation marks
                text.count('?'),  # Question marks
                text.count('.'),  # Periods
                len([c for c in text if c.isupper()]) / max(len(text), 1),  # Uppercase ratio
                text.count(' ') / max(len(text), 1),  # Space ratio
            ]
            features.append(text_features)
        
        return np.array(features)
    
    def save(self, filepath: str) -> None:
        """Save config"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load config"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self.is_fitted = data['is_fitted']
    
    def get_feature_names(self) -> List[str]:
        """Get statistical feature names"""
        return [
            'char_count', 'word_count', 'long_word_count',
            'exclamation_count', 'question_count', 'period_count',
            'uppercase_ratio', 'space_ratio'
        ]


def get_feature_extractor(extractor_type: str, config: Dict[str, Any]) -> BaseFeatureExtractor:
    """Factory function to create feature extractors"""
    extractors = {
        'tfidf': TFIDFExtractor,
        'bow': BagOfWordsExtractor,
        'huggingface': HuggingFaceExtractor,
        'statistical': StatisticalExtractor
    }
    
    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {list(extractors.keys())}")
    
    return extractors[extractor_type](config)