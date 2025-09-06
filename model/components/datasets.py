from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ..base import DataProcessor


class BaseDataset(ABC):
    """Abstract base class for dataset providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.texts = None
        self.labels = None
        
    @abstractmethod
    def load(self) -> Tuple[List[str], List[int]]:
        """
        Load dataset and return texts and labels
        
        Returns:
            Tuple of (texts, labels)
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if self.texts is None or self.labels is None:
            self.load()
        return DataProcessor.get_dataset_stats(self.texts, self.labels)
    
    def create_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Create train/test split"""
        if self.texts is None or self.labels is None:
            self.load()
        return DataProcessor.create_train_test_split(
            self.texts, self.labels, test_size=test_size, random_state=random_state
        )


class BerkeleyDataset(BaseDataset):
    """Berkeley hate speech dataset provider"""
    
    def load(self) -> Tuple[List[str], List[int]]:
        """Load Berkeley dataset"""
        sample_size = self.config.get('sample_size', None)
        split = self.config.get('split', None)
        
        try:
            data = DataProcessor.load_berkeley_dataset(split=split)
            texts = data['texts']
            labels = data['labels']
            
            # Apply sample size limit if specified
            if sample_size and sample_size < len(texts):
                texts = texts[:sample_size]
                labels = labels[:sample_size]
            
            # Apply preprocessing
            if self.config.get('preprocess', True):
                texts = DataProcessor.preprocess_text(texts, **self.config.get('preprocess_kwargs', {}))
            
            self.texts = texts
            self.labels = labels
            
            return texts, labels
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Berkeley dataset: {e}")


class CSVDataset(BaseDataset):
    """CSV file dataset provider"""
    
    def load(self) -> Tuple[List[str], List[int]]:
        """Load dataset from CSV file"""
        filepath = self.config.get('filepath')
        text_column = self.config.get('text_column', 'text')
        label_column = self.config.get('label_column', 'hatespeech')
        
        if not filepath:
            raise ValueError("CSV dataset requires 'filepath' in config")
        
        try:
            data = pd.read_csv(filepath)
            
            if text_column not in data.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
            
            texts = data[text_column].tolist()
            
            # Handle labels (optional for inference-only datasets)
            if label_column in data.columns:
                raw_labels = data[label_column].tolist()
                labels = DataProcessor.binarize_labels(raw_labels)
            else:
                labels = [0] * len(texts)  # Dummy labels
                
            # Apply preprocessing
            if self.config.get('preprocess', True):
                texts = DataProcessor.preprocess_text(texts, **self.config.get('preprocess_kwargs', {}))
            
            # Apply sample size limit if specified
            sample_size = self.config.get('sample_size', None)
            if sample_size and sample_size < len(texts):
                texts = texts[:sample_size]
                labels = labels[:sample_size]
            
            self.texts = texts
            self.labels = labels
            
            return texts, labels
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV dataset from {filepath}: {e}")


class SyntheticDataset(BaseDataset):
    """Synthetic dataset for testing and development"""
    
    def load(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic hate speech data"""
        num_samples = self.config.get('num_samples', 100)
        positive_ratio = self.config.get('positive_ratio', 0.3)
        random_state = self.config.get('random_state', 42)
        
        np.random.seed(random_state)
        
        # Sample hate speech patterns
        hate_patterns = [
            "I hate {group} people",
            "{group} are terrible",
            "All {group} should die",
            "Kill all {group}",
            "{group} are disgusting",
            "I wish {group} didn't exist"
        ]
        
        neutral_patterns = [
            "I love this movie",
            "Great weather today",
            "Looking forward to the weekend",
            "This restaurant has good food",
            "Nice to meet you",
            "Have a wonderful day"
        ]
        
        groups = ["Muslims", "Christians", "immigrants", "women", "men", "Jews", "Black people", "white people"]
        
        texts = []
        labels = []
        
        num_positive = int(num_samples * positive_ratio)
        num_negative = num_samples - num_positive
        
        # Generate hate speech samples
        for _ in range(num_positive):
            pattern = np.random.choice(hate_patterns)
            group = np.random.choice(groups)
            text = pattern.format(group=group)
            texts.append(text)
            labels.append(1)
        
        # Generate neutral samples
        for _ in range(num_negative):
            text = np.random.choice(neutral_patterns)
            texts.append(text)
            labels.append(0)
        
        # Shuffle
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        self.texts = list(texts)
        self.labels = list(labels)
        
        return self.texts, self.labels


class InMemoryDataset(BaseDataset):
    """Dataset from in-memory data"""
    
    def load(self) -> Tuple[List[str], List[int]]:
        """Load from pre-provided texts and labels"""
        texts = self.config.get('texts')
        labels = self.config.get('labels')
        
        if texts is None:
            raise ValueError("InMemoryDataset requires 'texts' in config")
        if labels is None:
            raise ValueError("InMemoryDataset requires 'labels' in config")
        
        # Apply preprocessing
        if self.config.get('preprocess', True):
            texts = DataProcessor.preprocess_text(texts, **self.config.get('preprocess_kwargs', {}))
        
        self.texts = texts
        self.labels = labels
        
        return texts, labels


def get_dataset_provider(dataset_type: str, config: Dict[str, Any]) -> BaseDataset:
    """Factory function to create dataset providers"""
    providers = {
        'berkeley': BerkeleyDataset,
        'csv': CSVDataset,
        'synthetic': SyntheticDataset,
        'memory': InMemoryDataset
    }
    
    if dataset_type not in providers:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(providers.keys())}")
    
    return providers[dataset_type](config)