import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
import datasets
import re

# Dataset configuration mappings
DATASET_CONFIGS = {
    'berkeley': {
        'text_column': 'text',
        'label_column': 'hatespeech'
    },
    'english_test': {
        'text_column': 'text',
        'label_column': 'label'
    }
}


class DataProcessor:
    """Utilities for data loading, preprocessing, and standardization"""
    
    @staticmethod
    def load_berkeley_dataset(split: Optional[str] = None) -> Dict[str, Any]:
        """
        Load Berkeley hate speech dataset
        
        Args:
            split: Optional dataset split to load ('train', 'test', etc.)
            
        Returns:
            Dataset dictionary with standardized format
        """
        try:
            dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')
            
            # Convert to pandas for easier manipulation
            if split:
                data = dataset[split].to_pandas()
            else:
                # Combine all splits if no specific split requested
                data = pd.concat([dataset[s].to_pandas() for s in dataset.keys()], ignore_index=True)
            
            return {
                'texts': data['text'].tolist(),
                'labels': DataProcessor.binarize_labels(data['hatespeech'].tolist()),
                'raw_data': data
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load Berkeley dataset: {e}")
    
    @staticmethod
    def load_evaluation_data(filepath: str, text_column: str = None, label_column: str = None) -> Dict[str, Any]:
        """
        Load evaluation dataset from CSV file with flexible column mapping
        
        Args:
            filepath: Path to CSV file
            text_column: Name of text column (auto-detected if None)
            label_column: Name of label column (auto-detected if None)
            
        Returns:
            Dataset dictionary with standardized format
        """
        try:
            data = pd.read_csv(filepath)
            
            # Auto-detect text column if not specified
            if text_column is None:
                text_candidates = ['text', 'comment', 'content', 'message']
                text_column = next((col for col in text_candidates if col in data.columns), None)
                if text_column is None:
                    raise ValueError(f"Could not find text column. Available columns: {list(data.columns)}")
            
            if text_column not in data.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV. Available columns: {list(data.columns)}")
            
            # Auto-detect label column if not specified
            if label_column is None:
                label_candidates = ['label', 'hatespeech', 'hate_speech', 'target', 'class']
                label_column = next((col for col in label_candidates if col in data.columns), None)
            
            # Handle case where labels might not exist (inference only)
            if label_column and label_column in data.columns:
                labels = DataProcessor.binarize_labels(data[label_column].tolist())
            else:
                labels = None
                
            return {
                'texts': data[text_column].tolist(),
                'labels': labels,
                'raw_data': data,
                'text_column': text_column,
                'label_column': label_column
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load evaluation data from {filepath}: {e}")
    
    @staticmethod
    def load_dataset_with_config(filepath: str, dataset_type: str) -> Dict[str, Any]:
        """
        Load dataset using predefined configuration
        
        Args:
            filepath: Path to CSV file  
            dataset_type: Dataset type ('berkeley', 'english_test', etc.)
            
        Returns:
            Dataset dictionary with standardized format
        """
        if dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset type '{dataset_type}'. Available: {list(DATASET_CONFIGS.keys())}")
        
        config = DATASET_CONFIGS[dataset_type]
        return DataProcessor.load_evaluation_data(
            filepath, 
            text_column=config['text_column'], 
            label_column=config['label_column']
        )
    
    @staticmethod
    def binarize_labels(labels: List[Any]) -> List[int]:
        """
        Convert hate speech labels to binary format
        
        Args:
            labels: Raw labels (0 = negative, 1+ = positive)
            
        Returns:
            Binary labels (0 or 1)
        """
        return [1 if label > 0 else 0 for label in labels]
    
    @staticmethod
    def preprocess_text(texts: List[str], 
                       lowercase: bool = True,
                       remove_urls: bool = True,
                       remove_mentions: bool = True,
                       remove_extra_whitespace: bool = True) -> List[str]:
        """
        Standardized text preprocessing pipeline
        
        Args:
            texts: List of text strings
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_extra_whitespace: Clean up whitespace
            
        Returns:
            Preprocessed text list
        """
        processed_texts = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
                
            # Remove URLs
            if remove_urls:
                text = re.sub(r'http\S+|www\S+', '', text)
            
            # Remove mentions
            if remove_mentions:
                text = re.sub(r'@\w+', '', text)
            
            # Convert to lowercase
            if lowercase:
                text = text.lower()
            
            # Remove extra whitespace
            if remove_extra_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
            
            processed_texts.append(text)
        
        return processed_texts
    
    @staticmethod
    def create_train_test_split(texts: List[str], 
                               labels: List[int],
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Create reproducible train/test splits
        
        Args:
            texts: List of text inputs
            labels: List of binary labels
            test_size: Fraction for test set
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels  # Maintain class balance
        )
    
    @staticmethod
    def get_dataset_stats(texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """
        Generate dataset statistics
        
        Args:
            texts: List of text inputs
            labels: List of binary labels
            
        Returns:
            Statistics dictionary
        """
        total_samples = len(texts)
        positive_samples = sum(labels)
        negative_samples = total_samples - positive_samples
        
        text_lengths = [len(text.split()) for text in texts]
        
        return {
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'positive_ratio': positive_samples / total_samples if total_samples > 0 else 0,
            'avg_text_length': np.mean(text_lengths),
            'min_text_length': np.min(text_lengths),
            'max_text_length': np.max(text_lengths),
            'median_text_length': np.median(text_lengths)
        }