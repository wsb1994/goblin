#!/usr/bin/env python3
"""
Data exploration script for Berkeley hate speech dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.base import DataProcessor
import pandas as pd


def explore_berkeley_dataset():
    """Explore and display Berkeley dataset statistics"""
    print("Loading Berkeley hate speech dataset...")
    
    try:
        # Load dataset
        data = DataProcessor.load_berkeley_dataset()
        texts = data['texts']
        labels = data['labels']
        
        print(f"Successfully loaded {len(texts)} samples")
        print()
        
        # Dataset statistics
        stats = DataProcessor.get_dataset_stats(texts, labels)
        print("Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Positive samples: {stats['positive_samples']:,}")
        print(f"  Negative samples: {stats['negative_samples']:,}")
        print(f"  Positive ratio: {stats['positive_ratio']:.3f}")
        print(f"  Average text length: {stats['avg_text_length']:.1f} words")
        print(f"  Text length range: {stats['min_text_length']} - {stats['max_text_length']} words")
        print()
        
        # Sample data
        print("Sample positive examples:")
        positive_indices = [i for i, label in enumerate(labels) if label == 1][:3]
        for i, idx in enumerate(positive_indices, 1):
            print(f"  {i}. {texts[idx][:100]}...")
        print()
        
        print("Sample negative examples:")
        negative_indices = [i for i, label in enumerate(labels) if label == 0][:3]
        for i, idx in enumerate(negative_indices, 1):
            print(f"  {i}. {texts[idx][:100]}...")
        print()
        
        # Preprocessing example
        print("Text preprocessing example:")
        sample_text = texts[0]
        print(f"Original: {sample_text}")
        
        preprocessed = DataProcessor.preprocess_text([sample_text])
        print(f"Preprocessed: {preprocessed[0]}")
        print()
        
        # Train/test split example
        print("Creating train/test split (80/20)...")
        X_train, X_test, y_train, y_test = DataProcessor.create_train_test_split(
            texts[:1000],  # Use subset for demo
            labels[:1000]
        )
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train positive ratio: {sum(y_train)/len(y_train):.3f}")
        print(f"Test positive ratio: {sum(y_test)/len(y_test):.3f}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have the 'datasets' package installed:")
        print("pip install datasets")
        return False
        
    return True


def explore_evaluation_data():
    """Explore the evaluation CSV file"""
    eval_file = "English_test.csv"
    
    if not os.path.exists(eval_file):
        print(f"Evaluation file {eval_file} not found in current directory")
        return False
        
    try:
        data = DataProcessor.load_evaluation_data(eval_file)
        texts = data['texts']
        labels = data['labels']
        
        print(f"Evaluation dataset: {len(texts)} samples")
        
        if labels:
            stats = DataProcessor.get_dataset_stats(texts, labels)
            print(f"Positive ratio: {stats['positive_ratio']:.3f}")
        else:
            print("No labels found - inference only dataset")
            
        print(f"Sample text: {texts[0][:100]}...")
        
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return False
        
    return True


if __name__ == "__main__":
    print("=== Berkeley Dataset Exploration ===")
    berkeley_success = explore_berkeley_dataset()
    
    print("\n=== Evaluation Dataset Exploration ===")
    eval_success = explore_evaluation_data()
    
    if berkeley_success and eval_success:
        print("\n✅ Data exploration completed successfully!")
    else:
        print("\n❌ Some data exploration failed. Check the output above.")