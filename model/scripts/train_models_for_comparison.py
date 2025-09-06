#!/usr/bin/env python3
"""
Train models on Berkeley data for comparison (fast version)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.pipeline import create_composable_model
from model.integration import ModelRegistry
from model.base import DataProcessor
import json


def create_production_models():
    """Create production models that train efficiently"""
    
    # Model configurations for comparison
    model_configs = [
        {
            'name': 'bow_logistic_model',
            'dataset': 'berkeley',
            'features': 'bow', 
            'architecture': 'logistic',
            'description': 'Bag of Words + Logistic Regression',
            'dataset_config': {'sample_size': 5000},
            'feature_config': {'max_features': 5000, 'ngram_range': [1, 2]},
            'architecture_config': {'C': 0.5}
        },
        {
            'name': 'huggingface_lightgbm_model',
            'dataset': 'berkeley',
            'features': 'huggingface',
            'architecture': 'lightgbm', 
            'description': 'HuggingFace Embeddings + LightGBM',
            'dataset_config': {'sample_size': 5000},
            'feature_config': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'max_length': 128,
                'batch_size': 32
            },
            'architecture_config': {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
        }
    ]
    
    print("🏗️  Training Models on Berkeley Data")
    print("=" * 50)
    
    trained_models = []
    
    for config in model_configs:
        print(f"\n🔧 Training: {config['description']}")
        
        try:
            model = create_composable_model(
                dataset_type=config['dataset'],
                feature_type=config['features'],
                architecture_type=config['architecture'], 
                model_name=config['name'],
                dataset_config=config['dataset_config'],
                feature_config=config['feature_config'],
                architecture_config=config['architecture_config']
            )
            
            print("   ⏳ Training...")
            model.train()
            
            trained_models.append({
                'name': config['name'],
                'model': model,
                'description': config['description']
            })
            
            print("   ✅ Completed")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    return trained_models


def quick_test(trained_models):
    """Quick test on sample data"""
    
    test_texts = [
        "I love this beautiful day!",
        "I hate all those people and wish they were dead",
        "The weather is nice today",
        "Kill yourself, nobody likes you"
    ]
    
    print(f"\n🧪 Quick Test on Sample Data")
    print("=" * 40)
    
    for model_info in trained_models:
        model = model_info['model']
        name = model_info['name']
        
        print(f"\n🔍 {name}:")
        
        try:
            predictions = model.predict(test_texts)
            
            for text, prob in zip(test_texts, predictions):
                label = "HATE" if prob > 0.5 else "OK"
                print(f"   [{prob:.3f}] {label}: {text[:50]}...")
                
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")


if __name__ == "__main__":
    print("🚀 Model Training & Comparison Pipeline")
    print("📚 Training on: Berkeley Hate Speech Dataset")
    print()
    
    models = create_production_models()
    
    if models:
        print(f"\n✅ Trained {len(models)} models successfully")
        quick_test(models)
    else:
        print("❌ No models trained successfully")
        
    print(f"\n🎉 Training completed!")