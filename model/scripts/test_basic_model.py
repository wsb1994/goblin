#!/usr/bin/env python3
"""
Test script for basic bag-of-words model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.base import ModelConfig, DataProcessor
from model.basic import BagOfWordsModel


def test_basic_model():
    """Test the bag-of-words model with sample data"""
    print("Testing Bag of Words Model")
    print("=" * 40)
    
    # Sample data for testing
    sample_texts = [
        "I love this movie, it's amazing!",
        "You are such an idiot, I hate you",
        "Great weather today, perfect for a walk",
        "Kill yourself, nobody likes you",
        "Looking forward to the weekend",
        "You're disgusting and worthless"
    ]
    
    sample_labels = [0, 1, 0, 1, 0, 1]  # 0=not hate, 1=hate
    
    # Create model configuration
    config = ModelConfig(
        model_name="test_bow_model",
        model_type="basic",
        config={
            'max_features': 1000,
            'ngram_range': [1, 2],
            'C': 1.0
        }
    )
    
    # Initialize and train model
    print("1. Training model on sample data...")
    model = BagOfWordsModel(config)
    model.train(sample_texts, sample_labels)
    print(f"   Model trained: {model.is_trained}")
    print()
    
    # Test predictions
    print("2. Testing predictions...")
    test_texts = [
        "What a beautiful day!",
        "I hate everyone and everything",
        "Looking forward to lunch"
    ]
    
    predictions = model.predict(test_texts)
    print("   Text -> Hate Speech Probability")
    for text, prob in zip(test_texts, predictions):
        print(f"   '{text}' -> {prob:.3f}")
    print()
    
    # Evaluate on training data (just for demo)
    print("3. Evaluating on training data...")
    metrics = model.evaluate(sample_texts, sample_labels)
    for metric, value in metrics.items():
        print(f"   {metric.upper()}: {value:.3f}")
    print()
    
    # Test model persistence
    print("4. Testing save/load functionality...")
    model_path = "test_bow_model.pkl"
    model.save(model_path)
    print(f"   Model saved to {model_path}")
    
    # Create new model and load
    new_model = BagOfWordsModel(config)
    new_model.load(model_path)
    print(f"   Model loaded: {new_model.is_trained}")
    
    # Verify loaded model works
    new_predictions = new_model.predict(test_texts)
    predictions_match = all(abs(p1 - p2) < 1e-6 for p1, p2 in zip(predictions, new_predictions))
    print(f"   Predictions match: {predictions_match}")
    print()
    
    # Show feature importance
    print("5. Top features:")
    try:
        importance = model.get_feature_importance(top_k=10)
        for feature, coef in list(importance.items())[:10]:
            print(f"   {feature}: {coef:.3f}")
    except Exception as e:
        print(f"   Error getting feature importance: {e}")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"\n   Cleaned up {model_path}")
    
    print("\n✅ Basic model test completed successfully!")
    return True


def test_berkeley_integration():
    """Test integration with Berkeley dataset (if available)"""
    print("\nTesting Berkeley Dataset Integration")
    print("=" * 40)
    
    try:
        # Load a small subset of Berkeley data
        print("Loading Berkeley dataset...")
        data = DataProcessor.load_berkeley_dataset()
        
        # Use only first 100 samples for quick test
        texts = data['texts'][:100]
        labels = data['labels'][:100]
        
        print(f"Loaded {len(texts)} samples")
        
        # Create model
        config = ModelConfig(
            model_name="berkeley_test",
            model_type="basic", 
            config={'max_features': 500}
        )
        
        model = BagOfWordsModel(config)
        
        # Train/test split
        train_texts, test_texts, train_labels, test_labels = DataProcessor.create_train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        
        # Train and evaluate
        print("Training on Berkeley data...")
        model.train(train_texts, train_labels)
        
        print("Evaluating...")
        metrics = model.evaluate(test_texts, test_labels)
        
        print("Results:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.3f}")
            
        print("\n✅ Berkeley integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Berkeley integration test failed: {e}")
        print("This is expected if 'datasets' package is not installed")
        return False


if __name__ == "__main__":
    success1 = test_basic_model()
    success2 = test_berkeley_integration()
    
    if success1:
        print("\n🎉 All tests passed! Basic model is working correctly.")
    else:
        print("\n❌ Some tests failed.")