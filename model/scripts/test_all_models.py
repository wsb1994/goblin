#!/usr/bin/env python3
"""
Test script to demonstrate all model types working together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.base import ModelConfig


def test_basic_models():
    """Test basic models"""
    print("🔧 Testing Basic Models")
    print("=" * 40)
    
    try:
        from model.basic import BagOfWordsModel
        
        config = ModelConfig(
            model_name="test_bow",
            model_type="basic",
            config={'max_features': 1000}
        )
        
        model = BagOfWordsModel(config)
        
        # Sample training data
        texts = ["I love this", "I hate you", "Great day today", "Kill everyone"]
        labels = [0, 1, 0, 1]
        
        model.train(texts, labels)
        
        # Test prediction
        test_texts = ["Nice weather", "You suck"]
        predictions = model.predict(test_texts)
        
        print(f"✅ Basic model working: {predictions}")
        return True
        
    except ImportError as e:
        print(f"❌ Basic models not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Basic model test failed: {e}")
        return False


def test_sophisticated_models():
    """Test sophisticated models"""
    print("\n🚀 Testing Sophisticated Models")
    print("=" * 40)
    
    try:
        from model.sophisticated import HuggingFaceLightGBMModel
        
        config = ModelConfig(
            model_name="test_hf_lgb",
            model_type="sophisticated",
            config={
                'hf_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'num_boost_round': 10  # Small for testing
            }
        )
        
        model = HuggingFaceLightGBMModel(config)
        
        # Sample training data (needs more for embeddings)
        texts = [
            "I love this movie, it's amazing!",
            "You are such an idiot, I hate you", 
            "Great weather today, perfect for a walk",
            "Kill yourself, nobody likes you",
            "Looking forward to the weekend",
            "You're disgusting and worthless",
            "Beautiful sunset tonight",
            "I wish you were dead"
        ]
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        
        print("Training sophisticated model (this may take a moment)...")
        model.train(texts, labels)
        
        # Test prediction
        test_texts = ["Nice weather today", "You are terrible"]
        predictions = model.predict(test_texts)
        
        print(f"✅ Sophisticated model working: {predictions}")
        return True
        
    except ImportError as e:
        print(f"❌ Sophisticated models not available: {e}")
        print("Install with: pip install transformers torch lightgbm")
        return False
    except Exception as e:
        print(f"❌ Sophisticated model test failed: {e}")
        return False


def test_api_models():
    """Test API-based models"""
    print("\n🌐 Testing API Models")
    print("=" * 40)
    
    try:
        from model.api import ClaudeHateSpeechModel
        
        # Note: This will fail without a real API key
        config = ModelConfig(
            model_name="test_claude",
            model_type="api",
            config={
                'api_key': 'test_key_will_fail',  # This will fail, but tests the import
                'claude_model': 'claude-3-haiku-20240307'
            }
        )
        
        try:
            model = ClaudeHateSpeechModel(config)
            print("✅ API model initialized (but will fail without real API key)")
            return True
        except ValueError as e:
            if "API key" in str(e):
                print("✅ API model structure working (needs real API key to function)")
                return True
            raise e
        
    except ImportError as e:
        print(f"❌ API models not available: {e}")
        print("Install with: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ API model test failed: {e}")
        return False


def test_training_infrastructure():
    """Test training infrastructure"""
    print("\n📊 Testing Training Infrastructure")
    print("=" * 40)
    
    try:
        from model.training import ModelTrainer, ModelEvaluator
        from model.basic import BagOfWordsModel
        
        # Create a simple model
        config = ModelConfig(
            model_name="test_training",
            model_type="basic",
            config={'max_features': 500}
        )
        
        model = BagOfWordsModel(config)
        
        # Sample data
        texts = [f"Sample text {i}" for i in range(20)]
        labels = [i % 2 for i in range(20)]  # Alternating 0,1
        
        # Test trainer
        trainer = ModelTrainer(save_dir="test_outputs")
        results = trainer.train_model(model, texts, labels, save_model=False, verbose=False)
        
        print(f"✅ Training infrastructure working: F1={results['val_metrics']['f1']:.3f}")
        
        # Clean up
        import shutil
        if os.path.exists("test_outputs"):
            shutil.rmtree("test_outputs")
            
        return True
        
    except ImportError as e:
        print(f"❌ Training infrastructure not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Training infrastructure test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing All Model Types")
    print("=" * 60)
    
    results = {
        'basic': test_basic_models(),
        'sophisticated': test_sophisticated_models(), 
        'api': test_api_models(),
        'training': test_training_infrastructure()
    }
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for model_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_type.upper():<15} {status}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_pass}/{total_tests} model types working")
    
    if total_pass == total_tests:
        print("\n🎉 All model types are working correctly!")
        return True
    else:
        print("\n⚠️  Some model types have issues (likely due to missing dependencies)")
        print("\nTo install all dependencies:")
        print("pip install datasets scikit-learn transformers torch lightgbm anthropic matplotlib seaborn")
        return False


if __name__ == "__main__":
    main()