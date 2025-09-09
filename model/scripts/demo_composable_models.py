#!/usr/bin/env python3
"""
Demo script showing the new composable model system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.pipeline import create_composable_model
from model.training import ModelTrainer, ModelEvaluator


def demo_basic_combinations():
    """Demo basic feature extractor + architecture combinations"""
    print("🧩 Composable Model System Demo")
    print("=" * 50)
    print("Mixing and matching Dataset + Features + Architecture")
    print()
    
    # Create different combinations
    combinations = [
        {
            'name': 'Berkeley + TF-IDF + Logistic',
            'dataset': 'berkeley',
            'features': 'tfidf', 
            'architecture': 'logistic',
            'dataset_config': {'sample_size': 1000},
            'feature_config': {'max_features': 5000, 'ngram_range': [1, 2]},
            'architecture_config': {'C': 1.0}
        },
        {
            'name': 'Berkeley + Bag-of-Words + Random Forest',
            'dataset': 'berkeley',
            'features': 'bow',
            'architecture': 'random_forest',
            'dataset_config': {'sample_size': 1000},
            'feature_config': {'max_features': 3000, 'ngram_range': [1, 1]},
            'architecture_config': {'n_estimators': 50}
        },
        {
            'name': 'Synthetic + Statistical + SVM',
            'dataset': 'synthetic',
            'features': 'statistical',
            'architecture': 'svm',
            'dataset_config': {'num_samples': 500, 'positive_ratio': 0.3},
            'feature_config': {},
            'architecture_config': {'C': 0.1, 'kernel': 'linear'}
        }
    ]
    
    models = []
    results = []
    
    for combo in combinations:
        print(f"🔧 Creating: {combo['name']}")
        
        try:
            model = create_composable_model(
                dataset_type=combo['dataset'],
                feature_type=combo['features'],
                architecture_type=combo['architecture'],
                model_name=combo['name'].lower().replace(' ', '_'),
                dataset_config=combo['dataset_config'],
                feature_config=combo['feature_config'],
                architecture_config=combo['architecture_config']
            )
            
            # Quick training
            print("   Training...")
            model.train()
            
            # Quick evaluation on training data (for demo)
            if model.dataset_provider.texts:
                metrics = model.evaluate(
                    model.dataset_provider.texts[:100], 
                    model.dataset_provider.labels[:100]
                )
                
                print(f"   ✅ F1: {metrics['f1']:.3f}, AUC: {metrics['auc_roc']:.3f}")
                
                models.append(model)
                results.append({
                    'name': combo['name'],
                    'f1': metrics['f1'],
                    'auc': metrics['auc_roc']
                })
        
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print()
    
    # Summary
    print("📊 Results Summary:")
    print("-" * 30)
    for result in results:
        print(f"{result['name']:<30} F1: {result['f1']:.3f} AUC: {result['auc']:.3f}")
    
    return models


def demo_advanced_combination():
    """Demo the HuggingFace + Logistic combination you mentioned"""
    print("\n🚀 Advanced Demo: HuggingFace Embeddings + Logistic Regression")
    print("=" * 60)
    
    try:
        # This is the combination you specifically mentioned
        model = create_composable_model(
            dataset_type='berkeley',
            feature_type='huggingface',
            architecture_type='logistic',
            model_name='hf_embeddings_logistic',
            dataset_config={
                'sample_size': 500,  # Small for demo
                'preprocess': True
            },
            feature_config={
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'max_length': 128,
                'batch_size': 16
            },
            architecture_config={
                'C': 1.0,
                'max_iter': 500
            }
        )
        
        print("Components:")
        print(f"  📚 Dataset: Berkeley hate speech (500 samples)")
        print(f"  🔤 Features: HuggingFace sentence-transformers/all-MiniLM-L6-v2")
        print(f"  🧠 Architecture: Logistic Regression")
        print()
        
        print("Training...")
        model.train()
        
        # Get model info
        info = model.get_model_info()
        print(f"✅ Training completed!")
        print(f"   Feature dimensions: {info.get('num_features', 'Unknown')}")
        
        # Test predictions
        test_texts = [
            "I love this beautiful day!",
            "I hate all those people",
            "The weather is nice today"
        ]
        
        print("\n🔍 Test Predictions:")
        predictions = model.predict(test_texts)
        for text, prob in zip(test_texts, predictions):
            label = "HATE" if prob > 0.5 else "OK"
            print(f"   [{prob:.3f}] {label}: {text}")
        
        # Feature importance (from logistic regression coefficients)
        try:
            importance = model.get_feature_importance(top_k=5)
            if importance:
                print(f"\n📈 Top Features (embedding dimensions):")
                for feature, score in list(importance.items())[:5]:
                    print(f"   {feature}: {score:.4f}")
        except:
            print("\n📈 Feature importance not available for embeddings")
        
        return model
        
    except Exception as e:
        print(f"❌ Advanced demo failed: {e}")
        print("This likely requires: pip install transformers torch")
        return None


def demo_easy_configuration():
    """Show how easy it is to create different combinations"""
    print("\n⚡ Quick Configuration Examples")
    print("=" * 40)
    
    examples = [
        "create_composable_model('berkeley', 'tfidf', 'logistic', 'basic_model')",
        "create_composable_model('csv', 'huggingface', 'lightgbm', 'advanced_model')", 
        "create_composable_model('synthetic', 'statistical', 'random_forest', 'test_model')",
        "create_composable_model('berkeley', 'bow', 'svm', 'svm_model')"
    ]
    
    print("Creating models is now as simple as:")
    for example in examples:
        print(f"  {example}")
    
    print(f"\nAvailable combinations:")
    print(f"  📚 Datasets: berkeley, csv, synthetic, memory")
    print(f"  🔤 Features: tfidf, bow, huggingface, statistical") 
    print(f"  🧠 Architectures: logistic, svm, random_forest, lightgbm, xgboost")
    
    total_combinations = 4 * 4 * 5  # datasets * features * architectures
    print(f"\n🎯 Total possible combinations: {total_combinations}")


def main():
    """Run all demos"""
    try:
        # Demo basic combinations
        models = demo_basic_combinations()
        
        # Demo advanced combination
        advanced_model = demo_advanced_combination()
        
        # Show configuration examples
        demo_easy_configuration()
        
        print(f"\n🎉 Composable Model System Demo Completed!")
        print(f"You can now mix and match any combination of:")
        print(f"  - Data sources (Berkeley, CSV files, synthetic data)")
        print(f"  - Feature extraction (TF-IDF, embeddings, statistical)")
        print(f"  - ML algorithms (Logistic, SVM, Random Forest, LightGBM, etc.)")
        
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()