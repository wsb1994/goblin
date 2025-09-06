#!/usr/bin/env python3
"""
Comprehensive training and evaluation pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.base import ModelConfig
from model.basic import BagOfWordsModel
from model.sophisticated import HuggingFaceLightGBMModel
from model.training import ModelTrainer, ModelEvaluator


def create_model_configs():
    """Create different model configurations for comparison"""
    configs = [
        ModelConfig(
            model_name="bow_optimized",
            model_type="basic",
            config={
                'max_features': 5000,
                'ngram_range': [1, 2],
                'C': 0.5,
                'min_df': 2  # Better for larger datasets
            }
        ),
        ModelConfig(
            model_name="hf_lightgbm",
            model_type="sophisticated",
            config={
                'hf_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'max_length': 128,
                'batch_size': 32,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'random_state': 42
            }
        )
    ]
    return configs


def main():
    """Run comprehensive training and evaluation"""
    print("Hate Speech Detection - Training & Evaluation Pipeline")
    print("=" * 60)
    
    # Create model configurations
    configs = create_model_configs()
    models = []
    for config in configs:
        if config.model_type == "basic":
            models.append(BagOfWordsModel(config))
        elif config.model_type == "sophisticated":
            models.append(HuggingFaceLightGBMModel(config))
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    print(f"Created {len(models)} model configurations:")
    for model in models:
        print(f"  - {model.config.model_name}: {model.config.config}")
    print()
    
    # Initialize trainer
    trainer = ModelTrainer(save_dir="training_outputs")
    
    try:
        # Train all models on Berkeley dataset
        print("Training models on Berkeley dataset...")
        results = trainer.load_berkeley_and_train(
            models=models,
            sample_size=20000,  # Increased from 5000 to 20000
            validation_split=0.2,
            verbose=True
        )
        
        print("\nTraining completed!")
        
        # Model comparison
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        # Load test data for evaluation
        from model.base import DataProcessor
        print("Loading test data...")
        
        # Use English_test.csv if available, otherwise use Berkeley data
        try:
            print("Loading English test dataset...")
            test_data = DataProcessor.load_dataset_with_config("English_test.csv", "english_test")
            
            if test_data['labels'] is not None:
                test_texts = test_data['texts'][:1000]  # Use subset
                test_labels = test_data['labels'][:1000]
                print(f"Using English test data: {len(test_texts)} samples")
                print(f"Detected columns - Text: '{test_data['text_column']}', Label: '{test_data['label_column']}'")
            else:
                raise ValueError("No labels found in English test data")
                
        except Exception as e:
            print(f"Could not load test data: {e}")
            print("Using Berkeley data for evaluation...")
            berkeley_data = DataProcessor.load_berkeley_dataset()
            test_texts = berkeley_data['texts'][5000:6000]  # Different subset
            test_labels = berkeley_data['labels'][5000:6000]
        
        print(f"Evaluation set: {len(test_texts)} samples")
        
        # Compare models
        comparison_df = ModelEvaluator.compare_models(
            models, test_texts, test_labels
        )
        
        print("\nModel Performance Comparison:")
        print(comparison_df.round(4))
        
        # Find optimal thresholds
        print("\n" + "=" * 60)
        print("THRESHOLD OPTIMIZATION")
        print("=" * 60)
        
        for model in models:
            if model.is_trained:
                threshold_analysis = ModelEvaluator.find_optimal_threshold(
                    model, test_texts, test_labels, metric='f1'
                )
                
                print(f"\n{model.config.model_name}:")
                print(f"  Optimal threshold: {threshold_analysis['optimal_threshold']:.3f}")
                print(f"  Optimal F1 score: {threshold_analysis['optimal_score']:.4f}")
        
        # Error analysis for best model
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS")
        print("=" * 60)
        
        # Find best model by F1 score
        best_model = None
        best_f1 = 0
        
        for model in models:
            if model.is_trained:
                metrics = ModelEvaluator.evaluate_model(model, test_texts, test_labels)
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_model = model
        
        if best_model:
            print(f"Analyzing errors for best model: {best_model.config.model_name}")
            error_analysis = ModelEvaluator.analyze_errors(
                best_model, test_texts, test_labels, num_examples=5
            )
            
            print(f"\nFalse Positives ({error_analysis['fp_count']} total):")
            for i, fp in enumerate(error_analysis['false_positives'], 1):
                print(f"  {i}. [{fp['probability']:.3f}] {fp['text'][:100]}...")
            
            print(f"\nFalse Negatives ({error_analysis['fn_count']} total):")
            for i, fn in enumerate(error_analysis['false_negatives'], 1):
                print(f"  {i}. [{fn['probability']:.3f}] {fn['text'][:100]}...")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check 'training_outputs/' directory for saved models and results.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\nThis might be due to missing dependencies. Try:")
        print("pip install datasets scikit-learn matplotlib seaborn")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Training pipeline completed successfully!")
    else:
        print("\n❌ Training pipeline failed.")