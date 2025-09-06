from typing import Dict, List, Any, Optional, Tuple
import time
import json
from pathlib import Path

from ..base import BaseModel, DataProcessor


class ModelTrainer:
    """Centralized training pipeline for all model types"""
    
    def __init__(self, save_dir: str = "model_outputs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def train_model(self, 
                   model: BaseModel,
                   texts: List[str],
                   labels: List[int],
                   validation_split: float = 0.2,
                   save_model: bool = True,
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Train a model with validation and logging
        
        Args:
            model: Model instance to train
            texts: Training texts
            labels: Training labels
            validation_split: Fraction for validation set
            save_model: Whether to save the trained model
            verbose: Whether to print progress
            
        Returns:
            Training results dictionary
        """
        if verbose:
            print(f"Training {model.config.model_name} ({model.config.model_type})")
            print(f"Total samples: {len(texts)}")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = DataProcessor.create_train_test_split(
            texts, labels, test_size=validation_split
        )
        
        if verbose:
            print(f"Train samples: {len(train_texts)}")
            print(f"Validation samples: {len(val_texts)}")
        
        # Training timing
        start_time = time.time()
        
        # Train model
        model.train(train_texts, train_labels)
        
        training_time = time.time() - start_time
        
        # Validation evaluation
        val_metrics = model.evaluate(val_texts, val_labels)
        train_metrics = model.evaluate(train_texts, train_labels)
        
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print("Validation metrics:")
            for metric, value in val_metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        # Prepare results
        results = {
            'model_info': model.get_model_info(),
            'training_time': training_time,
            'train_samples': len(train_texts),
            'val_samples': len(val_texts),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': time.time()
        }
        
        # Save model and results
        if save_model:
            model_path = self.save_dir / f"{model.config.model_name}.pkl"
            results_path = self.save_dir / f"{model.config.model_name}_results.json"
            
            model.save(str(model_path))
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            if verbose:
                print(f"Model saved to {model_path}")
                print(f"Results saved to {results_path}")
        
        return results
    
    def train_multiple_models(self,
                            models: List[BaseModel],
                            texts: List[str], 
                            labels: List[int],
                            **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models and compare results
        
        Args:
            models: List of model instances
            texts: Training texts
            labels: Training labels
            **kwargs: Arguments passed to train_model
            
        Returns:
            Dictionary mapping model names to results
        """
        all_results = {}
        
        print("Training multiple models...")
        print("=" * 50)
        
        for i, model in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Training {model.config.model_name}")
            print("-" * 30)
            
            results = self.train_model(model, texts, labels, **kwargs)
            all_results[model.config.model_name] = results
        
        # Summary comparison
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"{'Model':<20} {'Val F1':<8} {'Val AUC':<8} {'Time':<8}")
        print("-" * 50)
        
        for name, results in all_results.items():
            val_metrics = results['val_metrics']
            print(f"{name:<20} {val_metrics['f1']:<8.3f} {val_metrics['auc_roc']:<8.3f} {results['training_time']:<8.1f}s")
        
        return all_results
    
    def load_berkeley_and_train(self,
                              models: List[BaseModel],
                              sample_size: Optional[int] = None,
                              **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Convenience method to load Berkeley data and train models
        
        Args:
            models: List of models to train
            sample_size: Optional limit on number of samples
            **kwargs: Arguments passed to train_multiple_models
            
        Returns:
            Training results for all models
        """
        print("Loading Berkeley hate speech dataset...")
        data = DataProcessor.load_berkeley_dataset()
        
        texts = data['texts']
        labels = data['labels']
        
        if sample_size and sample_size < len(texts):
            print(f"Using {sample_size} samples (subset of {len(texts)})")
            texts = texts[:sample_size]
            labels = labels[:sample_size]
        
        # Dataset stats
        stats = DataProcessor.get_dataset_stats(texts, labels)
        print(f"Dataset: {stats['total_samples']} samples, {stats['positive_ratio']:.3f} positive ratio")
        
        return self.train_multiple_models(models, texts, labels, **kwargs)