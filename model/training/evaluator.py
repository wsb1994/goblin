from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import pandas as pd

from ..base import BaseModel, DataProcessor


class ModelEvaluator:
    """Comprehensive evaluation utilities for hate speech models"""
    
    @staticmethod
    def evaluate_model(model: BaseModel, 
                      texts: List[str], 
                      labels: List[int],
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            texts: Test texts
            labels: Ground truth labels
            threshold: Classification threshold
            
        Returns:
            Comprehensive evaluation results
        """
        # Get predictions
        probabilities = model.predict(texts)
        binary_predictions = (probabilities >= threshold).astype(int)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(labels, binary_predictions),
            'precision': precision_score(labels, binary_predictions, zero_division=0),
            'recall': recall_score(labels, binary_predictions, zero_division=0),
            'f1': f1_score(labels, binary_predictions, zero_division=0),
            'auc_roc': roc_auc_score(labels, probabilities),
            'threshold': threshold
        }
        
        # Confusion matrix
        cm = confusion_matrix(labels, binary_predictions)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]), 
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        # Classification report
        report = classification_report(labels, binary_predictions, output_dict=True)
        metrics['classification_report'] = report
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(labels, probabilities)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # Precision-Recall curve data
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        metrics['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
        
        return metrics
    
    @staticmethod
    def compare_models(models: List[BaseModel],
                      texts: List[str],
                      labels: List[int],
                      model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models on the same test set
        
        Args:
            models: List of trained models
            texts: Test texts
            labels: Ground truth labels
            model_names: Optional custom names for models
            
        Returns:
            DataFrame with comparison metrics
        """
        if model_names is None:
            model_names = [model.config.model_name for model in models]
            
        results = []
        
        for model, name in zip(models, model_names):
            metrics = ModelEvaluator.evaluate_model(model, texts, labels)
            
            result = {
                'Model': name,
                'Model Type': model.config.model_type,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'AUC-ROC': metrics['auc_roc'],
                'True Positives': metrics['confusion_matrix']['tp'],
                'False Positives': metrics['confusion_matrix']['fp'],
                'True Negatives': metrics['confusion_matrix']['tn'],
                'False Negatives': metrics['confusion_matrix']['fn']
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def find_optimal_threshold(model: BaseModel,
                             texts: List[str],
                             labels: List[int],
                             metric: str = 'f1') -> Dict[str, float]:
        """
        Find optimal classification threshold
        
        Args:
            model: Trained model
            texts: Validation texts
            labels: Ground truth labels
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Dictionary with optimal threshold and corresponding metrics
        """
        probabilities = model.predict(texts)
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_threshold = 0.5
        best_score = 0.0
        
        results = []
        
        for threshold in thresholds:
            binary_pred = (probabilities >= threshold).astype(int)
            
            scores = {
                'threshold': threshold,
                'f1': f1_score(labels, binary_pred, zero_division=0),
                'precision': precision_score(labels, binary_pred, zero_division=0),
                'recall': recall_score(labels, binary_pred, zero_division=0)
            }
            
            if scores[metric] > best_score:
                best_score = scores[metric]
                best_threshold = threshold
            
            results.append(scores)
        
        return {
            'optimal_threshold': best_threshold,
            'optimal_score': best_score,
            'metric_optimized': metric,
            'threshold_analysis': results
        }
    
    @staticmethod
    def plot_roc_curves(models: List[BaseModel],
                       texts: List[str],
                       labels: List[int],
                       model_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
        """
        Plot ROC curves for multiple models
        
        Args:
            models: List of trained models
            texts: Test texts
            labels: Ground truth labels
            model_names: Optional custom names
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        if model_names is None:
            model_names = [model.config.model_name for model in models]
        
        for model, name in zip(models, model_names):
            probabilities = model.predict(texts)
            fpr, tpr, _ = roc_curve(labels, probabilities)
            auc = roc_auc_score(labels, probabilities)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Hate Speech Detection Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    @staticmethod
    def analyze_errors(model: BaseModel,
                      texts: List[str],
                      labels: List[int],
                      num_examples: int = 10) -> Dict[str, List[Dict]]:
        """
        Analyze model errors with examples
        
        Args:
            model: Trained model
            texts: Test texts
            labels: Ground truth labels
            num_examples: Number of examples to show per error type
            
        Returns:
            Dictionary with false positive and false negative examples
        """
        probabilities = model.predict(texts)
        binary_predictions = (probabilities >= 0.5).astype(int)
        
        # Find errors
        false_positives = []
        false_negatives = []
        
        for i, (text, true_label, pred_label, prob) in enumerate(
            zip(texts, labels, binary_predictions, probabilities)
        ):
            if true_label == 0 and pred_label == 1:  # False positive
                false_positives.append({
                    'text': text,
                    'probability': float(prob),
                    'index': i
                })
            elif true_label == 1 and pred_label == 0:  # False negative
                false_negatives.append({
                    'text': text,
                    'probability': float(prob),
                    'index': i
                })
        
        # Sort by confidence (probability distance from 0.5)
        false_positives.sort(key=lambda x: x['probability'], reverse=True)
        false_negatives.sort(key=lambda x: x['probability'])
        
        return {
            'false_positives': false_positives[:num_examples],
            'false_negatives': false_negatives[:num_examples],
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives)
        }