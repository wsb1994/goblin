#!/usr/bin/env python3
"""
Model comparison script for goblin execution
"""

import sys
import json
import os
from pathlib import Path

# Add model package to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from model.training import ModelEvaluator


def main():
    """Compare results from different models"""
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Need at least 2 model results to compare'}))
        sys.exit(1)
    
    try:
        # Parse input results from previous goblin steps
        model_results = []
        for arg in sys.argv[1:]:
            try:
                result = json.loads(arg)
                model_results.append(result)
            except json.JSONDecodeError:
                # Skip non-JSON arguments
                continue
        
        if len(model_results) < 2:
            print(json.dumps({'error': 'Could not parse enough model results'}))
            sys.exit(1)
        
        # Extract predictions for comparison
        comparison_data = []
        for result in model_results:
            if 'predictions' in result:
                model_name = result.get('model', 'unknown')
                predictions = result['predictions']
                
                comparison_data.append({
                    'model': model_name,
                    'predictions': [p['hate_speech_probability'] for p in predictions],
                    'classifications': [p['classification'] for p in predictions],
                    'summary': result.get('summary', {})
                })
        
        # Calculate comparison metrics
        comparison = {
            'models_compared': len(comparison_data),
            'model_names': [data['model'] for data in comparison_data],
            'results': comparison_data,
            'analysis': {
                'average_hate_rates': {},
                'prediction_correlations': {},
                'agreement_rates': {}
            }
        }
        
        # Add analysis
        for data in comparison_data:
            model_name = data['model']
            hate_rate = sum(1 for c in data['classifications'] if c == 'hate') / len(data['classifications'])
            avg_prob = sum(data['predictions']) / len(data['predictions'])
            
            comparison['analysis']['average_hate_rates'][model_name] = {
                'hate_classification_rate': hate_rate,
                'average_probability': avg_prob
            }
        
        # Calculate agreement between models
        if len(comparison_data) == 2:
            model1_classes = comparison_data[0]['classifications']
            model2_classes = comparison_data[1]['classifications']
            
            if len(model1_classes) == len(model2_classes):
                agreements = sum(1 for c1, c2 in zip(model1_classes, model2_classes) if c1 == c2)
                agreement_rate = agreements / len(model1_classes)
                
                comparison['analysis']['agreement_rates']['overall'] = agreement_rate
        
        print(json.dumps(comparison, indent=2))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'component': 'model_comparison'
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()