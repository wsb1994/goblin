#!/usr/bin/env python3
"""
Multi-model comparison script for goblin execution
"""

import sys
import json
import os
from pathlib import Path


def main():
    """Compare results from multiple models"""
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
                continue
        
        if len(model_results) < 2:
            print(json.dumps({'error': 'Could not parse enough model results'}))
            sys.exit(1)
        
        # Build comprehensive comparison
        comparison = {
            'comparison_type': 'multi_model',
            'models_count': len(model_results),
            'models': [],
            'summary_statistics': {},
            'performance_ranking': []
        }
        
        # Process each model's results
        for result in model_results:
            if 'predictions' in result:
                model_name = result.get('model', 'unknown')
                predictions = result['predictions']
                summary = result.get('summary', {})
                
                model_data = {
                    'name': model_name,
                    'total_predictions': len(predictions),
                    'hate_predictions': sum(1 for p in predictions if p['classification'] == 'hate'),
                    'average_probability': sum(p['hate_speech_probability'] for p in predictions) / len(predictions),
                    'confidence_distribution': {
                        'high_confidence_hate': sum(1 for p in predictions if p['hate_speech_probability'] > 0.8),
                        'medium_confidence_hate': sum(1 for p in predictions if 0.5 < p['hate_speech_probability'] <= 0.8),
                        'low_confidence_normal': sum(1 for p in predictions if 0.2 <= p['hate_speech_probability'] <= 0.5),
                        'high_confidence_normal': sum(1 for p in predictions if p['hate_speech_probability'] < 0.2)
                    }
                }
                
                comparison['models'].append(model_data)
        
        # Calculate rankings and statistics
        if comparison['models']:
            # Rank by different criteria
            by_hate_rate = sorted(comparison['models'], 
                                key=lambda x: x['hate_predictions'] / x['total_predictions'], 
                                reverse=True)
            by_avg_prob = sorted(comparison['models'], 
                               key=lambda x: x['average_probability'], 
                               reverse=True)
            
            comparison['performance_ranking'] = {
                'by_hate_detection_rate': [m['name'] for m in by_hate_rate],
                'by_average_probability': [m['name'] for m in by_avg_prob]
            }
            
            # Summary statistics
            hate_rates = [m['hate_predictions'] / m['total_predictions'] for m in comparison['models']]
            avg_probs = [m['average_probability'] for m in comparison['models']]
            
            comparison['summary_statistics'] = {
                'hate_rate_range': {
                    'min': min(hate_rates),
                    'max': max(hate_rates),
                    'spread': max(hate_rates) - min(hate_rates)
                },
                'probability_range': {
                    'min': min(avg_probs),
                    'max': max(avg_probs),
                    'spread': max(avg_probs) - min(avg_probs)
                },
                'model_agreement': 'high' if max(hate_rates) - min(hate_rates) < 0.1 else 
                                 'medium' if max(hate_rates) - min(hate_rates) < 0.3 else 'low'
            }
        
        print(json.dumps(comparison, indent=2))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'component': 'multi_model_comparison'
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()