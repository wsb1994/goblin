#!/usr/bin/env python3
"""
Goblin Adapter - Bridge between hate speech models and goblin execution system

This script serves as the interface between goblin's subprocess execution
and our Python-based hate speech detection models.
"""

import sys
import json
import argparse
import traceback
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add model package to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.integration.model_registry import ModelRegistry


class GoblinAdapter:
    """Adapter to run hate speech models as goblin scripts"""
    
    def __init__(self):
        self.registry = ModelRegistry()
    
    def run_model(self, 
                  model_name: str, 
                  input_texts: List[str],
                  output_format: str = 'json') -> str:
        """
        Run a model on input texts and return results
        
        Args:
            model_name: Name of registered model
            input_texts: List of texts to classify
            output_format: Output format ('json', 'csv', 'scores')
            
        Returns:
            Formatted results string
        """
        try:
            # Load model from registry
            model = self.registry.get_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found or could not be loaded")
            
            # Run predictions
            probabilities = model.predict(input_texts)
            
            # Format results
            results = []
            for text, prob in zip(input_texts, probabilities):
                results.append({
                    'text': text,
                    'hate_speech_probability': float(prob),
                    'classification': 'hate' if prob > 0.5 else 'normal',
                    'model': model_name
                })
            
            # Return formatted output
            if output_format == 'json':
                return json.dumps({
                    'model': model_name,
                    'predictions': results,
                    'summary': {
                        'total_texts': len(input_texts),
                        'hate_predictions': sum(1 for r in results if r['classification'] == 'hate'),
                        'average_probability': sum(r['hate_speech_probability'] for r in results) / len(results)
                    }
                }, indent=2)
            
            elif output_format == 'csv':
                lines = ['text,probability,classification,model']
                for result in results:
                    lines.append(f'"{result["text"]}",{result["hate_speech_probability"]},{result["classification"]},{result["model"]}')
                return '\n'.join(lines)
            
            elif output_format == 'scores':
                return '\n'.join([str(r['hate_speech_probability']) for r in results])
            
            else:
                raise ValueError(f"Unknown output format: {output_format}")
        
        except Exception as e:
            error_msg = {
                'error': str(e),
                'model': model_name,
                'traceback': traceback.format_exc()
            }
            return json.dumps(error_msg, indent=2)
    
    def test_model(self, model_name: str) -> bool:
        """
        Test if a model can be loaded and used
        
        Args:
            model_name: Name of model to test
            
        Returns:
            True if model works, False otherwise
        """
        try:
            model = self.registry.get_model(model_name)
            if model is None:
                return False
            
            # Test with a simple prediction
            test_text = ["This is a test message"]
            predictions = model.predict(test_text)
            
            # Verify we get a reasonable prediction
            return len(predictions) == 1 and 0 <= predictions[0] <= 1
        
        except Exception as e:
            print(f"Model test failed: {e}", file=sys.stderr)
            return False
    
    def list_models(self) -> str:
        """List all available models"""
        models = self.registry.list_models()
        
        if not models:
            return json.dumps({'message': 'No models registered'})
        
        model_info = []
        for name, info in models.items():
            model_info.append({
                'name': name,
                'type': info['model_type'],
                'description': info['description'],
                'tags': info['tags']
            })
        
        return json.dumps({
            'models': model_info,
            'count': len(model_info)
        }, indent=2)


def main():
    """Main entry point for goblin script execution"""
    parser = argparse.ArgumentParser(description='Goblin Adapter for Hate Speech Models')
    parser.add_argument('model_name', help='Name of the model to run')
    parser.add_argument('--input', type=str, help='Input text or JSON string')
    parser.add_argument('--input-file', type=str, help='Path to input file')
    parser.add_argument('--output-format', choices=['json', 'csv', 'scores'], 
                       default='json', help='Output format')
    parser.add_argument('--test', action='store_true', help='Test model loading')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    adapter = GoblinAdapter()
    
    try:
        # Handle special commands
        if args.list_models:
            print(adapter.list_models())
            return
        
        if args.test:
            success = adapter.test_model(args.model_name)
            print('true' if success else 'false')
            return
        
        # Get input texts
        input_texts = []
        
        if args.input_file:
            # Load from file
            with open(args.input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Try to parse as JSON first
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        input_texts = [str(item) for item in data]
                    elif isinstance(data, dict) and 'texts' in data:
                        input_texts = [str(item) for item in data['texts']]
                    else:
                        input_texts = [str(data)]
                except json.JSONDecodeError:
                    # Treat as plain text, one line per text
                    input_texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        elif args.input:
            # Parse input argument
            try:
                # Try JSON first
                data = json.loads(args.input)
                if isinstance(data, list):
                    input_texts = [str(item) for item in data]
                else:
                    input_texts = [str(data)]
            except json.JSONDecodeError:
                # Treat as plain text
                input_texts = [args.input]
        
        else:
            # Read from stdin
            stdin_content = sys.stdin.read().strip()
            if stdin_content:
                try:
                    data = json.loads(stdin_content)
                    if isinstance(data, list):
                        input_texts = [str(item) for item in data]
                    elif isinstance(data, dict) and 'texts' in data:
                        input_texts = [str(item) for item in data['texts']]
                    else:
                        input_texts = [str(data)]
                except json.JSONDecodeError:
                    # Treat as plain text
                    input_texts = [stdin_content]
            else:
                # Default test input
                input_texts = ["This is a test message"]
        
        if not input_texts:
            raise ValueError("No input texts provided")
        
        # Run model
        result = adapter.run_model(args.model_name, input_texts, args.output_format)
        print(result)
        
    except Exception as e:
        error_output = json.dumps({
            'error': str(e),
            'traceback': traceback.format_exc()
        }, indent=2)
        print(error_output, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()