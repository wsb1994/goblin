import numpy as np
import json
import time
from typing import List, Dict, Any, Optional
import re

from ..base import APIBaseModel, ModelConfig, DataProcessor, get_credential_manager

# Try to import anthropic, but make it optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ClaudeHateSpeechModel(APIBaseModel):
    """Claude API-based hate speech detection model"""
    
    def __init__(self, config: ModelConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required for this model. "
                "Install with: pip install anthropic"
            )
            
        super().__init__(config)
        
        # Configuration - try credential manager first, then config
        credential_manager = get_credential_manager()
        
        self.api_key = config.get('api_key') or credential_manager.get_credential('claude_api_key')
        self.model = config.get('claude_model', 'claude-3-haiku-20240307')
        self.max_tokens = config.get('max_tokens', 1024)
        self.temperature = config.get('temperature', 0.0)
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)  # seconds between requests
        
        if not self.api_key:
            instructions = credential_manager.get_setup_instructions()
            raise ValueError(f"Claude API key not found. Please set up credentials:\n{instructions}")
        
        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Prompt templates
        self.system_prompt = """You are an expert content moderator specializing in hate speech detection. Your task is to analyze text content and determine if it contains hate speech.

Hate speech is defined as content that:
- Attacks or demeans individuals or groups based on protected characteristics (race, ethnicity, religion, gender, sexual orientation, nationality, disability, etc.)
- Promotes violence or discrimination against protected groups  
- Uses slurs, derogatory terms, or dehumanizing language
- Incites hatred or hostility toward protected groups

You must respond with ONLY a JSON object in this exact format:
{"probability": X.XX, "reasoning": "brief explanation"}

Where probability is a number between 0.0 and 1.0:
- 0.0 = definitely not hate speech
- 1.0 = definitely hate speech
- Values in between indicate confidence level"""

        self.few_shot_examples = []  # Will be populated during training
    
    def _create_prompt(self, text: str, use_few_shot: bool = True) -> str:
        """Create prompt for hate speech classification"""
        prompt_parts = []
        
        # Add few-shot examples if available and requested
        if use_few_shot and hasattr(self, '_few_shot_examples') and self._few_shot_examples:
            prompt_parts.append("Here are some examples:")
            
            for example_text, label in self._few_shot_examples[:5]:  # Use max 5 examples
                label_text = "hate speech" if label == 1 else "not hate speech"
                prob = 0.9 if label == 1 else 0.1
                prompt_parts.append(f'Text: "{example_text}"')
                prompt_parts.append(f'Classification: {{"probability": {prob}, "reasoning": "This is {label_text}"}}')
                prompt_parts.append("")
        
        # Add the text to classify
        prompt_parts.append(f'Now classify this text:\n"{text}"')
        
        return "\n".join(prompt_parts)
    
    def _call_claude_api(self, text: str) -> Dict[str, Any]:
        """Make API call to Claude"""
        try:
            prompt = self._create_prompt(text)
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract JSON from response
            response_text = message.content[0].text
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^}]*"probability"[^}]*\}', response_text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if 'probability' in result:
                        return {
                            'probability': float(result['probability']),
                            'reasoning': result.get('reasoning', ''),
                            'raw_response': response_text
                        }
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to extract probability from text
            prob_match = re.search(r'"probability":\s*([\d.]+)', response_text)
            if prob_match:
                return {
                    'probability': float(prob_match.group(1)),
                    'reasoning': 'Extracted from text',
                    'raw_response': response_text
                }
            
            # If we can't parse, return neutral
            return {
                'probability': 0.5,
                'reasoning': 'Could not parse API response',
                'raw_response': response_text
            }
            
        except Exception as e:
            print(f"API call failed: {e}")
            # Return neutral probability on error
            return {
                'probability': 0.5,
                'reasoning': f'API error: {str(e)}',
                'raw_response': ''
            }
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Generate probability predictions using Claude API"""
        if not self.is_trained:
            raise RuntimeError("Model must be 'trained' (initialized) before making predictions")
        
        probabilities = []
        
        print(f"Making {len(texts)} API calls to Claude...")
        
        for i, text in enumerate(texts):
            if i > 0:
                time.sleep(self.rate_limit_delay)  # Rate limiting
            
            if i % 10 == 0 and i > 0:
                print(f"Processed {i}/{len(texts)} texts...")
            
            result = self._call_claude_api(text)
            probabilities.append(result['probability'])
        
        return np.array(probabilities)
    
    def predict_single_with_reasoning(self, text: str) -> Dict[str, Any]:
        """Get prediction with reasoning for a single text"""
        result = self._call_claude_api(text)
        return result
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Get predictions
        probabilities = self.predict(texts)
        binary_predictions = (probabilities > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, binary_predictions),
            'precision': precision_score(labels, binary_predictions, zero_division=0),
            'recall': recall_score(labels, binary_predictions, zero_division=0),
            'f1': f1_score(labels, binary_predictions, zero_division=0),
            'auc_roc': roc_auc_score(labels, probabilities)
        }
        
        return metrics
    
    def get_api_usage_info(self) -> Dict[str, Any]:
        """Get information about API usage and costs"""
        return {
            'model': self.model,
            'rate_limit_delay': self.rate_limit_delay,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'few_shot_examples': len(getattr(self, '_few_shot_examples', []))
        }