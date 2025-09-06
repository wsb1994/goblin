from typing import Dict, List, Any, Optional, Union
import json
import os
from pathlib import Path
import importlib.util

from ..base import BaseModel
from ..basic import BagOfWordsModel
from ..pipeline import ComposableModel, create_composable_model

# Try to import sophisticated models
try:
    from ..sophisticated import HuggingFaceLightGBMModel
    SOPHISTICATED_AVAILABLE = True
except ImportError:
    SOPHISTICATED_AVAILABLE = False

try:
    from ..api import ClaudeHateSpeechModel
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


class ModelRegistry:
    """Registry for discovering, loading, and managing hate speech models"""
    
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = registry_path
        self.models: Dict[str, Dict[str, Any]] = {}
        self.loaded_models: Dict[str, BaseModel] = {}
        
        # Load existing registry
        self.load_registry()
    
    def load_registry(self) -> None:
        """Load model registry from file"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.models = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load registry from {self.registry_path}: {e}")
                self.models = {}
        else:
            self.models = {}
    
    def save_registry(self) -> None:
        """Save model registry to file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.models, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save registry to {self.registry_path}: {e}")
    
    def register_model(self, 
                      model_name: str,
                      model_type: str,
                      model_path: str,
                      config: Dict[str, Any] = None,
                      description: str = "",
                      tags: List[str] = None) -> None:
        """
        Register a model in the registry
        
        Args:
            model_name: Unique model identifier
            model_type: Type of model ('basic', 'sophisticated', 'api', 'composable')
            model_path: Path to saved model file
            config: Model configuration
            description: Human-readable description
            tags: List of tags for filtering
        """
        self.models[model_name] = {
            'model_type': model_type,
            'model_path': model_path,
            'config': config or {},
            'description': description,
            'tags': tags or [],
            'registered_at': str(Path(model_path).stat().st_mtime) if os.path.exists(model_path) else None
        }
        
        self.save_registry()
    
    def list_models(self, 
                   model_type: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        List registered models with optional filtering
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags (any match)
            
        Returns:
            Dictionary of matching models
        """
        filtered_models = {}
        
        for name, info in self.models.items():
            # Filter by type
            if model_type and info['model_type'] != model_type:
                continue
            
            # Filter by tags
            if tags and not any(tag in info['tags'] for tag in tags):
                continue
            
            filtered_models[name] = info
        
        return filtered_models
    
    def get_model(self, model_name: str, load_if_needed: bool = True) -> Optional[BaseModel]:
        """
        Get a model instance
        
        Args:
            model_name: Name of registered model
            load_if_needed: Whether to load model if not already loaded
            
        Returns:
            Model instance or None if not found
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if not load_if_needed or model_name not in self.models:
            return None
        
        try:
            model = self.load_model(model_name)
            self.loaded_models[model_name] = model
            return model
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return None
    
    def load_model(self, model_name: str) -> BaseModel:
        """
        Load a model from the registry
        
        Args:
            model_name: Name of registered model
            
        Returns:
            Loaded model instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_info = self.models[model_name]
        model_type = model_info['model_type']
        model_path = model_info['model_path']
        config = model_info['config']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model instance based on type
        if model_type == 'basic':
            from ..base import ModelConfig
            model_config = ModelConfig(**config)
            model = BagOfWordsModel(model_config)
            
        elif model_type == 'sophisticated' and SOPHISTICATED_AVAILABLE:
            from ..base import ModelConfig
            model_config = ModelConfig(**config)
            model = HuggingFaceLightGBMModel(model_config)
            
        elif model_type == 'api' and API_AVAILABLE:
            from ..base import ModelConfig
            model_config = ModelConfig(**config)
            model = ClaudeHateSpeechModel(model_config)
            
        elif model_type == 'composable':
            from ..base import ModelConfig
            model_config = ModelConfig(**config)
            model = ComposableModel(model_config)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load the trained model
        model.load(model_path)
        
        return model
    
    def auto_discover_models(self, search_paths: List[str] = None) -> int:
        """
        Auto-discover models in specified directories
        
        Args:
            search_paths: List of directories to search
            
        Returns:
            Number of models discovered
        """
        if search_paths is None:
            search_paths = [
                'model_outputs',
                'training_outputs', 
                'models',
                '.'
            ]
        
        discovered = 0
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            # Look for model files
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.pkl') or file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        
                        try:
                            # Try to determine model type from file structure
                            if self._try_register_discovered_model(file_path):
                                discovered += 1
                        except Exception:
                            continue  # Skip files that can't be loaded
        
        if discovered > 0:
            self.save_registry()
        
        return discovered
    
    def _try_register_discovered_model(self, file_path: str) -> bool:
        """Try to register a discovered model file"""
        file_name = Path(file_path).stem
        
        # Skip if already registered
        if file_name in self.models:
            return False
        
        # Try to determine model type
        if '_results.json' in file_path:
            return False  # Skip results files
        
        if file_path.endswith('.json'):
            # Could be a composable model
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'dataset_type' in data and 'feature_type' in data:
                        self.register_model(
                            model_name=file_name,
                            model_type='composable',
                            model_path=file_path,
                            config=data.get('config', {}),
                            description=f"Auto-discovered composable model: {data.get('dataset_type', 'unknown')} + {data.get('feature_type', 'unknown')} + {data.get('architecture_type', 'unknown')}",
                            tags=['auto-discovered', 'composable']
                        )
                        return True
            except:
                pass
        
        elif file_path.endswith('.pkl'):
            # Could be any model type - try to load and inspect
            self.register_model(
                model_name=file_name,
                model_type='unknown',
                model_path=file_path,
                config={},
                description=f"Auto-discovered model from {file_path}",
                tags=['auto-discovered']
            )
            return True
        
        return False
    
    def create_model_variants(self) -> List[str]:
        """
        Create common model variants for comparison
        
        Returns:
            List of created model names
        """
        variants = []
        
        # Basic variants
        basic_configs = [
            {
                'name': 'basic_tfidf_logistic',
                'dataset': 'berkeley',
                'features': 'tfidf',
                'architecture': 'logistic',
                'description': 'Basic TF-IDF + Logistic Regression'
            },
            {
                'name': 'basic_bow_svm',
                'dataset': 'berkeley', 
                'features': 'bow',
                'architecture': 'svm',
                'description': 'Bag of Words + SVM'
            },
            {
                'name': 'basic_tfidf_rf',
                'dataset': 'berkeley',
                'features': 'tfidf', 
                'architecture': 'random_forest',
                'description': 'TF-IDF + Random Forest'
            }
        ]
        
        for config in basic_configs:
            try:
                model = create_composable_model(
                    dataset_type=config['dataset'],
                    feature_type=config['features'],
                    architecture_type=config['architecture'],
                    model_name=config['name'],
                    dataset_config={'sample_size': 2000},
                    feature_config={'max_features': 5000},
                    architecture_config={}
                )
                
                # Quick training on small dataset
                model.train()
                
                # Save model
                model_path = f"model_outputs/{config['name']}.json"
                os.makedirs('model_outputs', exist_ok=True)
                model.save(model_path)
                
                # Register
                self.register_model(
                    model_name=config['name'],
                    model_type='composable',
                    model_path=model_path,
                    config=model.config.__dict__,
                    description=config['description'],
                    tags=['basic', 'trained', config['features'], config['architecture']]
                )
                
                variants.append(config['name'])
                
            except Exception as e:
                print(f"Failed to create variant {config['name']}: {e}")
        
        return variants
    
    def get_model_for_goblin(self, model_name: str) -> Dict[str, Any]:
        """
        Get model information formatted for goblin integration
        
        Args:
            model_name: Name of registered model
            
        Returns:
            Dictionary with goblin script configuration
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_info = self.models[model_name]
        
        return {
            'name': model_name,
            'command': f'python -m model.integration.goblin_adapter {model_name}',
            'timeout': 30000,  # 30 seconds
            'test_command': f'python -m model.integration.goblin_adapter {model_name} --test',
            'require_test': False,
            'model_type': model_info['model_type'],
            'description': model_info['description'],
            'tags': model_info['tags']
        }