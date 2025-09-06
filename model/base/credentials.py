import os
import json
from typing import Dict, Optional, Any
from pathlib import Path


class CredentialManager:
    """Secure credential management for API-based models"""
    
    def __init__(self, credentials_file: Optional[str] = None):
        """
        Initialize credential manager
        
        Args:
            credentials_file: Path to credentials file (defaults to .env or config file)
        """
        self.credentials_file = credentials_file
        self._credentials = {}
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load credentials from multiple sources in priority order"""
        # Priority 1: Environment variables
        self._load_from_env()
        
        # Priority 2: Credentials file
        if self.credentials_file:
            self._load_from_file(self.credentials_file)
        
        # Priority 3: Default locations
        self._load_from_default_locations()
    
    def _load_from_env(self) -> None:
        """Load API credentials from environment variables"""
        env_mappings = {
            'ANTHROPIC_API_KEY': 'claude_api_key',
            'OPENAI_API_KEY': 'openai_api_key',
            'CLAUDE_API_KEY': 'claude_api_key',  # Alternative naming
            'HUGGINGFACE_TOKEN': 'hf_token',
            'HF_TOKEN': 'hf_token'  # Alternative naming
        }
        
        for env_var, cred_key in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._credentials[cred_key] = value
    
    def _load_from_file(self, filepath: str) -> None:
        """Load credentials from JSON or .env file"""
        try:
            path = Path(filepath)
            if not path.exists():
                return
            
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    file_creds = json.load(f)
                    self._credentials.update(file_creds)
            
            elif filepath.endswith('.env') or 'env' in filepath:
                # Simple .env parser
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            
                            # Map common env var names to our keys
                            if key in ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY']:
                                self._credentials['claude_api_key'] = value
                            elif key == 'OPENAI_API_KEY':
                                self._credentials['openai_api_key'] = value
                            elif key in ['HUGGINGFACE_TOKEN', 'HF_TOKEN']:
                                self._credentials['hf_token'] = value
                            else:
                                self._credentials[key.lower()] = value
        
        except Exception as e:
            print(f"Warning: Could not load credentials from {filepath}: {e}")
    
    def _load_from_default_locations(self) -> None:
        """Load from default credential locations"""
        # Common locations to check
        default_locations = [
            '.env',
            'credentials.json',
            os.path.expanduser('~/.goblin_credentials.json'),
            os.path.expanduser('~/.config/goblin/credentials.json'),
            '.goblin_credentials'
        ]
        
        for location in default_locations:
            if os.path.exists(location):
                self._load_from_file(location)
    
    def get_credential(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a credential by key
        
        Args:
            key: Credential key
            default: Default value if not found
            
        Returns:
            Credential value or default
        """
        return self._credentials.get(key, default)
    
    def has_credential(self, key: str) -> bool:
        """Check if credential exists"""
        return key in self._credentials and self._credentials[key] is not None
    
    def get_api_config(self, api_type: str) -> Dict[str, Any]:
        """
        Get API configuration for a specific API type
        
        Args:
            api_type: 'claude', 'openai', etc.
            
        Returns:
            Configuration dictionary
        """
        if api_type.lower() == 'claude':
            return {
                'api_key': self.get_credential('claude_api_key'),
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 1024,
                'temperature': 0.0
            }
        elif api_type.lower() == 'openai':
            return {
                'api_key': self.get_credential('openai_api_key'),
                'model': 'gpt-3.5-turbo',
                'max_tokens': 1024,
                'temperature': 0.0
            }
        elif api_type.lower() == 'huggingface':
            return {
                'token': self.get_credential('hf_token')
            }
        else:
            raise ValueError(f"Unknown API type: {api_type}")
    
    def validate_credentials(self, required_apis: list = None) -> Dict[str, bool]:
        """
        Validate that required credentials are available
        
        Args:
            required_apis: List of APIs to validate ['claude', 'openai', etc.]
            
        Returns:
            Dictionary mapping API names to validation status
        """
        if required_apis is None:
            required_apis = ['claude', 'openai']
        
        validation = {}
        
        for api in required_apis:
            try:
                config = self.get_api_config(api)
                # Check if primary credential exists
                primary_key = 'api_key' if 'api_key' in config else 'token'
                validation[api] = config.get(primary_key) is not None
            except ValueError:
                validation[api] = False
        
        return validation
    
    def create_example_env_file(self, filepath: str = '.env.example') -> None:
        """Create an example .env file for users"""
        example_content = """# API Credentials for Goblin Hate Speech Models
# Copy this file to .env and fill in your actual API keys

# Claude API (Anthropic)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI API (optional)
OPENAI_API_KEY=your_openai_api_key_here

# HuggingFace Token (optional, for private models)
HUGGINGFACE_TOKEN=your_hf_token_here

# Alternative naming (any of these will work):
# CLAUDE_API_KEY=your_anthropic_api_key_here
# HF_TOKEN=your_hf_token_here
"""
        
        with open(filepath, 'w') as f:
            f.write(example_content)
        
        print(f"Created example credential file: {filepath}")
        print("Copy this to .env and add your real API keys")
    
    def get_setup_instructions(self) -> str:
        """Get setup instructions for users"""
        return """
🔐 API Credential Setup Instructions:

1. Create a .env file in your project directory:
   - Copy .env.example to .env
   - Add your actual API keys

2. Or set environment variables:
   export ANTHROPIC_API_KEY="your_key_here"
   
3. Or create ~/.goblin_credentials.json:
   {
     "claude_api_key": "your_key_here",
     "openai_api_key": "your_key_here" 
   }

4. Keep credentials secure:
   - Add .env to .gitignore
   - Never commit API keys to git
   - Use environment variables in production

5. Validate setup:
   python -c "from model.base import CredentialManager; cm = CredentialManager(); print(cm.validate_credentials())"
"""


# Global credential manager instance
_credential_manager = None

def get_credential_manager() -> CredentialManager:
    """Get global credential manager instance"""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager