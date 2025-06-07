"""Settings management for PyHub LLM."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ImportError:
    import toml as tomllib  # Python < 3.11 fallback

from dotenv import load_dotenv

from .exceptions import ConfigurationError


class Settings:
    """Settings management for PyHub LLM."""
    
    def __init__(self, config_file: Optional[str] = None):
        self._config: Dict[str, Any] = {}
        
        # Load from .env file
        load_dotenv()
        
        # Load from config file if provided
        if config_file:
            self._load_config_file(config_file)
        else:
            # Try to find pyproject.toml
            self._load_pyproject_toml()
    
    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from a TOML file."""
        path = Path(config_file)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {config_file}")
        
        try:
            with open(path, 'rb') as f:
                data = tomllib.load(f)
            
            # Extract pyhub-llm specific config
            self._config = data.get('tool', {}).get('pyhub-llm', {})
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}")
    
    def _load_pyproject_toml(self) -> None:
        """Try to load configuration from pyproject.toml."""
        # Look for pyproject.toml in current and parent directories
        current = Path.cwd()
        for path in [current] + list(current.parents):
            pyproject = path / "pyproject.toml"
            if pyproject.exists():
                try:
                    with open(pyproject, 'rb') as f:
                        data = tomllib.load(f)
                    
                    # Extract pyhub-llm specific config
                    self._config = data.get('tool', {}).get('pyhub-llm', {})
                    break
                except:
                    # Ignore errors and continue searching
                    pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Priority order:
        1. Environment variable (PYHUB_LLM_<KEY>)
        2. Config file
        3. Default value
        """
        # Check environment variable
        env_key = f"PYHUB_LLM_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        
        # Check config
        return self._config.get(key, default)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider.
        
        Priority order:
        1. Provider-specific environment variable (e.g., OPENAI_API_KEY)
        2. Generic environment variable (PYHUB_LLM_<PROVIDER>_API_KEY)
        3. Config file
        """
        provider_upper = provider.upper()
        
        # Common environment variable names
        common_env_vars = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'upstage': 'UPSTAGE_API_KEY',
        }
        
        # Check common environment variable
        if provider.lower() in common_env_vars:
            api_key = os.getenv(common_env_vars[provider.lower()])
            if api_key:
                return api_key
        
        # Check generic environment variable
        env_key = f"PYHUB_LLM_{provider_upper}_API_KEY"
        api_key = os.getenv(env_key)
        if api_key:
            return api_key
        
        # Check config file
        provider_config = self._config.get(provider.lower(), {})
        if isinstance(provider_config, dict):
            return provider_config.get('api_key')
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value (runtime only, not persisted)."""
        self._config[key] = value
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration with a dictionary."""
        self._config.update(config)