from pathlib import Path
import os
import yaml

class Config:
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self):
        config_path = Path(__file__).parent.parent / 'config.yml'
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {
            'api': {
                'retries': 3,
                'timeout': 10,
                'rate_limit': 100
            },
            'cache': {
                'ttl': 3600,
                'max_size': 1000
            },
            'logging': {
                'level': 'INFO',
                'file': 'financial_analyzer.log'
            }
        }
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
