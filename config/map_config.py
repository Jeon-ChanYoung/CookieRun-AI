import yaml
from pathlib import Path

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value)) 
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __repr__(self):
        return f"Config({self.__dict__})"

def load_config():
    config_path = Path(__file__).parent / "oven_of_witch.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)