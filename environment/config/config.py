import os

import yaml

from environment.config.runtime import PROJECT_ROOT, load_env

# Make keys from <root>/.env visible before anything reads the environment.
load_env()


def _load_config(config_path=None):
        # Resolve against the project root so VideoAgent runs from any cwd.
        if config_path is None:
            config_path = PROJECT_ROOT / 'environment' / 'config' / 'config.yml'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


config = _load_config()
