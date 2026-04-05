from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
