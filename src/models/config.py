from __future__ import annotations

import yaml
from pathlib import Path

from .client import ModelConfig


def load_model_config(model_key: str, config_path: str = "configs/models.yaml") -> ModelConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"설정 파일 파싱 실패 ({config_path}): {e}") from e

    models = data.get("models", {})
    if model_key not in models:
        raise KeyError(f"'{model_key}' 설정이 없습니다.")

    m = models[model_key]
    return ModelConfig(
        provider=m["provider"],
        model_name=m["model_name"],
        temperature=m.get("temperature", 0.2),
        max_tokens=m.get("max_tokens", 1024),
        timeout=m.get("timeout", 120),
        base_url=m.get("base_url"),
        api_key=m.get("api_key"),
        context_window=m.get("context_window", 8192),
        context_safety_margin=m.get("context_safety_margin", 256),
        reasoning_effort=m.get("reasoning_effort"),
    )
