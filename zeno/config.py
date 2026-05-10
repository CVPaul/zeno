from __future__ import annotations

import json
from pathlib import Path


def default_config_path() -> Path:
    return Path.cwd() / ".zeno" / "config.json"


class ConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_config_path()

    def model_for_backend(self, backend: str) -> str | None:
        data = self._read()
        models = data.get("models")
        if not isinstance(models, dict):
            return None
        model = models.get(backend)
        return model if isinstance(model, str) and model.strip() else None

    def save_model(self, backend: str, model: str) -> None:
        data = self._read()
        models = data.get("models")
        if not isinstance(models, dict):
            models = {}
        models[backend] = model
        data["models"] = models
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}
