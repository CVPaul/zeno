from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaManager:
    model: str
    base_url: str = "http://localhost:11434"
    startup_timeout: float = 15.0

    def ensure_ready(self) -> None:
        self._require_cli()
        if not self._is_running():
            self._start_service()
            self._wait_until_running()
        if not self._has_model():
            self._pull_model()

    def _require_cli(self) -> None:
        if shutil.which("ollama") is None:
            raise RuntimeError("ollama CLI not found. Install Ollama first: https://ollama.com/download")

    def _is_running(self) -> bool:
        try:
            self._get_json("/api/tags", timeout=1.0)
            return True
        except RuntimeError:
            return False

    def _start_service(self) -> None:
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError("failed to start Ollama service with `ollama serve`") from exc

    def _wait_until_running(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._is_running():
                return
            time.sleep(0.25)
        raise RuntimeError("Ollama service did not become ready after `ollama serve`")

    def _has_model(self) -> bool:
        data = self._get_json("/api/tags", timeout=5.0)
        models = data.get("models", [])
        if not isinstance(models, list):
            return False
        names = {model.get("name") for model in models if isinstance(model, dict)}
        return self.model in names or f"{self.model}:latest" in names

    def _pull_model(self) -> None:
        print(f"Pulling Ollama model: {self.model}")
        try:
            self._post_json("/api/pull", {"name": self.model, "stream": False}, timeout=None)
        except RuntimeError as exc:
            raise RuntimeError(f"failed to pull Ollama model: {self.model}") from exc

    def _get_json(self, path: str, timeout: float) -> dict[str, object]:
        try:
            with urllib.request.urlopen(f"{self.base_url.rstrip('/')}{path}", timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama service is not reachable at {self.base_url}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama service returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Ollama service returned invalid response")
        return data

    def _post_json(self, path: str, payload: dict[str, object], timeout: float | None) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama service is not reachable at {self.base_url}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama service returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Ollama service returned invalid response")
        return data
