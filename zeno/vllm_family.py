from __future__ import annotations

import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def default_backend() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "vllm-mlx"
    return "vllm"


def default_model_name(model: str | None = None, backend: str | None = None) -> str:
    if model:
        return model
    selected_backend = backend or default_backend()
    if selected_backend == "vllm-mlx":
        return DEFAULT_MLX_MODEL
    return DEFAULT_VLLM_MODEL


@dataclass(frozen=True)
class VllmFamilyManager:
    model: str
    backend: str
    base_url: str = "http://localhost:8000"
    startup_timeout: float = 120.0

    def ensure_ready(self) -> None:
        if self._is_running():
            return
        command = self._command()
        self._require_command(command[0])
        self._start_service(command)
        self._wait_until_running()

    def openai_base_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1"

    def _command(self) -> list[str]:
        if self.backend == "vllm-mlx":
            return ["vllm-mlx", "serve", self.model, "--port", self._port()]
        if self.backend == "vllm":
            return ["vllm", "serve", self.model, "--port", self._port()]
        raise RuntimeError(f"Unsupported backend: {self.backend}")

    def _require_command(self, command: str) -> None:
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} command not found. Install the {self.backend} backend first.")

    def _start_service(self, command: list[str]) -> None:
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"failed to start {self.backend} service") from exc

    def _wait_until_running(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._is_running():
                return
            time.sleep(0.5)
        raise RuntimeError(f"{self.backend} service did not become ready")

    def _is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.openai_base_url()}/models", timeout=1.0):
                return True
        except urllib.error.URLError:
            return False

    def _port(self) -> str:
        return self.base_url.rstrip("/").rsplit(":", 1)[-1]
