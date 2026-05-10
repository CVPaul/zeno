from __future__ import annotations

import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .llmfit import recommend_model
from .logging import VerboseLogger


DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def default_backend() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "vllm-mlx"
    return "vllm"


def default_model_name(model: str | None = None, backend: str | None = None, log: VerboseLogger | None = None) -> str:
    if model:
        if log is not None:
            log(f"using explicit model: {model}")
        return model
    selected_backend = backend or default_backend()
    recommendation = recommend_model(selected_backend, log=log)
    if recommendation is not None:
        return recommendation.model
    if selected_backend == "vllm-mlx":
        if log is not None:
            log(f"using built-in default model for vllm-mlx: {DEFAULT_MLX_MODEL}")
        return DEFAULT_MLX_MODEL
    if log is not None:
        log(f"using built-in default model for vllm: {DEFAULT_VLLM_MODEL}")
    return DEFAULT_VLLM_MODEL


@dataclass(frozen=True)
class VllmFamilyManager:
    model: str
    backend: str
    base_url: str = "http://localhost:8000"
    startup_timeout: float = 120.0
    log: VerboseLogger | None = None
    log_dir: Path | None = None
    device: str | None = None

    def ensure_ready(self) -> None:
        self._log(f"checking {self.backend} server at {self.openai_base_url()}/models")
        if self._is_running():
            self._log(f"{self.backend} server is already running")
            return
        command = self._command()
        self._require_command(command[0])
        self._log(f"starting {self.backend} service for model {self.model}")
        self._log(f"command: {' '.join(command)}")
        process = self._start_service(command)
        self._wait_until_running(process)

    def openai_base_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1"

    def _command(self) -> list[str]:
        if self.backend == "vllm-mlx":
            return ["vllm-mlx", "serve", self.model, "--port", self._port()]
        if self.backend == "vllm":
            command = ["vllm", "serve", self.model, "--port", self._port()]
            if self.device is not None:
                command.extend(["--device", self.device])
            return command
        raise RuntimeError(f"Unsupported backend: {self.backend}")

    def _require_command(self, command: str) -> None:
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} command not found. Install the {self.backend} backend first.")
        self._log(f"found backend command: {command}")

    def _start_service(self, command: list[str]) -> subprocess.Popen[bytes]:
        log_path = self._backend_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("ab")
        header = f"\n--- zeno starting {self.backend}: {' '.join(command)} ---\n".encode("utf-8")
        log_file.write(header)
        log_file.flush()
        self._log(f"backend log: {log_path}")
        try:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=None if self.log is not None else subprocess.STDOUT,
                start_new_session=True,
            )
            log_file.close()
            return process
        except OSError as exc:
            log_file.close()
            raise RuntimeError(f"failed to start {self.backend} service") from exc

    def _wait_until_running(self, process: subprocess.Popen[bytes] | None = None) -> None:
        self._log(f"waiting up to {self.startup_timeout:.0f}s for {self.openai_base_url()}/models")
        deadline = time.monotonic() + self.startup_timeout
        next_log = time.monotonic()
        while time.monotonic() < deadline:
            if self._is_running():
                self._log(f"{self.backend} service is ready")
                return
            if process is not None and process.poll() is not None:
                raise RuntimeError(self._startup_failure_message(f"{self.backend} service exited with code {process.returncode}"))
            now = time.monotonic()
            if self.log is not None and now >= next_log:
                elapsed = self.startup_timeout - (deadline - now)
                self._log(f"still waiting for {self.backend} service ({elapsed:.0f}s elapsed)")
                next_log = now + 5.0
            time.sleep(0.5)
        raise RuntimeError(self._startup_failure_message(f"{self.backend} service did not become ready"))

    def _is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.openai_base_url()}/models", timeout=1.0):
                return True
        except urllib.error.URLError:
            return False

    def _port(self) -> str:
        return self.base_url.rstrip("/").rsplit(":", 1)[-1]

    def _log(self, message: str) -> None:
        if self.log is not None:
            self.log(message)

    def _backend_log_path(self) -> Path:
        root = self.log_dir or Path.cwd() / ".zeno" / "logs"
        safe_model = self.model.replace("/", "_").replace(":", "_")
        return root / f"{self.backend}-{safe_model}.log"

    def _startup_failure_message(self, headline: str) -> str:
        log_path = self._backend_log_path()
        message = f"{headline}. See backend log: {log_path}"
        tail = self._log_tail(log_path)
        if tail:
            message = f"{message}\nLast backend log lines:\n{tail}"
        return message

    def _log_tail(self, path: Path, max_lines: int = 20) -> str:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-max_lines:])
