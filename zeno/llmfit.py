from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

from .logging import VerboseLogger


@dataclass(frozen=True)
class LlmfitRecommendation:
    model: str
    source: str


def recommend_model(backend: str, use_case: str = "coding", log: VerboseLogger | None = None) -> LlmfitRecommendation | None:
    if shutil.which("llmfit") is None:
        if log is not None:
            log("llmfit not found; using built-in default model")
        return None
    runtime = "mlx" if backend == "vllm-mlx" else "vllm"
    command = _recommend_command(runtime, use_case, 1)
    if log is not None:
        log(f"running model recommendation: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except OSError as exc:
        if log is not None:
            log(f"llmfit failed to start: {exc}; using built-in default model")
        return None
    except subprocess.CalledProcessError as exc:
        if log is not None:
            detail = exc.stderr.strip() if exc.stderr else str(exc)
            log(f"llmfit recommendation failed: {detail}; using built-in default model")
        return None
    model = parse_recommended_model(result.stdout)
    if model is None:
        if log is not None:
            log("llmfit returned no usable model; using built-in default model")
        return None
    if log is not None:
        log(f"llmfit recommended model: {model}")
    return LlmfitRecommendation(model=model, source="llmfit")


def recommend_models(backend: str, limit: int = 5, use_case: str = "coding", log: VerboseLogger | None = None) -> list[LlmfitRecommendation]:
    if shutil.which("llmfit") is None:
        if log is not None:
            log("llmfit not found; no model recommendations available")
        return []
    runtime = "mlx" if backend == "vllm-mlx" else "vllm"
    command = _recommend_command(runtime, use_case, limit)
    if log is not None:
        log(f"running model recommendations: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except OSError as exc:
        if log is not None:
            log(f"llmfit failed to start: {exc}")
        return []
    except subprocess.CalledProcessError as exc:
        if log is not None:
            detail = exc.stderr.strip() if exc.stderr else str(exc)
            log(f"llmfit recommendations failed: {detail}")
        return []
    models = parse_recommended_models(result.stdout)
    return [LlmfitRecommendation(model=model, source="llmfit") for model in models]


def parse_recommended_model(text: str) -> str | None:
    models = parse_recommended_models(text)
    return models[0] if models else None


def parse_recommended_models(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    candidates = data if isinstance(data, list) else _candidate_lists(data)
    models: list[str] = []
    for candidate in candidates:
        model = _model_from_candidate(candidate)
        if model and model not in models:
            models.append(model)
    return models


def _recommend_command(runtime: str, use_case: str, limit: int) -> list[str]:
    return ["llmfit", "recommend", "--json", "--limit", str(limit), "--use-case", use_case, "--force-runtime", runtime]


def _candidate_lists(data: object) -> list[object]:
    if not isinstance(data, dict):
        return []
    for key in ("recommendations", "models", "results", "data"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return [data]


def _model_from_candidate(candidate: object) -> str | None:
    if not isinstance(candidate, dict):
        return None
    for key in ("model", "name", "id", "model_id", "hf_model", "repo", "repo_id"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = candidate.get("model")
    if isinstance(nested, dict):
        return _model_from_candidate(nested)
    return None
