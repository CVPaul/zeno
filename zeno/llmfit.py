from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class LlmfitRecommendation:
    model: str
    source: str


def recommend_model(backend: str, use_case: str = "coding") -> LlmfitRecommendation | None:
    if shutil.which("llmfit") is None:
        return None
    runtime = "mlx" if backend == "vllm-mlx" else "vllm"
    command = ["llmfit", "recommend", "--json", "--limit", "1", "--use-case", use_case, "--force-runtime", runtime]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    model = parse_recommended_model(result.stdout)
    if model is None:
        return None
    return LlmfitRecommendation(model=model, source="llmfit")


def parse_recommended_model(text: str) -> str | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    candidates = data if isinstance(data, list) else _candidate_lists(data)
    for candidate in candidates:
        model = _model_from_candidate(candidate)
        if model:
            return model
    return None


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
