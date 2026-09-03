"""Configuration loading and project-relative paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    """Typed view of the configuration values needed by Phase 1/2."""

    random_seed: int
    target_column: str
    time_column: str
    amount_column: str
    test_size: float
    validation_size: float
    risk_thresholds: dict[str, float]
    model_params: dict[str, dict[str, Any]]
    raw_dir: Path
    processed_dir: Path
    primary_file: Path
    duplicate_handling: str


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load the YAML configuration without relying on the current directory."""
    config_path = path or DEFAULT_CONFIG_PATH
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    data = raw["data"]
    split = raw["split"]
    return ProjectConfig(
        random_seed=int(raw["project"]["random_seed"]),
        target_column=str(data["target_column"]),
        time_column=str(data["time_column"]),
        amount_column=str(data["amount_column"]),
        test_size=float(split["test_size"]),
        validation_size=float(split["validation_size"]),
        risk_thresholds={key: float(value) for key, value in raw["risk_thresholds"].items()},
        model_params=dict(raw["models"]),
        raw_dir=_project_path(data["raw_dir"]),
        processed_dir=_project_path(data["processed_dir"]),
        primary_file=_project_path(data["primary_file"]),
        duplicate_handling=str(raw.get("data_quality", {}).get("duplicate_handling", "retain")),
    )
