"""Bind ``(condition, dataset, model)`` tuples into concrete experiment cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentCell:
    """A single cell of the experiment matrix."""

    condition: str
    dataset: str
    model: str
    seed: int
    n_samples: int
    extra: dict[str, Any]


def expand_matrix(cfg: dict[str, Any]) -> list[ExperimentCell]:
    """Expand an experiment config (e.g. ``full_matrix.yaml``) into cells.

    Loads each condition / dataset / model YAML from the ``configs/`` directory
    and stores the resolved dicts in ``cell.extra`` so the runner can retrieve
    them without knowing the project layout.

    Env-var placeholders of the form ``${oc.env:VAR,default}`` in the loaded
    sub-configs are resolved at expansion time.
    """
    import re
    import os
    from pathlib import Path

    import yaml

    def _find_configs_root() -> Path:
        if "_configs_root" in cfg:
            return Path(cfg["_configs_root"])
        cwd = Path.cwd()
        for p in [cwd, *cwd.parents]:
            if (p / "pyproject.toml").exists():
                return p / "configs"
        return cwd / "configs"

    def _resolve_env(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _resolve_env(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve_env(v) for v in obj]
        if isinstance(obj, str):
            def _sub(m: re.Match[str]) -> str:
                parts = m.group(1).split(",", 1)
                return os.environ.get(parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
            return re.sub(r"\$\{oc\.env:([^}]+)\}", _sub, obj)
        return obj

    def _load_sub(kind: str, name: str) -> dict[str, Any]:
        path = configs_root / kind / f"{name}.yaml"
        if not path.exists():
            return {"name": name}
        with path.open(encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return _resolve_env(raw)

    configs_root = _find_configs_root()

    conditions: list[str] = list(cfg.get("conditions", []))
    datasets: list[str] = list(cfg.get("datasets", []))
    models: list[str] = list(cfg.get("models", []))
    seeds: list[int] = [int(s) for s in cfg.get("seeds", [cfg.get("seed", 42)])]
    n_samples: int = int(cfg.get("n_samples", 1))

    cells: list[ExperimentCell] = []
    for condition in conditions:
        cond_cfg = _load_sub("conditions", condition)
        for dataset in datasets:
            ds_cfg = _load_sub("datasets", dataset)
            for model in models:
                model_cfg = _load_sub("models", model)
                for seed in seeds:
                    cells.append(
                        ExperimentCell(
                            condition=condition,
                            dataset=dataset,
                            model=model,
                            seed=seed,
                            n_samples=n_samples,
                            extra={
                                "condition_cfg": cond_cfg,
                                "dataset_cfg": ds_cfg,
                                "model_cfg": model_cfg,
                            },
                        )
                    )
    return cells
