from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from waterint.angle_z import run_angle_z
from waterint.config import load_config
from waterint.density import run_density
from waterint.hbond import run_hbond
from waterint.sfg import run_sfg


RUNNERS = {
    "density": run_density,
    "angle-z": run_angle_z,
    "oh-orientation": run_angle_z,
    "hbond": run_hbond,
    "sfg": run_sfg,
}


@dataclass(frozen=True)
class BatchTask:
    name: str
    module: str
    config_path: Path
    config_overrides: dict[str, Any]
    source_index: int


@dataclass(frozen=True)
class BatchTaskResult:
    name: str
    module: str
    config_path: Path
    status: str
    elapsed_s: float
    artifacts: list[str]
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    config_path: Path
    tasks: list[BatchTaskResult]
    summary_path: Path | None

    @property
    def failed(self) -> list[BatchTaskResult]:
        return [task for task in self.tasks if task.status == "failed"]


def load_batch_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Batch config must be a YAML mapping: {config_path}")
    data["_config_path"] = str(config_path.resolve())
    data["_config_dir"] = str(config_path.resolve().parent)
    return data


def expand_batch_tasks(batch_config: dict[str, Any]) -> list[BatchTask]:
    tasks_raw = batch_config.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("Batch config requires a non-empty tasks list.")
    base_dir = Path(str(batch_config["_config_dir"]))
    global_defaults = batch_config.get("defaults", {})
    if global_defaults is None:
        global_defaults = {}
    if not isinstance(global_defaults, dict):
        raise ValueError("batch.defaults must be a mapping when set.")

    tasks: list[BatchTask] = []
    for index, raw_task in enumerate(tasks_raw, start=1):
        if not isinstance(raw_task, dict):
            raise ValueError(f"tasks[{index}] must be a mapping.")
        tasks.extend(_expand_one_task(raw_task, index, base_dir, global_defaults))
    return tasks


def run_batch(
    batch_config_path: str | Path,
    *,
    dry_run: bool = False,
    continue_on_error: bool | None = None,
) -> BatchResult:
    batch_config = load_batch_config(batch_config_path)
    tasks = expand_batch_tasks(batch_config)
    if continue_on_error is None:
        continue_on_error = bool(batch_config.get("continue_on_error", False))
    summary_path = _summary_path(batch_config)

    results: list[BatchTaskResult] = []
    for task in tasks:
        if dry_run:
            results.append(
                BatchTaskResult(
                    name=task.name,
                    module=task.module,
                    config_path=task.config_path,
                    status="dry_run",
                    elapsed_s=0.0,
                    artifacts=[],
                )
            )
            continue
        start = time.perf_counter()
        try:
            result = _run_task(task)
            elapsed = time.perf_counter() - start
            results.append(
                BatchTaskResult(
                    name=task.name,
                    module=task.module,
                    config_path=task.config_path,
                    status="complete",
                    elapsed_s=elapsed,
                    artifacts=[str(path) for path in _result_artifacts(result)],
                )
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            results.append(
                BatchTaskResult(
                    name=task.name,
                    module=task.module,
                    config_path=task.config_path,
                    status="failed",
                    elapsed_s=elapsed,
                    artifacts=[],
                    error=str(exc),
                )
            )
            if not continue_on_error:
                break

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _write_summary(summary_path, batch_config, results, dry_run=dry_run)
    return BatchResult(config_path=Path(batch_config["_config_path"]), tasks=results, summary_path=summary_path)


def _expand_one_task(
    raw_task: dict[str, Any],
    index: int,
    base_dir: Path,
    global_defaults: dict[str, Any],
) -> list[BatchTask]:
    module = str(raw_task.get("module", global_defaults.get("module", ""))).strip()
    if module not in RUNNERS:
        supported = ", ".join(sorted(RUNNERS))
        raise ValueError(f"tasks[{index}].module must be one of: {supported}.")

    task_defaults = _merged_mapping(global_defaults.get(module, {}), raw_task.get("defaults", {}))
    task_defaults = _merged_mapping(global_defaults.get("task", {}), task_defaults)
    shared_overrides = _merged_mapping(global_defaults.get("overrides", {}), raw_task.get("overrides", {}))

    repeat = raw_task.get("repeat")
    if repeat is None:
        name = str(raw_task.get("name", f"{index:03d}_{module}"))
        config_path = _resolve_config_path(raw_task.get("config"), base_dir, index=index)
        return [
            BatchTask(
                name=name,
                module=module,
                config_path=config_path,
                config_overrides=shared_overrides,
                source_index=index,
            )
        ]

    if not isinstance(repeat, list) or not repeat:
        raise ValueError(f"tasks[{index}].repeat must be a non-empty list when set.")
    expanded: list[BatchTask] = []
    for repeat_index, item in enumerate(repeat, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"tasks[{index}].repeat[{repeat_index}] must be a mapping.")
        merged = _merged_mapping(task_defaults, item)
        name_template = str(raw_task.get("name", f"{index:03d}_{module}_{{id}}"))
        context = {key: value for key, value in merged.items() if isinstance(value, (str, int, float))}
        context.setdefault("index", repeat_index)
        context.setdefault("module", module)
        if "id" not in context:
            context["id"] = repeat_index
        name = _format_template(name_template, context)
        config_path = _resolve_config_path(merged.get("config", raw_task.get("config")), base_dir, index=index)
        item_overrides = _merged_mapping(shared_overrides, merged.get("overrides", {}))
        expanded.append(
            BatchTask(
                name=name,
                module=module,
                config_path=config_path,
                config_overrides=item_overrides,
                source_index=index,
            )
        )
    return expanded


def _run_task(task: BatchTask) -> Any:
    config = load_config(task.config_path)
    if task.config_overrides:
        config = _deep_merge(config, task.config_overrides)
        config["_config_path"] = str(task.config_path.resolve())
        config["_config_dir"] = str(task.config_path.resolve().parent)
    return RUNNERS[task.module](config)


def _resolve_config_path(raw_path: Any, base_dir: Path, *, index: int) -> Path:
    if raw_path in {None, ""}:
        raise ValueError(f"tasks[{index}].config is required.")
    path = Path(str(raw_path)).expanduser()
    return path if path.is_absolute() else base_dir / path


def _merged_mapping(*mappings: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in mappings:
        if mapping is None:
            continue
        if not isinstance(mapping, dict):
            raise ValueError("Expected a mapping.")
        result = _deep_merge(result, mapping)
    return result


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _format_template(template: str, context: dict[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        raise ValueError(f"Unknown task name template key: {exc.args[0]}") from exc


def _result_artifacts(result: Any) -> list[Path]:
    paths: list[Path] = []
    for attr in ("csv_path", "raw_csv_path", "png_path", "metadata_path"):
        value = getattr(result, attr, None)
        if value:
            paths.append(Path(value))
    for attr in ("csv_paths", "png_paths", "cf_paths", "ft_paths"):
        mapping = getattr(result, attr, None)
        if isinstance(mapping, dict):
            paths.extend(Path(value) for value in mapping.values())
    return paths


def _summary_path(batch_config: dict[str, Any]) -> Path | None:
    output_cfg = batch_config.get("output", {})
    if output_cfg is None:
        return None
    if not isinstance(output_cfg, dict):
        raise ValueError("batch.output must be a mapping when set.")
    raw_path = output_cfg.get("summary", "batch_summary.json")
    if raw_path in {None, False, ""}:
        return None
    path = Path(str(raw_path)).expanduser()
    return path if path.is_absolute() else Path(str(batch_config["_config_dir"])) / path


def _write_summary(
    path: Path,
    batch_config: dict[str, Any],
    results: list[BatchTaskResult],
    *,
    dry_run: bool,
) -> None:
    payload = {
        "batch_config": str(batch_config["_config_path"]),
        "dry_run": dry_run,
        "tasks_total": len(results),
        "tasks_complete": sum(1 for result in results if result.status == "complete"),
        "tasks_failed": sum(1 for result in results if result.status == "failed"),
        "tasks": [
            {
                "name": result.name,
                "module": result.module,
                "config": str(result.config_path),
                "status": result.status,
                "elapsed_s": result.elapsed_s,
                "artifacts": result.artifacts,
                "error": result.error,
            }
            for result in results
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
