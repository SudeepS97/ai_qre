"""Experiment tracking: run directory, params, metrics, artifacts, summary.json."""

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(inner_value) for inner_value in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@dataclass
class ExperimentRun:
    """Single run: log params, metrics, artifacts; finalize writes summary.json."""

    run_id: str
    name: str
    created_at: str
    root: Path
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def log_params(self, params: dict[str, Any]) -> None:
        """Merge params and overwrite params.json."""
        self.params.update(_to_jsonable(params))
        self._write_json("params.json", self.params)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge metrics and overwrite metrics.json."""
        self.metrics.update(_to_jsonable(metrics))
        self._write_json("metrics.json", self.metrics)

    def log_artifact_text(self, name: str, content: str) -> None:
        """Write text content to root/artifacts/name."""
        artifact_path = self.root / "artifacts" / name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(content, encoding="utf-8")

    def log_artifact_json(self, name: str, payload: dict[str, Any]) -> None:
        """Write JSON payload to root/artifacts/name."""
        artifact_path = self.root / "artifacts" / name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(_to_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def finalize(self) -> None:
        """Write summary.json with run_id, name, created_at, params, metrics, tags."""
        summary = {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
        }
        self._write_json("summary.json", summary)

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        output_path = self.root / name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_to_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class ExperimentTracker:
    """Creates run directories and ExperimentRun instances; start_run returns a run and calls finalize once."""

    def __init__(self, root_dir: str = "runs") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def start_run(
        self, name: str, tags: dict[str, str] | None = None
    ) -> ExperimentRun:
        """Create run directory (root_dir/{timestamp}-{token}), return ExperimentRun with initial summary.json."""
        created_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        token = hashlib.sha1(
            f"{name}-{created_at}".encode("utf-8")
        ).hexdigest()[:10]
        run_id = f"{created_at}-{token}"
        root = self.root_dir / run_id
        root.mkdir(parents=True, exist_ok=True)
        run = ExperimentRun(
            run_id=run_id,
            name=name,
            created_at=created_at,
            root=root,
            tags=dict(tags or {}),
        )
        run.finalize()
        return run
