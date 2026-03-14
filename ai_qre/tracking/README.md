# tracking

This folder provides **experiment tracking**: persisting run metadata, parameters, metrics, and artifacts to a directory so you can reproduce and compare research runs.

---

## Files

### `experiment.py`

- **`_to_jsonable(value)`**
  Helper that recurses through dataclasses, dicts, lists, tuples, and NumPy-like arrays (`.tolist()`) to produce a JSON-serializable structure. Used when writing params, metrics, and artifacts.

- **`ExperimentRun`** (dataclass)
  Represents a single run. Fields: `run_id`, `name`, `created_at`, `root` (Path), `params`, `metrics`, `tags` (all mutable where applicable).

  - **`log_params(params: dict)`**: Merges into `self.params`, then overwrites `root/params.json` with the full params (after `_to_jsonable`).
  - **`log_metrics(metrics: dict)`**: Same for metrics; overwrites `root/metrics.json`.
  - **`log_artifact_text(name, content: str)`**: Writes `content` to `root/artifacts/<name>`.
  - **`log_artifact_json(name, payload: dict)`**: Writes `_to_jsonable(payload)` as JSON to `root/artifacts/<name>`.
  - **`finalize()`**: Writes `root/summary.json` with run_id, name, created_at, params, metrics, tags. Called automatically when the run is created and can be called again before closing the run.
  - **`_write_json(name, payload)`**: Ensures parent dir exists and writes JSON to `root/<name>`.

- **`ExperimentTracker`**
  Creates and manages run directories.
  - **Constructor**: `ExperimentTracker(root_dir="runs")`. Creates `root_dir` if needed.
  - **`start_run(name, tags=None) -> ExperimentRun`**:
    - Generates a run id: `{YYYYMMDDTHHMMSSZ}-{first 10 chars of sha1(name-created_at)}`.
    - Creates directory `root_dir / run_id`.
    - Builds an `ExperimentRun` with that root and the given name/tags.
    - Calls `run.finalize()` so an initial `summary.json` exists.
    - Returns the run so the caller can call `log_params`, `log_metrics`, `log_artifact_*`, and optionally `finalize()` again.

**Directory layout per run**:

- `params.json`, `metrics.json`, `summary.json` in the run root.
- `artifacts/<name>` for each artifact (text or JSON).

**Exposed in**: `ResearchExtensions.experiments` and `ai_qre.tracking.__init__` (`ExperimentRun`, `ExperimentTracker`).
