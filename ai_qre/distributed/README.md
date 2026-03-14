# distributed

This folder provides **parallel execution** for research: run the same function on many tasks (e.g. parameter sweeps, multiple backtests) using multiprocessing.

---

## Files

### `runner.py`

Single class: **`DistributedResearchRunner`**, a thin wrapper around Python’s `multiprocessing.Pool`.

- **Constructor**: `DistributedResearchRunner(workers=None, chunksize=1)`.

  - `workers`: number of processes; default `None` means `multiprocessing.cpu_count()`.
  - `chunksize`: number of tasks per chunk sent to workers (can reduce overhead for many small tasks).

- **`run(func, tasks) -> list[TResult]`**

  - `func`: callable that takes a **single** argument (one task).
  - `tasks`: sequence of tasks (e.g. list of configs or ticker lists).
  - Uses `pool.map(func, tasks, chunksize=self.chunksize)` and returns the list of results in order.
  - Returns `[]` if `tasks` is empty.

- **`starmap(func, tasks) -> list[TResult]`**
  - `func`: callable that takes **multiple** arguments (one per element of each tuple).
  - `tasks`: sequence of **tuples** (e.g. `[(arg1, arg2), ...]`).
  - Uses `pool.starmap(func, tasks, chunksize=self.chunksize)` and returns the list of results.
  - Returns `[]` if `tasks` is empty.

**Usage**: Build a list of tasks (e.g. different alpha configs or date ranges), pass a function that runs one backtest or one optimization, and call `run` or `starmap` to execute in parallel. The function must be picklable and runnable in a worker process (avoid lambdas that capture non-picklable state if needed).

**Exposed in**: `ResearchExtensions.distributed`.
