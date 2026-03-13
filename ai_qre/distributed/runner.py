import multiprocessing as mp
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

TTask = TypeVar("TTask")
TResult = TypeVar("TResult")


class DistributedResearchRunner:
    def __init__(self, workers: int | None = None, chunksize: int = 1) -> None:
        self.workers = int(workers or mp.cpu_count())
        self.chunksize = int(chunksize)

    def run(
        self, func: Callable[[TTask], TResult], tasks: Sequence[TTask]
    ) -> list[TResult]:
        if not tasks:
            return []
        with mp.Pool(self.workers) as pool:
            return list(pool.map(func, list(tasks), chunksize=self.chunksize))

    def starmap(
        self, func: Callable[..., TResult], tasks: Sequence[tuple[Any, ...]]
    ) -> list[TResult]:
        if not tasks:
            return []
        with mp.Pool(self.workers) as pool:
            return list(
                pool.starmap(func, list(tasks), chunksize=self.chunksize)
            )
