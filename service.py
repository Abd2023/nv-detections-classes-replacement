from __future__ import annotations

from src.executors import EXECUTORS


def get_executor(name: str):
    return EXECUTORS[name]


def bootstrap_all() -> dict[str, dict[str, str]]:
    return {name: executor.bootstrap() for name, executor in EXECUTORS.items()}


if __name__ == "__main__":
    for executor_name, bootstrap_data in bootstrap_all().items():
        print(f"{executor_name}: {bootstrap_data['status']}")

