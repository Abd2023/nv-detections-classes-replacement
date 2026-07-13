from __future__ import annotations

from src.executors.package_init_placeholder_executor import PackageInitPlaceholderExecutor


EXECUTORS = {
    "PackageInitPlaceholderExecutor": PackageInitPlaceholderExecutor,
}

__all__ = ["EXECUTORS", "PackageInitPlaceholderExecutor"]

